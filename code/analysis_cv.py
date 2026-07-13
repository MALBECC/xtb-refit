#!/usr/bin/env python3
"""Generalization / extrapolation analysis for the GFN2-xTB re-fit.

Runs two INDEPENDENT cross-validation studies (each re-fits from scratch on a train
split and measures held-out error with the fitted params). Completely separate from
the production run — reads the same ORCA data (config.ENGRAD_DIR) but writes only
under fit/<SYSTEM>/analysis/ and never touches the production param file.

  (A) random_split  — all (node x replica) structures pooled, split 80/20 at RANDOM.
      Replicas of a given node are scattered across train & val, so every node is seen
      in training. Measures INTERPOLATION: predicting unseen thermal snapshots of
      reaction-coordinate positions the model has already seen.

  (B) node_cv       — K-fold over the 48 reaction-coordinate NODES. Each fold trains on
      the other nodes (ALL their replicas) and predicts the held-out block of nodes,
      with every replica of a held-out node kept out of training. Concatenating the
      folds gives a HELD-OUT ERROR FOR EVERY NODE = a map of where the fit extrapolates
      well vs where more DFT sampling is needed. Measures EXTRAPOLATION.

Energy is scored the way the objective/plots do it: relative (mean-centered) per
replica-curve over that replica's full node profile, so the reference is stable and a
held-out node's error is its deviation from the trained reaction profile. Charges and
force components are per-atom absolute errors.

Knobs (env-overridable; sensible HPC defaults):
  XTBFIT_ANALYSIS_POP/_ITERS/_NCPUS   GA size / cores for the analysis fits
  XTBFIT_ANALYSIS_SEED                random-split RNG seed (default 42)
  XTBFIT_ANALYSIS_RAND_REPLICAS       replicas used for the random split (default 1-20)
  XTBFIT_ANALYSIS_NODE_REPLICAS       replicas used for node-CV      (default 1-10)
  XTBFIT_ANALYSIS_KFOLDS              node-CV folds (default 6 -> 8 nodes/fold)

Usage:  python analysis_cv.py            (runs both studies; resumable — skips done fits)
        python analysis_cv.py random     (only random split)
        python analysis_cv.py nodecv     (only per-node CV)
"""
import os, sys, json, time, glob
import numpy as np
import common as K            # sets fork/OMP/SCF env before xtb import
import config as C
import assemble_dataset as A  # reuse find_compact / parse_one (importing does not run main)
import plot_lib as P          # reuse r2/rmse + colors

from algorithms import DDGeneticAlgorithm as DDGA
from objectivefunctions import ECGFittingV3Json
from parameters import parm_HCONS, XTBParam


# ── knobs ─────────────────────────────────────────────────────────────────────
def _reps(name, default):
    v = os.environ.get(name)
    if not v:
        return default
    out = []
    for tok in v.replace(",", " ").split():
        if "-" in tok:
            a, b = tok.split("-"); out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return sorted(set(out))

A_POP    = int(os.environ.get("XTBFIT_ANALYSIS_POP", 40))
A_ITERS  = int(os.environ.get("XTBFIT_ANALYSIS_ITERS", 40))
A_NCPUS  = int(os.environ.get("XTBFIT_ANALYSIS_NCPUS", C.N_CPUS))
SEED     = int(os.environ.get("XTBFIT_ANALYSIS_SEED", 42))
RAND_REPS = _reps("XTBFIT_ANALYSIS_RAND_REPLICAS", list(range(1, 21)))   # f1..f20
NODE_REPS = _reps("XTBFIT_ANALYSIS_NODE_REPLICAS", list(range(1, 11)))   # f1..f10
KFOLDS   = int(os.environ.get("XTBFIT_ANALYSIS_KFOLDS", 6))
VAL_FRAC = 0.20

OUT   = os.environ.get("XTBFIT_ANALYSIS_OUT", os.path.join(C.FIT_DIR, "analysis"))
DEFDIR = os.path.join(OUT, "default_param")   # stock GFN2 param dir (shared)


# ── data loading (same exclusions as production) ────────────────────────────────
def load_records():
    files = A.find_compact(C.ENGRAD_DIR)
    recs = [A.parse_one(p) for p in files]
    def excluded(r):
        tok = f"node{r['node']}_f{r['fidx']}"
        return any(x == tok or x == r["label"] for x in C.EXCLUDE)
    recs = [r for r in recs if not excluded(r)]
    return recs


def write_curve(recs_for_fidx, out_path):
    """One replica's (subset of) nodes -> a dataset_f*.json curve (nodes in order)."""
    nodes = sorted(recs_for_fidx, key=lambda r: r["node"])
    ds = {
        "energies":       [r["energy_kcal"] for r in nodes],
        "atomic_numbers": {r["label"]: r["Z"] for r in nodes},
        "coordinates":    {r["label"]: r["coords_bohr"] for r in nodes},
        "charges":        {r["label"]: r["charges"] for r in nodes},
        "gradients":      {r["label"]: r["gradient"] for r in nodes},
        "system": {"chrg": {r["label"]: r["chrg"] for r in nodes},
                   "mult": {r["label"]: r["mult"] for r in nodes}},
    }
    json.dump(ds, open(out_path, "w"))


def write_train_curves(train_recs, work_dir):
    """Group training records by replica -> curve files. A curve needs >=2 nodes for the
    relative-energy objective (npoints>1)."""
    os.makedirs(work_dir, exist_ok=True)
    for f in glob.glob(os.path.join(work_dir, "dataset_f*.json")):
        os.remove(f)
    byf = {}
    for r in train_recs:
        byf.setdefault(r["fidx"], []).append(r)
    paths = []
    for fidx in sorted(byf):
        if len(byf[fidx]) < 2:
            continue
        p = os.path.join(work_dir, f"dataset_f{fidx}.json")
        write_curve(byf[fidx], p)
        paths.append(p)
    return paths


# ── the fit (mirrors code/fit.py; lighter GA by default) ────────────────────────
# The DDGA parallelises with multiprocessing.Pool.apply_async(self.f, ...), which
# PICKLES the objective. A local closure can't be pickled, so — exactly like fit.py —
# the objective is module-level and reads a module global set just before each run.
# (Pool uses the fork start method, so workers inherit _FUNCT set in the parent.)
_FUNCT = None


def _objective(inp):
    return _FUNCT.evaluate(solution=inp[0], thr_id=inp[1])


def run_fit(train_paths, fit_dir, pop, iters, n_cpus, tag=""):
    global _FUNCT
    os.makedirs(fit_dir, exist_ok=True)
    param_out = os.path.join(fit_dir, "param_gfn2-xtb.txt")
    if os.path.exists(param_out):
        print(f"[{tag}] param file exists -> skip fit ({param_out})", flush=True)
        return np.loadtxt(os.path.join(fit_dir, "best_solution.txt")), None, None, 0.0
    work = os.path.join(fit_dir, "ga_work"); os.makedirs(work, exist_ok=True)
    orig = K.default_gfn2_params()
    cvo = [[True, True, True] for _ in train_paths]
    funct = ECGFittingV3Json(patterns=C.PATTERNS, parm=parm_HCONS, paths=train_paths,
                             cv_objectives=cvo, out_folder=work, **C.OBJ)
    _FUNCT = funct
    s0 = np.array(funct.evaluate(solution=list(orig), thr_id=0))
    print(f"[{tag}] fitting {len(train_paths)} curves  baseline SCORE={s0[0]:.3e}", flush=True)
    vb = np.array([[v - C.BOUNDS_FRAC * abs(v), v + C.BOUNDS_FRAC * abs(v)] for v in orig])
    ap = {'max_num_iteration': iters, 'population_size': pop, 'mutation_probability': 0.9,
          'elit_ratio': 0.15, 'crossover_probability': 0.95, 'parents_portion': 0.4,
          'crossover_type': 'uniform', 'max_iteration_without_improv': C.MNIWI}
    model = DDGA(function=_objective, dimension=75, variable_boundaries=vb,
                 algorithm_parameters=ap, convergence_curve=False, progress_bar=False)
    t = time.time()
    model.run(guess=list(orig), log_path=os.path.join(fit_dir, "ga_logger.log"), n_cpus=n_cpus)
    dt = time.time() - t
    best = np.array(model.best_variable)
    XTBParam(parm_HCONS, C.PATTERNS, list(best)).print_param_file(param_out)
    np.savetxt(os.path.join(fit_dir, "best_solution.txt"), best, header="75 optimized GFN2 params")
    with open(os.path.join(fit_dir, "fit_summary.txt"), "w") as f:
        f.write(f"tag {tag}\nbaseline SCORE {s0[0]:.4e} -> best SCORE {float(model.best_function):.4e}\n")
        f.write(f"train_curves {len(train_paths)} POP {pop} ITERS {iters} N_CPUS {n_cpus} "
                f"bounds {C.BOUNDS_FRAC} etemp {C.XTB_ETEMP} ref_ener {C.OBJ['ref_ener']}\n")
        f.write(f"time_s {dt:.0f}\n")
    print(f"[{tag}] DONE fit SCORE {s0[0]:.3e} -> {float(model.best_function):.3e} in {dt:.0f}s", flush=True)
    return best, float(s0[0]), float(model.best_function), dt


# ── parallel xtb evaluation ─────────────────────────────────────────────────────
def _eval_one(task):
    label, Z, coords_A, chrg, mult, param_dir = task
    import numpy as _np
    import common as _K
    e, q, g = _K.xtb_sp(_np.asarray(Z), _np.asarray(coords_A, float), chrg, mult, param_dir)
    if e is None:
        return label, None, None, None
    return label, float(e) * _K.H2KCAL, _np.asarray(q, float), _np.asarray(g, float)


def evaluate(records, replicas, param_dir, n_cpus):
    """Run xtb over the FULL node set of each given replica. Returns {label: {e,q,g}}."""
    from multiprocessing import Pool
    tasks = []
    for r in records:
        if r["fidx"] in replicas:
            coords_A = (np.asarray(r["coords_bohr"], float) / C.ANGSTROM_TO_BOHR).tolist()
            tasks.append((r["label"], r["Z"], coords_A, r["chrg"], r["mult"], param_dir))
    out = {}
    with Pool(n_cpus) as pool:
        for label, e, q, g in pool.imap_unordered(_eval_one, tasks, chunksize=8):
            out[label] = {"e": e, "q": q, "g": g}
    return out


def structure_rows(records, replicas, pred_ref, pred_def):
    """Per-structure error record. Energy mean-centered per replica over its full profile
    (nan-safe). Charge/force are per-atom arrays. force = -gradient."""
    by_f = {}
    for r in records:
        if r["fidx"] in replicas:
            by_f.setdefault(r["fidx"], []).append(r)
    rows = []
    for fidx, rs in by_f.items():
        rs = sorted(rs, key=lambda r: r["node"])
        labels = [r["label"] for r in rs]
        edft = np.array([r["energy_kcal"] for r in rs], float)
        eref = np.array([pred_ref[l]["e"] if pred_ref[l]["e"] is not None else np.nan for l in labels])
        edef = np.array([pred_def[l]["e"] if pred_def[l]["e"] is not None else np.nan for l in labels])
        mc = lambda a: a - np.nanmean(a)
        mE_dft, mE_ref, mE_def = mc(edft), mc(eref), mc(edef)
        for i, r in enumerate(rs):
            l = r["label"]
            qd = np.asarray(r["charges"], float)
            gd = np.asarray(r["gradient"], float)                    # DFT gradient (Eh/Bohr)
            qr, gr = pred_ref[l]["q"], pred_ref[l]["g"]
            qf, gf = pred_def[l]["q"], pred_def[l]["g"]
            rows.append({
                "fidx": fidx, "node": r["node"], "label": l,
                "dE_dft": float(mE_dft[i]), "dE_ref": float(mE_ref[i]), "dE_def": float(mE_def[i]),
                "q_dft": qd, "q_ref": qr, "q_def": qf,
                "f_dft": -gd, "f_ref": (None if gr is None else -gr), "f_def": (None if gf is None else -gf),
            })
    return rows


# ── aggregation helpers ─────────────────────────────────────────────────────────
def _pool_qf(rows, which):
    """Concatenate per-atom charge and force-component values for parity/MAE.
    which in {'ref','def'}. Returns (q_dft, q_pred, f_dft, f_pred) 1-D arrays over finite."""
    qd, qp, fd, fp = [], [], [], []
    for r in rows:
        pq, pf = r["q_" + which], r["f_" + which]
        if pq is not None:
            qd.extend(r["q_dft"]); qp.extend(pq)
        if pf is not None:
            fd.extend(r["f_dft"].ravel()); fp.extend(pf.ravel())
    return (np.array(qd), np.array(qp), np.array(fd), np.array(fp))


def _mae(err, k=10.0):
    """Robust MAE of an error array: drop non-finite AND gross outliers (|e-median| >
    k*MAD, k=10 so only truly pathological points are skipped — per user instruction to
    'skip points that look off'). Returns (trimmed_mean, median, n_skipped)."""
    err = np.asarray(err, float)
    err = err[np.isfinite(err)]
    if err.size == 0:
        return float("nan"), float("nan"), 0
    med = np.median(err)
    mad = np.median(np.abs(err - med))
    if mad <= 0:
        return float(np.mean(err)), float(med), 0
    inl = np.abs(err - med) <= k * mad
    return float(np.mean(err[inl])), float(med), int((~inl).sum())


def maes_on(rows):
    """Pooled default/refit error stats over a set of rows for energy (rel, kcal/mol),
    charge (e), force (Eh/Bohr). Each metric/series is (trimmed_mean, median, n_skipped),
    where the trim skips gross outliers. Energy uses the mean-centered per-curve values."""
    eE_ref = [abs(r["dE_ref"] - r["dE_dft"]) for r in rows]
    eE_def = [abs(r["dE_def"] - r["dE_dft"]) for r in rows]
    qd_r, qp_r, fd_r, fp_r = _pool_qf(rows, "ref")
    qd_d, qp_d, fd_d, fp_d = _pool_qf(rows, "def")
    return {
        "E": {"def": _mae(eE_def), "ref": _mae(eE_ref)},
        "q": {"def": _mae(np.abs(qp_d - qd_d)), "ref": _mae(np.abs(qp_r - qd_r))},
        "F": {"def": _mae(np.abs(fp_d - fd_d)), "ref": _mae(np.abs(fp_r - fd_r))},
    }


def per_node(rows):
    """Per-node held-out MAEs. Returns dict node -> {n, E_def,E_ref,q_def,q_ref,F_def,F_ref}."""
    byn = {}
    for r in rows:
        byn.setdefault(r["node"], []).append(r)
    out = {}
    for n, rs in byn.items():
        m = maes_on(rs)
        rec = {"n": len({r["fidx"] for r in rs})}
        for key in ("E", "q", "F"):
            rec[f"{key}_def"] = m[key]["def"][0]       # trimmed mean
            rec[f"{key}_ref"] = m[key]["ref"][0]
            rec[f"{key}_ref_med"] = m[key]["ref"][1]   # robust median
            rec[f"{key}_skip"] = m[key]["ref"][2]
        out[n] = rec
    return out


# ── plots ────────────────────────────────────────────────────────────────────────
def parity_fig(rows, out_png, suptitle):
    import matplotlib.pyplot as plt
    RED, GRN = P.RED, P.GRN
    Ed = np.array([r["dE_dft"] for r in rows]); Er = np.array([r["dE_ref"] for r in rows])
    Ef = np.array([r["dE_def"] for r in rows])
    qd_r, qp_r, fd_r, fp_r = _pool_qf(rows, "ref")
    qd_d, qp_d, fd_d, fp_d = _pool_qf(rows, "def")

    def panel(ax, act, dflt, ref, unit, title, ms, alpha):
        lo = np.nanmin([act.min(), dflt.min(), ref.min()]); hi = np.nanmax([act.max(), dflt.max(), ref.max()])
        pad = 0.05 * (hi - lo); lim = [lo - pad, hi + pad]
        ax.plot(lim, lim, 'k--', lw=1, alpha=.7)
        ax.scatter(act, dflt, s=ms, c=RED, alpha=alpha, lw=0,
                   label=f"default  R²={P.r2(act,dflt):.3f} RMSE={P.rmse(act,dflt):.3g}")
        ax.scatter(act, ref, s=ms, c=GRN, alpha=alpha, lw=0, marker='^',
                   label=f"re-fit   R²={P.r2(act,ref):.3f} RMSE={P.rmse(act,ref):.3g}")
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
        ax.set_xlabel(f"reference / {unit}"); ax.set_ylabel(f"GFN2-xTB / {unit}")
        ax.set_title(title, fontsize=11); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="upper left")

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 6.2))
    panel(ax[0], Ed, Ef, Er, "kcal mol$^{-1}$", "Energy (relative, per curve)", 26, .7)
    panel(ax[1], fd_r, fd_d, fp_r, "E$_h$ Bohr$^{-1}$", "Force components (all atoms)", 4, .25)
    panel(ax[2], qd_r, qp_d, qp_r, "e", "Mulliken charges (all atoms)", 8, .4)
    fig.suptitle(suptitle, fontsize=13)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.90, bottom=0.13, wspace=0.28)
    fig.savefig(out_png, dpi=150); plt.close(fig)


def node_map_fig(pn, out_png, suptitle):
    import matplotlib.pyplot as plt
    RED, GRN = P.RED, P.GRN
    nodes = sorted(pn)
    def col(key): return np.array([pn[n][key] for n in nodes])
    fig, ax = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    for a, (kd, kr, lab, unit) in zip(ax, [
            ("E_def", "E_ref", "energy (rel)", "kcal mol$^{-1}$"),
            ("q_def", "q_ref", "Mulliken charge", "e"),
            ("F_def", "F_ref", "force component", "E$_h$ Bohr$^{-1}$")]):
        d, r = col(kd), col(kr)
        a.plot(nodes, d, 's--', color=RED, lw=1.4, ms=4, label="default")
        a.plot(nodes, r, '^-', color=GRN, lw=1.8, ms=5, label="re-fit (held-out)")
        thr = np.nanmean(r) + np.nanstd(r)
        worst = [n for n in nodes if pn[n][kr] > thr]
        for n in worst:
            a.annotate(str(n), (n, pn[n][kr]), fontsize=7, color=GRN,
                       xytext=(0, 5), textcoords="offset points", ha="center")
        a.axhline(np.nanmean(r), ls=":", color=GRN, lw=1, alpha=.7)
        a.set_ylabel(f"held-out MAE {lab}\n/ {unit}"); a.grid(alpha=.3); a.legend(fontsize=9, loc="upper right")
    ax[-1].set_xlabel("reaction-coordinate node index")
    fig.suptitle(suptitle, fontsize=13); fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=150); plt.close(fig)


# ── study A: random split ────────────────────────────────────────────────────────
def study_random(records):
    outdir = os.path.join(OUT, "random_split"); os.makedirs(outdir, exist_ok=True)
    pool = [r for r in records if r["fidx"] in set(RAND_REPS)]
    rng = np.random.RandomState(SEED)
    idx = np.arange(len(pool)); rng.shuffle(idx)
    ncut = int(round(VAL_FRAC * len(pool)))
    val_keys = {(pool[i]["node"], pool[i]["fidx"]) for i in idx[:ncut]}
    train_recs = [r for r in pool if (r["node"], r["fidx"]) not in val_keys]
    val_recs   = [r for r in pool if (r["node"], r["fidx"]) in val_keys]
    print(f"[random] replicas {RAND_REPS[0]}-{RAND_REPS[-1]} | pool {len(pool)} -> "
          f"train {len(train_recs)} / val {len(val_recs)} (seed {SEED})", flush=True)

    paths = write_train_curves(train_recs, os.path.join(outdir, "train_curves"))
    run_fit(paths, outdir, A_POP, A_ITERS, A_NCPUS, tag="random")

    reps = set(RAND_REPS)
    pred_ref = evaluate(records, reps, outdir, A_NCPUS)
    pred_def = evaluate(records, reps, DEFDIR, A_NCPUS)
    rows = structure_rows(records, reps, pred_ref, pred_def)
    vk = val_keys
    v_rows = [r for r in rows if (r["node"], r["fidx"]) in vk]
    t_rows = [r for r in rows if (r["node"], r["fidx"]) not in vk]
    mv, mt = maes_on(v_rows), maes_on(t_rows)

    parity_fig(v_rows, os.path.join(outdir, "val_parity.png"),
               f"Random 80/20 split — HELD-OUT val — {C.SYSTEM}")
    pn = per_node(v_rows)
    _write_node_csv(pn, os.path.join(outdir, "per_node_val.csv"))

    L = []
    L.append(f"RANDOM SPLIT (node-agnostic; interpolation test) — {C.SYSTEM}")
    L.append(f"replicas {RAND_REPS[0]}-{RAND_REPS[-1]}  pool {len(pool)}  train {len(train_recs)}  "
             f"val {len(val_recs)}  val_frac {VAL_FRAC}  seed {SEED}")
    L.append(f"GA POP {A_POP} ITERS {A_ITERS} bounds ±{C.BOUNDS_FRAC*100:.0f}%")
    L.append("")
    L.append("MAE = mean after skipping gross outliers (>10*MAD); (median) is fully robust.")
    L.append(f"{'quantity':10s} {'default':>10s} {'refit(train)':>13s} {'refit(val)':>12s} {'val median':>12s}  gap(val-train)")
    nskip = 0
    for k, unit in (("E", "kcal/mol"), ("q", "e"), ("F", "Eh/Bohr")):
        d, rt, rv = mt[k]["def"][0], mt[k]["ref"][0], mv[k]["ref"][0]
        rv_med = mv[k]["ref"][1]; nskip += mv[k]["ref"][2]
        L.append(f"{k:10s} {d:10.4f} {rt:13.4f} {rv:12.4f} {rv_med:12.4f}  {rv-rt:+.4f}  {unit}")
    L.append(f"(val points skipped as gross outliers across E/q/F: {nskip})")
    L.append("")
    L.append("Small train->val gap = interpolates well (safe for MD in the sampled region).")
    open(os.path.join(outdir, "summary.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    return {"train": mt, "val": mv}


# ── study B: per-node K-fold CV ───────────────────────────────────────────────────
def _node_folds(nodes, k):
    """Split nodes into k CONTIGUOUS blocks along the reaction coordinate, so each fold
    holds out a genuine gap (extrapolation test) rather than isolated interpolatable
    points. Sizes differ by at most one when 48 doesn't divide evenly."""
    nodes = sorted(nodes)
    n = len(nodes)
    base, rem = divmod(n, k)
    folds, s = [], 0
    for i in range(k):
        sz = base + (1 if i < rem else 0)
        folds.append(nodes[s:s + sz]); s += sz
    return folds


def study_nodecv(records):
    outdir = os.path.join(OUT, "node_cv"); os.makedirs(outdir, exist_ok=True)
    reps = set(NODE_REPS)
    sub = [r for r in records if r["fidx"] in reps]
    all_nodes = sorted({r["node"] for r in sub})
    folds = _node_folds(all_nodes, KFOLDS)
    print(f"[nodecv] replicas {NODE_REPS[0]}-{NODE_REPS[-1]} | {len(all_nodes)} nodes | "
          f"{KFOLDS} folds (contiguous): {[ (f[0],f[-1]) for f in folds ]}", flush=True)

    pred_def = evaluate(records, reps, DEFDIR, A_NCPUS)   # default params, once
    all_val_rows = []
    fold_info = []
    for fi, held in enumerate(folds):
        held = set(held)
        fdir = os.path.join(outdir, f"fold{fi+1}")
        train_recs = [r for r in sub if r["node"] not in held]
        paths = write_train_curves(train_recs, os.path.join(fdir, "train_curves"))
        print(f"[nodecv] fold {fi+1}/{len(folds)} hold nodes {sorted(held)}  "
              f"train {len(train_recs)} struct / {len(paths)} curves", flush=True)
        run_fit(paths, fdir, A_POP, A_ITERS, A_NCPUS, tag=f"nodecv-f{fi+1}")
        pred_ref = evaluate(records, reps, fdir, A_NCPUS)
        rows = structure_rows(records, reps, pred_ref, pred_def)
        v_rows = [r for r in rows if r["node"] in held]
        all_val_rows.extend(v_rows)
        fold_info.append((fi + 1, sorted(held), len(train_recs)))

    pn = per_node(all_val_rows)
    _write_node_csv(pn, os.path.join(outdir, "per_node_heldout.csv"))
    node_map_fig(pn, os.path.join(outdir, "per_node_error_map.png"),
                 f"Per-node HELD-OUT error (leave-node-out CV) — {C.SYSTEM}\n"
                 f"peaks = reaction-coordinate regions needing more DFT data")
    mall = maes_on(all_val_rows)

    # rank worst nodes by re-fit held-out energy error
    worst = sorted(pn, key=lambda n: pn[n]["E_ref"], reverse=True)
    L = []
    L.append(f"PER-NODE LEAVE-NODE-OUT CV (extrapolation test) — {C.SYSTEM}")
    L.append(f"replicas {NODE_REPS[0]}-{NODE_REPS[-1]}  nodes {len(all_nodes)}  folds {KFOLDS} (contiguous blocks)")
    L.append(f"GA POP {A_POP} ITERS {A_ITERS} bounds ±{C.BOUNDS_FRAC*100:.0f}%")
    L.append("")
    L.append("Overall held-out error (mean after skipping gross outliers; median in parens):")
    for k, unit in (("E", "kcal/mol"), ("q", "e"), ("F", "Eh/Bohr")):
        L.append(f"  {k}: default {mall[k]['def'][0]:.4f} -> re-fit {mall[k]['ref'][0]:.4f} "
                 f"(median {mall[k]['ref'][1]:.4f}){'':2s} {unit}  [skipped {mall[k]['ref'][2]}]")
    L.append("")
    L.append("Worst-10 nodes by held-out re-fit ENERGY error (where more DFT data would help):")
    L.append(f"  {'node':>4s} {'E_ref':>8s} {'E_def':>8s} {'q_ref':>7s} {'F_ref':>8s}")
    for n in worst[:10]:
        L.append(f"  {n:>4d} {pn[n]['E_ref']:8.3f} {pn[n]['E_def']:8.3f} {pn[n]['q_ref']:7.4f} {pn[n]['F_ref']:8.4f}")
    L.append("")
    L.append("Folds:")
    for fi, held, ntr in fold_info:
        L.append(f"  fold {fi}: held nodes {held[0]}-{held[-1]} ({len(held)} nodes), train {ntr} struct")
    open(os.path.join(outdir, "summary.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    return {"overall": mall, "worst": worst[:10]}


def _write_node_csv(pn, path):
    lines = ["node,n_replicas,E_def,E_ref,E_ref_median,q_def,q_ref,F_def,F_ref,n_skipped"]
    for n in sorted(pn):
        p = pn[n]
        lines.append(f"{n},{p['n']},{p['E_def']:.5f},{p['E_ref']:.5f},{p['E_ref_med']:.5f},"
                     f"{p['q_def']:.6f},{p['q_ref']:.6f},{p['F_def']:.6f},{p['F_ref']:.6f},"
                     f"{p['E_skip']+p['q_skip']+p['F_skip']}")
    open(path, "w").write("\n".join(lines) + "\n")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    os.makedirs(OUT, exist_ok=True)
    K.write_param_dir(K.default_gfn2_params(), DEFDIR)   # stock GFN2 params for baseline
    print(f"=== analysis start | POP {A_POP} ITERS {A_ITERS} N_CPUS {A_NCPUS} | out {OUT} ===", flush=True)
    records = load_records()
    print(f"loaded {len(records)} structures (after exclusions {C.EXCLUDE})", flush=True)
    t0 = time.time()
    if which in ("all", "random"):
        study_random(records)
    if which in ("all", "nodecv"):
        study_nodecv(records)
    print(f"=== analysis done in {(time.time()-t0)/60:.1f} min ===", flush=True)


if __name__ == "__main__":
    main()
