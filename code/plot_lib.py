"""Shared plotting/analysis helpers: run xtb (default & re-fit) over a dataset and
build the standard figures. Used by make_plots.py (train) and validate.py (val)."""
import os
import numpy as np
import common as K
import config as C

RED, GRN, BLU = "#d1495b", "#2e7d32", "#1f4e79"


def r2(a, p):
    ss = np.sum((a - p) ** 2); tot = np.sum((a - a.mean()) ** 2)
    return 1 - ss / tot if tot > 0 else np.nan

def rmse(a, p):
    return np.sqrt(np.mean((a - p) ** 2))


def collect(dset_dir, def_dir, ref_dir):
    """Run xtb (default + re-fit) over every structure in dset_dir.
    Returns dict of per-curve energy profiles and pooled q/F predictions/refs."""
    out = {"curves": [], "q_act": [], "q_def": [], "q_ref": [],
           "F_act": [], "F_def": [], "F_ref": []}
    for p in K.curve_paths(dset_dir):
        c = K.load_curve(p); order = np.argsort(c["node_idx"])
        bases = [c["bases"][i] for i in order]; nodes = np.array([c["node_idx"][i] for i in order])
        edft = c["E_dft_kcal"][order]
        edef, eref = [], []
        for b in bases:
            e0, q0, g0 = K.xtb_sp(c["Z"][b], c["coords_A"][b], c["chrg"][b], c["mult"][b], def_dir)
            e1, q1, g1 = K.xtb_sp(c["Z"][b], c["coords_A"][b], c["chrg"][b], c["mult"][b], ref_dir)
            edef.append(e0 * K.H2KCAL); eref.append(e1 * K.H2KCAL)
            out["q_act"].extend(c["q_dft"][b]); out["q_def"].extend(q0); out["q_ref"].extend(q1)
            out["F_act"].extend((-c["grad_dft"][b]).ravel())      # force = -gradient
            out["F_def"].extend((-np.array(g0)).ravel()); out["F_ref"].extend((-np.array(g1)).ravel())
        tag = os.path.basename(p).split("_")[-1].replace(".json", "")
        out["curves"].append({"tag": tag, "nodes": nodes,
                              "dft": edft, "def": np.array(edef), "ref": np.array(eref)})
    for k in ("q_act", "q_def", "q_ref", "F_act", "F_def", "F_ref"):
        out[k] = np.array(out[k])
    return out


def _mc(a):     # mean-center
    return a - a.mean()


def maes(data):
    """Pooled MAEs (energy mean-centered per curve, charge, force)."""
    eE_def, eE_ref = [], []
    for cv in data["curves"]:
        yd, ya, yr = _mc(cv["dft"]), _mc(cv["def"]), _mc(cv["ref"])
        eE_def.extend(np.abs(ya - yd)); eE_ref.extend(np.abs(yr - yd))
    dq_def = np.abs(data["q_def"] - data["q_act"]); dq_ref = np.abs(data["q_ref"] - data["q_act"])
    dF_def = np.abs(data["F_def"] - data["F_act"]); dF_ref = np.abs(data["F_ref"] - data["F_act"])
    return {"E": (np.mean(eE_def), np.mean(eE_ref)),
            "q": (dq_def.mean(), dq_ref.mean()),
            "F": (dF_def.mean(), dF_ref.mean())}


def parity_panel(ax, act, dflt, ref, unit, title, ms, alpha):
    lo = min(act.min(), dflt.min(), ref.min()); hi = max(act.max(), dflt.max(), ref.max())
    pad = 0.05 * (hi - lo); lim = [lo - pad, hi + pad]
    ax.plot(lim, lim, 'k--', lw=1, alpha=.7, zorder=1)
    ax.scatter(act, dflt, s=ms, c=RED, alpha=alpha, lw=0, zorder=2,
               label=f"default R²={r2(act,dflt):.3f} RMSE={rmse(act,dflt):.3g}")
    ax.scatter(act, ref, s=ms, c=GRN, alpha=alpha, lw=0, marker='^', zorder=3,
               label=f"re-fit  R²={r2(act,ref):.3f} RMSE={rmse(act,ref):.3g}")
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
    ax.set_xlabel(f"reference / {unit}"); ax.set_ylabel(f"GFN2-xTB / {unit}")
    ax.set_title(title, fontsize=11); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="upper left")


def figure_parity(data, out_png, suptitle):
    import matplotlib.pyplot as plt
    E_act = np.concatenate([_mc(cv["dft"]) for cv in data["curves"]])
    E_def = np.concatenate([_mc(cv["def"]) for cv in data["curves"]])
    E_ref = np.concatenate([_mc(cv["ref"]) for cv in data["curves"]])
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 6.2))
    parity_panel(ax[0], E_act, E_def, E_ref, "kcal mol$^{-1}$", "Energy (relative, per curve)", 26, .7)
    parity_panel(ax[1], data["F_act"], data["F_def"], data["F_ref"], "E$_h$ Bohr$^{-1}$",
                 "Force components (all atoms)", 4, .25)
    parity_panel(ax[2], data["q_act"], data["q_def"], data["q_ref"], "e",
                 "Mulliken charges (all atoms)", 8, .4)
    fig.suptitle(suptitle, fontsize=13)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.90, bottom=0.13, wspace=0.28)
    fig.savefig(out_png, dpi=150); plt.close(fig)


def figure_profile(data, out_png, suptitle):
    import matplotlib.pyplot as plt
    cs = data["curves"]; n = len(cs)
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 5.2), squeeze=False)
    for ax, cv in zip(axes[0], cs):
        yd, ya, yr = _mc(cv["dft"]), _mc(cv["def"]), _mc(cv["ref"])
        mae = lambda a: np.mean(np.abs(a - yd))
        ax.plot(cv["nodes"], yd, 'o-', color="#111", lw=2.2, ms=5, label="DFT (reference)")
        ax.plot(cv["nodes"], ya, 's--', color=RED, lw=1.6, ms=4, label=f"xTB default (MAE {mae(ya):.1f})")
        ax.plot(cv["nodes"], yr, '^-', color=GRN, lw=1.6, ms=4, label=f"xTB re-fit (MAE {mae(yr):.1f})")
        ax.set_xlabel("string node"); ax.set_ylabel("rel. E / kcal mol$^{-1}$ (mean-centered)")
        ax.set_title(f"replica {cv['tag']}"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.suptitle(suptitle, fontsize=12); fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=150); plt.close(fig)


def figure_errors(data, out_png, suptitle):
    import matplotlib.pyplot as plt
    eE_def, eE_ref = [], []; ydL, ydfL, yrfL = [], [], []
    for cv in data["curves"]:
        yd, ya, yr = _mc(cv["dft"]), _mc(cv["def"]), _mc(cv["ref"])
        eE_def.extend(np.abs(ya - yd)); eE_ref.extend(np.abs(yr - yd))
        ydL.extend(yd); ydfL.extend(ya); yrfL.extend(yr)
    eE_def, eE_ref = np.array(eE_def), np.array(eE_ref)
    dq_def, dq_ref = np.abs(data["q_def"]-data["q_act"]), np.abs(data["q_ref"]-data["q_act"])
    dF_def, dF_ref = np.abs(data["F_def"]-data["F_act"]), np.abs(data["F_ref"]-data["F_act"])
    yd, ydf, yrf = np.array(ydL), np.array(ydfL), np.array(yrfL)

    def box(ax, a, b, ylab, title, log=False):
        bp = ax.boxplot([a, b], tick_labels=["default", "re-fit"], widths=.55, patch_artist=True,
                        medianprops=dict(color="k", lw=1.5), flierprops=dict(marker='.', ms=3, alpha=.4))
        for pc, cc in zip(bp['boxes'], (RED, GRN)): pc.set_facecolor(cc); pc.set_alpha(.55)
        if log: ax.set_yscale('log')
        ax.set_ylabel(ylab); ax.set_title(title, fontsize=10); ax.grid(alpha=.3, axis='y')

    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))
    box(ax[0, 0], eE_def, eE_ref, "|ΔE| / kcal mol$^{-1}$", "Energy error per node")
    box(ax[0, 1], dq_def, dq_ref, "|Δq| / e", "Mulliken charge error per atom")
    box(ax[1, 0], dF_def, dF_ref, "|ΔF| / E$_h$ Bohr$^{-1}$", "Force residual per atom", log=True)
    axp = ax[1, 1]
    lim = [min(yd.min(), ydf.min(), yrf.min())-2, max(yd.max(), ydf.max(), yrf.max())+2]
    axp.plot(lim, lim, 'k--', lw=1, alpha=.6)
    axp.scatter(yd, ydf, s=18, c=RED, alpha=.6, label=f"default (RMSE {rmse(yd,ydf):.1f})")
    axp.scatter(yd, yrf, s=18, c=GRN, alpha=.6, marker='^', label=f"re-fit (RMSE {rmse(yd,yrf):.1f})")
    axp.set_xlabel("DFT rel. E"); axp.set_ylabel("xTB rel. E"); axp.set_title("Energy parity", fontsize=10)
    axp.legend(fontsize=8); axp.grid(alpha=.3); axp.set_aspect('equal')
    fig.suptitle(suptitle, fontsize=12); fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=150); plt.close(fig)


def figure_loss(fit_dir, out_png, suptitle):
    import re, matplotlib.pyplot as plt
    log = os.path.join(fit_dir, "ga_logger.log")
    if not os.path.exists(log):
        print("  (no ga_logger.log; skipping loss plot)"); return
    baseline = None
    fs = os.path.join(fit_dir, "fit_summary.txt")
    if os.path.exists(fs):
        m = re.search(r"baseline SCORE ([\d.eE+-]+)", open(fs).read()); baseline = float(m.group(1)) if m else None
    best = {}
    with open(log) as f:
        next(f)
        for line in f:
            c = line.split("\t")
            if len(c) < 6: continue
            try: it, G, Gr, Ch, E = int(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])
            except ValueError: continue
            if G >= 1e290: continue
            if it not in best or G < best[it][0]: best[it] = (G, Gr, Ch, E)
    if not best:
        print("  (no finite GA rows; skipping loss plot)"); return
    gens = sorted(best); G, Gr, Ch, E = ([best[i][k] for i in gens] for k in range(4))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].plot(gens, G, 'o-', color=BLU, lw=2, ms=4)
    if baseline: ax[0].axhline(baseline, ls='--', color="#999", lw=1)
    ax[0].set_yscale('log'); ax[0].set_xlabel("GA generation"); ax[0].set_ylabel("best global SCORE")
    ax[0].set_title(f"Loss: {G[0]:.2e} -> {G[-1]:.2e}", fontsize=10); ax[0].grid(alpha=.3, which='both')
    ax[1].plot(gens, Gr, 'o-', color=RED, label="Gradient"); ax[1].plot(gens, Ch, 's-', color="#edae49", label="Charge")
    ax[1].plot(gens, E, '^-', color=GRN, label="Energy"); ax[1].set_yscale('log')
    ax[1].set_xlabel("GA generation"); ax[1].set_ylabel("component score"); ax[1].legend(fontsize=8)
    ax[1].set_title("ECG components (Global = Grad×Chrg×E)", fontsize=10); ax[1].grid(alpha=.3, which='both')
    fig.suptitle(suptitle, fontsize=12); fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=150); plt.close(fig)
