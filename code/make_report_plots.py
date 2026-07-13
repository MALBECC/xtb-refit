#!/usr/bin/env python3
"""Illustrative report plots for a completed re-fit (readable summaries of the fit).

These are the figures used to communicate results; they complement (and are clearer
than) make_plots.py's built-in profile_3way.png, which draws one panel per replica and
is unreadable for many replicas. Needs FIT_DIR/param_gfn2-xtb.txt (from fit.py).

Produces, in FIT_DIR:
  dataset_overview.png       DFT reaction-profile ensemble + per-node thermal spread (DFT only)
  profile_3way_averaged.png  mean predicted profile DFT vs xTB-default vs re-fit (+ residual panel)
  profile_error_per_node.png per-node mean |error| vs DFT (energy / charge / force), default vs re-fit
  per_node_train_error.csv   the numbers behind profile_error_per_node

Uses all replicas by default (override with XTBFIT_REPORT_REPLICAS, e.g. "1-20").
Usage:  python make_report_plots.py
"""
import os, csv, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common as K
import config as C
import plot_lib as P
import analysis_cv as X          # reuse load_records() + parallel evaluate()

RED, GRN, BLU, BLK = P.RED, P.GRN, "#1f4e79", "#111"
REPS = X._reps("XTBFIT_REPORT_REPLICAS", None)   # None -> all replicas present


def _robust_mean(vals, k=10.0):
    a = np.array([v for v in vals if np.isfinite(v)])
    if a.size == 0:
        return np.nan
    m = np.median(a); mad = np.median(np.abs(a - m))
    return float(a.mean()) if mad <= 0 else float(a[np.abs(a - m) <= k * mad].mean())


def dataset_overview(records, out):
    byf = collections.defaultdict(list)
    for r in records:
        byf[r["fidx"]].append(r)
    prof, bynode = {}, collections.defaultdict(list)
    for f, rs in byf.items():
        rs = sorted(rs, key=lambda r: r["node"]); ns = [r["node"] for r in rs]
        v = np.array([r["energy_kcal"] for r in rs]); v = v - v.mean()
        prof[f] = (ns, v)
        for n, e in zip(ns, v):
            bynode[n].append(e)
    nodes = sorted(bynode)
    med = np.array([np.median(bynode[n]) for n in nodes])
    lo = np.array([np.percentile(bynode[n], 10) for n in nodes])
    hi = np.array([np.percentile(bynode[n], 90) for n in nodes])
    spread = np.array([np.std(bynode[n]) for n in nodes])
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.6))
    for f, (ns, v) in prof.items():
        ax[0].plot(ns, v, '-', color="0.6", lw=0.6, alpha=0.45)
    ax[0].plot(nodes, med, '-', color=BLU, lw=2.6, label=f"median profile ({len(prof)} replicas)")
    ax[0].fill_between(nodes, lo, hi, color=BLU, alpha=0.12, label="10-90th percentile")
    ax[0].set_xlabel("reaction-coordinate node (1=reactant … end=product)")
    ax[0].set_ylabel("relative energy / kcal mol$^{-1}$ (mean-centered per replica)")
    ax[0].set_title(f"DFT reference ensemble — {C.SYSTEM}", fontsize=11)
    ax[0].grid(alpha=.3); ax[0].legend(fontsize=8, loc="upper left")
    ax[1].bar(nodes, spread, color=GRN, alpha=0.75, width=0.85)
    ax[1].axhline(spread.mean(), ls="--", color="0.3", lw=1, label=f"mean {spread.mean():.1f}")
    ax[1].set_xlabel("reaction-coordinate node"); ax[1].set_ylabel("std of rel. E across replicas / kcal mol$^{-1}$")
    ax[1].set_title("Thermal spread per node", fontsize=11)
    ax[1].grid(alpha=.3, axis='y'); ax[1].legend(fontsize=8)
    fig.suptitle("Dataset overview (what the re-fit targets)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("saved", out)


def _profiles(records, reps, D, R):
    """mean-centered energies per replica -> per-node lists for dft/def/ref."""
    byf = collections.defaultdict(list)
    for r in records:
        if r["fidx"] in reps:
            byf[r["fidx"]].append(r)
    p = {k: collections.defaultdict(list) for k in ("dft", "def", "ref")}
    for f, rs in byf.items():
        rs = sorted(rs, key=lambda r: r["node"]); labs = [r["label"] for r in rs]
        dft = np.array([r["energy_kcal"] for r in rs]); dft -= dft.mean()
        de = np.array([D[l]["e"] if D[l]["e"] is not None else np.nan for l in labs]); de -= np.nanmean(de)
        re = np.array([R[l]["e"] if R[l]["e"] is not None else np.nan for l in labs]); re -= np.nanmean(re)
        for i, r in enumerate(rs):
            p["dft"][r["node"]].append(dft[i]); p["def"][r["node"]].append(de[i]); p["ref"][r["node"]].append(re[i])
    return p


def profile_averaged(records, reps, D, R, out):
    p = _profiles(records, reps, D, R); nodes = sorted(p["dft"])
    mean = lambda k: np.array([np.nanmean(p[k][n]) for n in nodes])
    sd = lambda k: np.array([np.nanstd(p[k][n]) for n in nodes])
    fig, ax = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for k, c, lab, mk, lw in [("dft", BLK, "DFT reference", "o-", 2.6),
                              ("def", RED, "GFN2-xTB default", "s--", 1.8),
                              ("ref", GRN, "GFN2-xTB re-fit", "^-", 2.0)]:
        m, s = mean(k), sd(k)
        ax[0].plot(nodes, m, mk, color=c, lw=lw, ms=4, label=lab)
        ax[0].fill_between(nodes, m - s, m + s, color=c, alpha=0.10, lw=0)
    ax[0].set_ylabel("mean relative energy / kcal mol$^{-1}$")
    ax[0].set_title(f"Averaged reaction profile — {C.SYSTEM} ({len(reps)} replicas, ±1σ shaded)", fontsize=12)
    ax[0].grid(alpha=.3); ax[0].legend(fontsize=10, loc="upper left")
    dd, dr = mean("def") - mean("dft"), mean("ref") - mean("dft")
    ax[1].axhline(0, color=BLK, lw=1)
    ax[1].plot(nodes, dd, 's--', color=RED, lw=1.5, ms=3, label=f"default (mean|Δ| {np.nanmean(np.abs(dd)):.2f})")
    ax[1].plot(nodes, dr, '^-', color=GRN, lw=1.8, ms=3, label=f"re-fit (mean|Δ| {np.nanmean(np.abs(dr)):.2f})")
    ax[1].set_ylabel("mean(xTB)−mean(DFT)\n/ kcal mol$^{-1}$"); ax[1].set_xlabel("reaction-coordinate node")
    ax[1].grid(alpha=.3); ax[1].legend(fontsize=9, loc="upper left")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("saved", out)


def error_per_node(records, reps, D, R, out, csv_out):
    byf = collections.defaultdict(list)
    for r in records:
        if r["fidx"] in reps:
            byf[r["fidx"]].append(r)
    errs = {q: {"def": collections.defaultdict(list), "ref": collections.defaultdict(list)} for q in ("E", "q", "F")}
    for f, rs in byf.items():
        rs = sorted(rs, key=lambda r: r["node"]); labs = [r["label"] for r in rs]
        edft = np.array([r["energy_kcal"] for r in rs]); edft -= edft.mean()
        for tag, S in (("def", D), ("ref", R)):
            ex = np.array([S[l]["e"] if S[l]["e"] is not None else np.nan for l in labs]); ex -= np.nanmean(ex)
            for i, r in enumerate(rs):
                n, l = r["node"], r["label"]
                errs["E"][tag][n].append(abs(ex[i] - edft[i]))
                if S[l]["q"] is not None:
                    errs["q"][tag][n].append(float(np.mean(np.abs(S[l]["q"] - np.asarray(r["charges"], float)))))
                    errs["F"][tag][n].append(float(np.mean(np.abs((-np.asarray(r["gradient"], float)) - (-S[l]["g"])))))
    nodes = sorted(errs["E"]["def"])
    curve = {q: {t: np.array([_robust_mean(errs[q][t][n]) for n in nodes]) for t in ("def", "ref")} for q in ("E", "q", "F")}
    info = [("E", "energy (relative)", "kcal mol$^{-1}$"), ("q", "Mulliken charge", "e"),
            ("F", "force component", "E$_h$ Bohr$^{-1}$")]
    fig, ax = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    for a, (q, lab, unit) in zip(ax, info):
        d, r = curve[q]["def"], curve[q]["ref"]
        a.plot(nodes, d, 's--', color=RED, lw=1.5, ms=4, label=f"default (mean {np.nanmean(d):.3g})")
        a.plot(nodes, r, '^-', color=GRN, lw=1.9, ms=4, label=f"re-fit (mean {np.nanmean(r):.3g})")
        a.fill_between(nodes, r, d, where=r < d, color=GRN, alpha=0.12, interpolate=True)
        a.fill_between(nodes, r, d, where=r >= d, color=RED, alpha=0.10, interpolate=True)
        a.set_ylabel(f"mean |error| {lab}\n/ {unit}"); a.grid(alpha=.3); a.legend(fontsize=9, loc="upper right")
    ax[-1].set_xlabel("reaction-coordinate node")
    fig.suptitle(f"Per-node average error vs DFT — {C.SYSTEM} (train, {len(reps)} replicas)\n"
                 "green = re-fit better; red = worse", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("saved", out)
    with open(csv_out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["node", "E_def", "E_ref", "q_def", "q_ref", "F_def", "F_ref"])
        for i, n in enumerate(nodes):
            w.writerow([n] + [f"{curve[q][t][i]:.5f}" for q in ("E", "q", "F") for t in ("def", "ref")])
    print("saved", csv_out)


def main():
    ref = os.path.join(C.FIT_DIR, "param_gfn2-xtb.txt")
    assert os.path.exists(ref), f"no {ref} — run fit.py first"
    records = X.load_records()
    reps = set(REPS) if REPS else {r["fidx"] for r in records}
    print(f"report plots for {C.SYSTEM}: {len(records)} structures, replicas {sorted(reps)[:3]}…{sorted(reps)[-1]}")
    defdir = os.path.join(C.FIT_DIR, "_report_default_param")
    K.write_param_dir(K.default_gfn2_params(), defdir)
    print("evaluating default + re-fit over the data (parallel) …")
    D = X.evaluate(records, reps, defdir, C.N_CPUS)
    R = X.evaluate(records, reps, C.FIT_DIR, C.N_CPUS)
    dataset_overview(records, os.path.join(C.FIT_DIR, "dataset_overview.png"))
    profile_averaged(records, reps, D, R, os.path.join(C.FIT_DIR, "profile_3way_averaged.png"))
    error_per_node(records, reps, D, R, os.path.join(C.FIT_DIR, "profile_error_per_node.png"),
                   os.path.join(C.FIT_DIR, "per_node_train_error.csv"))
    print("done.")


if __name__ == "__main__":
    main()
