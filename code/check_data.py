#!/usr/bin/env python3
"""Data quality-control for a new system's ORCA *_engrad.json BEFORE fitting.

Run this FIRST on any new dataset. It catches the three classes of problem that
silently break (or bias) the GFN2-xTB re-fit — the same ones found in msrb_50rep:

  1. Duplicate / mislabeled files  — a file whose internal "label" field repeats
     another's (or disagrees with its filename). The pipeline keys off the internal
     label, so a duplicate mis-files into the wrong replica and makes one curve have
     more energies than unique structures -> the objective returns 1e300 for EVERY
     parameter set (found: node6_f12 was a byte-identical copy of node6_f11).

  2. SCF non-convergence with stock GFN2  — structures where default GFN2-xTB fails
     to converge even with Fermi smearing (etemp/maxiter from config). One such
     structure makes the GA baseline non-finite, so the fit can't start. These must
     go in EXCLUDE (found: node1_f1, node1_f19 — the fragile reactant geometry).

  3. Reference-data outliers  — bad DFT SCF states show up as energy spikes; blown-up
     geometries as huge gradient norms; broken population analysis as net-charge drift.
     Reported conservatively (robust z) so you can decide what to skip. msrb_50rep was
     clean beyond (1)+(2).

Usage:
  python check_data.py            # all checks (SCF scan uses XTBFIT_NCPUS cores)
  python check_data.py --no-scf   # skip the xtb SCF scan (fast, labels+outliers only)
  XTBFIT_SYSTEM=<newsys> python check_data.py

Prints a suggested XTBFIT_EXCLUDE string and any files to quarantine. It does NOT
modify any data — it only reports.
"""
import os, sys, json, glob, re, collections
import numpy as np
import common as K            # sets env before xtb import
import config as C
import assemble_dataset as A

LABEL_RE = re.compile(r"node(\d+)_f(\d+)", re.I)
DO_SCF = "--no-scf" not in sys.argv


def check_labels():
    """(1) duplicate internal labels + label/filename mismatches."""
    print("\n[1] DUPLICATE / MISLABELED FILES")
    files = sorted(glob.glob(os.path.join(C.ENGRAD_DIR, "**", "*_engrad.json"), recursive=True))
    label_to_paths = collections.defaultdict(list)
    mism = []
    for p in files:
        try:
            d = json.load(open(p))
        except (OSError, ValueError):
            print(f"  ! unreadable JSON: {p}"); continue
        lab = d.get("label") or os.path.splitext(os.path.basename(p))[0]
        label_to_paths[lab].append(p)
        mp, ml = LABEL_RE.search(p), LABEL_RE.search(lab)
        if mp and ml and (mp.group(1), mp.group(2)) != (ml.group(1), ml.group(2)):
            mism.append((p, lab))
    dups = {lab: ps for lab, ps in label_to_paths.items() if len(ps) > 1}
    if not dups and not mism:
        print(f"  OK — {len(files)} files, all labels unique and consistent with filenames.")
        return []
    quarantine = []
    for lab, ps in dups.items():
        print(f"  DUPLICATE label '{lab}' in {len(ps)} files:")
        for pp in ps:
            print(f"      {pp}")
        # keep the file whose name matches the label; quarantine the rest
        for pp in ps:
            mp = LABEL_RE.search(os.path.basename(pp))
            ml = LABEL_RE.search(lab)
            if mp and ml and (mp.group(1), mp.group(2)) != (ml.group(1), ml.group(2)):
                quarantine.append(pp)
    for p, lab in mism:
        print(f"  MISLABEL: {p}  internal label='{lab}' (disagrees with filename)")
    if quarantine:
        print("  -> SUGGEST quarantining (move out of engrad/):")
        for q in quarantine:
            print(f"       {q}")
    return quarantine


def _scf_worker(task):
    tok, Z, coords_A, chrg, mult, defdir = task
    e, _, _ = K.xtb_sp(np.asarray(Z), np.asarray(coords_A, float), chrg, mult, defdir)
    return tok, (e is None)


def check_scf(records):
    """(2) structures that fail SCF with stock GFN2 (current etemp/maxiter)."""
    print(f"\n[2] SCF CONVERGENCE with stock GFN2 (etemp={C.XTB_ETEMP} maxiter={C.XTB_MAXITER})")
    if not DO_SCF:
        print("  (skipped: --no-scf)"); return []
    from multiprocessing import Pool
    defdir = os.path.join(C.FIT_DIR, "_qc_default_param")
    K.write_param_dir(K.default_gfn2_params(), defdir)
    tasks = [(f"node{r['node']}_f{r['fidx']}", r["Z"],
              (np.asarray(r["coords_bohr"], float) / C.ANGSTROM_TO_BOHR).tolist(),
              r["chrg"], r["mult"], defdir) for r in records]
    ncpus = C.N_CPUS
    fails = []
    with Pool(ncpus) as pool:
        for label, failed in pool.imap_unordered(_scf_worker, tasks, chunksize=8):
            if failed:
                fails.append(label)
    if not fails:
        print(f"  OK — all {len(records)} structures converge with stock GFN2.")
    else:
        print(f"  {len(fails)} structure(s) FAIL SCF -> must be excluded:")
        for f in sorted(fails):
            print(f"      {f}")
    return sorted(fails)


def check_outliers(records):
    """(3) reference-data outliers: net charge, per-node energy z, gradient norms."""
    print("\n[3] REFERENCE-DATA OUTLIERS")
    # net charge (Mulliken sum vs formal)
    qbad = [(r["label"], r["qsum"], r["chrg"]) for r in records if abs(r["qsum"] - r["chrg"]) > 0.05]
    print(f"  net-charge (|Mulliken_sum - formal| > 0.05 e): {len(qbad)} "
          + (str([f"{l}({q:+.2f}!={c})" for l, q, c in qbad[:8]]) if qbad else "— none"))
    # per-node cross-replica robust z on mean-centered energies
    byf = collections.defaultdict(dict)
    for r in records:
        byf[r["fidx"]][r["node"]] = r["energy_kcal"]
    mc = {}
    for f, nd in byf.items():
        vals = np.array(list(nd.values())); m = vals.mean()
        for n, e in nd.items():
            mc[(n, f)] = e - m
    bynode = collections.defaultdict(list)
    for (n, f), v in mc.items():
        bynode[n].append((f, v))
    flagged = []
    for n, lst in bynode.items():
        vs = np.array([v for _, v in lst]); med = np.median(vs); mad = np.median(np.abs(vs - med)) or 1e-9
        for f, v in lst:
            z = abs(0.6745 * (v - med) / mad)
            if z > 6:
                flagged.append((z, n, f, abs(v - med)))
    flagged.sort(reverse=True)
    print(f"  energy outliers (robust |z|>6 within node across replicas): {len(flagged)}"
          + ("" if not flagged else " — CANDIDATES (inspect, likely bad DFT SCF state):"))
    for z, n, f, dev in flagged[:15]:
        print(f"      node{n}_f{f}  |z|={z:.1f}  dev={dev:.1f} kcal/mol")
    # gradient norms
    gn = np.array([float(np.sqrt((np.asarray(r["gradient"], float) ** 2).sum())) for r in records])
    hi = np.median(gn) + 8 * (np.median(np.abs(gn - np.median(gn))) or 1e-9)
    nbad = int((gn > hi).sum())
    print(f"  gradient norms: median {np.median(gn):.3f}  max {gn.max():.3f} Eh/Bohr  "
          f"(> {hi:.3f} robust-hi: {nbad} structures)")
    return [f"node{n}_f{f}" for _, n, f, _ in flagged]


def main():
    print(f"=== DATA QC for SYSTEM={C.SYSTEM} ===\n  engrad={C.ENGRAD_DIR}")
    files = A.find_compact(C.ENGRAD_DIR)
    if not files:
        sys.exit(f"No *_engrad.json under {C.ENGRAD_DIR}")
    records = [A.parse_one(p) for p in files]
    print(f"  parsed {len(records)} structures")

    quarantine = check_labels()
    scf_fail = check_scf(records)
    outliers = check_outliers(records)

    print("\n=== SUMMARY / SUGGESTED ACTIONS ===")
    if quarantine:
        print("  Quarantine these files (move out of engrad/), then re-run this check:")
        for q in quarantine:
            print(f"     mv '{q}' '{os.path.join(C.DATA_DIR, '_quarantine')}/'")
    must = sorted(set(scf_fail))
    if must:
        merged = sorted(set(C.EXCLUDE) | set(must))
        print(f"  Add SCF failures to exclusions:\n     export XTBFIT_EXCLUDE=\"{','.join(merged)}\"")
        print("  (or edit the EXCLUDE default in config.py)")
    if outliers:
        print(f"  Inspect energy-outlier candidates before deciding to skip them: {outliers[:10]}")
    if not (quarantine or must or outliers):
        print("  Data looks clean. Proceed to assemble_dataset.py -> dryrun.py -> fit.py.")


if __name__ == "__main__":
    main()
