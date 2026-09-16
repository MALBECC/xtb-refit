"""Self-contained helpers for the re-fit pipeline (no external project deps besides
the bundled fitter in ../deps). Import this first in every script — it sets the
fork+OpenMP safety vars and the SCF-robustness env vars BEFORE xtb/numpy load."""
import os, sys, re, ast, glob, json

import config as C

# fork+OpenMP safety (must precede xtb/numpy import) + SCF robustness + headless mpl
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XTB_ETEMP", str(C.XTB_ETEMP))
os.environ.setdefault("XTB_MAXITER", str(C.XTB_MAXITER))

sys.path.insert(0, C.PARAMFITTER)                 # make the fitter importable

HARTREE_TO_KCAL_MOL = C.HARTREE_TO_KCAL_MOL
ANGSTROM_TO_BOHR    = C.ANGSTROM_TO_BOHR
H2KCAL = HARTREE_TO_KCAL_MOL


def default_gfn2_params():
    """Stock GFN2 seed for the SELECTED parameter set (C.PARAMSET): the 75 published
    H/C/N/O/S values, plus 20 stock values for each extra element the paramset tunes
    (C.EXTRA_ELEMENTS, e.g. Se / P+Mg). Order matches C.PATTERNS (VARIABLE1#.. positionally)."""
    txt = open(os.path.join(C.PARAMFITTER, "experiments", "DDGA_OPTIM_CCR.py")).read()
    orig = ast.literal_eval(re.search(r"orig\s*=\s*(\[.*?\])", txt, re.S).group(1))
    assert len(orig) == 75, f"expected 75 stock params in DDGA_OPTIM_CCR.py, read {len(orig)}"
    return orig + list(C.SEED_EXTRA)


def warn_objective_scale(s, label="baseline"):
    """Warn when an ECG score points at a BROKEN DATASET rather than bad parameters.

    Calibration note — a huge starting SCORE is NORMAL here. The energy term is unbounded
    until the parameters are close, so healthy fits on this pipeline start anywhere in
    ~1e4..1e20 and still converge:
        msrb_sel_50rep   baseline 1.86e+19 -> best 1.78e+04   (fine)
        msrb_50rep       baseline 8.63e+11 -> best 2.55e+03   (fine)
    What actually signals trouble is a structure that will not run. Same system, one
    structure different:
        refit (node1_f1 excluded)   5.93e+13 -> 1.18e+05      (fine)
        refit_v1_withNode1f1        7.19e+40 -> 4.56e+32      (never recovered)
    So we flag the failure wall and magnitudes far above the healthy band — NOT energy
    dominance, which is expected and present in the good runs too."""
    import math
    score = float(s[0])
    w = []
    if not math.isfinite(score) or score >= 1e299:
        w.append("SCORE is at the failure wall (>=1e299): at least one structure did not "
                 "run at all.")
    elif score > 1e30:
        w.append(f"SCORE={score:.3e} is far above the ~1e4..1e20 band healthy fits start "
                 f"from; in practice this has meant ONE bad structure dominating the set.")
    if w:
        print(f"\n  !! WARNING - {label} objective suggests a DATASET problem, not bad params:")
        for m in w:
            print(f"  !!   {m}")
        print("  !!   Look for 'Error in single-point calculation' above to get the base name,")
        print("  !!   then add it to XTBFIT_EXCLUDE, and/or raise XTB_ETEMP / XTB_MAXITER.")
        print("  !!   (A large-but-finite SCORE below ~1e30 is normal and fits fine.)\n")
    return bool(w)


def write_param_dir(values, dirpath):
    """Write a param_gfn2-xtb.txt holding `values` into dirpath (used as XTBPATH).
    Uses the SELECTED template C.PARM (HCONS or one of its element-extended variants)."""
    from parameters import XTBParam
    os.makedirs(dirpath, exist_ok=True)
    XTBParam(C.PARM, C.PATTERNS, list(values)).print_param_file(
        os.path.join(dirpath, "param_gfn2-xtb.txt"))
    return dirpath


def xtb_sp(Z, xyz_ang, chrg, mult, parm_folder):
    """GFN2-xTB single point with the param file in parm_folder (coords in Angstrom).
    Returns (energy_Hartree, charges[e], gradient[Eh/Bohr]) or (None, None, None)."""
    import numpy as np
    from runners import XtbRunner
    e, q, g = XtbRunner(chrg=chrg, multip=mult, parm_folder=parm_folder,
                        atomic_numbers=Z, coords=xyz_ang, id_="sp", nthreads=1).run()
    return (None, None, None) if e is None else (e / H2KCAL, np.array(q), np.array(g))


def curve_paths(dset_dir):
    """Sorted dataset_f*.json curve files in a dataset dir (train/ or val/)."""
    return sorted(glob.glob(os.path.join(dset_dir, "dataset_f*.json")))


def load_curve(path):
    """Parse a dataset_f*.json into ordered arrays for plotting/analysis."""
    import numpy as np
    d = json.load(open(path))
    bases = list(d["system"]["chrg"].keys())
    node_idx = [int(re.search(r"node(\d+)", b).group(1)) if re.search(r"node(\d+)", b)
                else i for i, b in enumerate(bases)]
    return {
        "bases": bases, "node_idx": node_idx,
        "E_dft_kcal": np.array(d["energies"], float),
        "Z":     {b: np.array(d["atomic_numbers"][b]) for b in bases},
        "chrg":  {b: int(d["system"]["chrg"][b]) for b in bases},
        "mult":  {b: int(d["system"]["mult"][b]) for b in bases},
        "coords_A": {b: np.array(d["coordinates"][b], float) / ANGSTROM_TO_BOHR for b in bases},
        "q_dft":    {b: np.array(d["charges"][b], float) for b in bases},
        "grad_dft": {b: np.array(d["gradients"][b], float) for b in bases},   # dE/dx, Eh/Bohr
    }
