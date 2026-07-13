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
    H/C/N/O/S values, plus the 20 Se ($Z=34) stock values when paramset=HCONSSe (95 total).
    Order matches C.PATTERNS (VARIABLE1#.. positionally)."""
    txt = open(os.path.join(C.PARAMFITTER, "experiments", "DDGA_OPTIM_CCR.py")).read()
    orig = ast.literal_eval(re.search(r"orig\s*=\s*(\[.*?\])", txt, re.S).group(1))
    assert len(orig) == 75, f"expected 75 stock params in DDGA_OPTIM_CCR.py, read {len(orig)}"
    return orig + list(C.SEED_EXTRA)


def write_param_dir(values, dirpath):
    """Write a param_gfn2-xtb.txt holding `values` into dirpath (used as XTBPATH).
    Uses the SELECTED template C.PARM (HCONS or the Se-extended HCONSSe)."""
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
