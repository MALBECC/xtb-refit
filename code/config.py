"""Central configuration for the GFN2-xTB re-fit pipeline.

Paths come from the command line so the repo stays code-only (data & results live
wherever you point them):
    -i / --input   INPUT   directory holding the raw ORCA *_engrad.json (read-only)
    -o / --output  OUTPUT  directory for EVERYTHING generated — assembled train/ + val/
                           curves, the fitted param file, plots, analysis/
    -s / --system  LABEL   optional display label (default: basename of OUTPUT)

Every script accepts these (they are parsed here at import and stripped from argv, so a
script's own args still work). Which flags a script needs:
    assemble / check_data / analysis_cv   -i (and -o)
    dryrun / fit / make_plots / make_report_plots / validate   -o   (train/ lives in OUTPUT)

If -i/-o are omitted the pipeline falls back to env vars (XTBFIT_INPUT / XTBFIT_OUTPUT)
and finally to the in-repo layout data/<SYSTEM>/ + fit/<SYSTEM> (backward compatible).
All scaling knobs are env-overridable (handy on HPC: no file edits).
"""
import os, sys

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(CODE_DIR)                       # project root (xtb_refit/)

def _envstr(name, default):   return os.environ.get(name, default)
def _envint(name, default):   return int(os.environ.get(name, default))
def _envfloat(name, default): return float(os.environ.get(name, default))
def _envlist(name, default):
    """Parse a replica spec: comma/space list and N-M ranges, e.g. '1-10', '1,2,3',
    '1-5 8 11-13'. 'all'/'none'/'' -> None (meaning: use everything, no split)."""
    v = os.environ.get(name)
    if v is None:              return default
    if v.strip().lower() in ("all", "none", ""):  return None
    out = []
    for tok in v.replace(",", " ").split():
        if "-" in tok:
            a, b = tok.split("-"); out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return out


def _pop_cli_io():
    """Extract -i/-o/-s (and --input/--output/--system, incl. '=' form) from sys.argv
    and remove them, so each script's own argument parsing is undisturbed."""
    flags = {"-i": "input", "--input": "input", "-o": "output", "--output": "output",
             "-s": "system", "--system": "system"}
    argv = sys.argv
    keep = [argv[0]] if argv else []
    vals = {}
    i = 1
    while i < len(argv):
        a = argv[i]
        matched = False
        for f, key in flags.items():
            if a == f and i + 1 < len(argv):
                vals[key] = argv[i + 1]; i += 2; matched = True; break
            if a.startswith(f + "="):
                vals[key] = a.split("=", 1)[1]; i += 1; matched = True; break
        if not matched:
            keep.append(a); i += 1
    sys.argv[:] = keep
    return vals


# ── which system / where its data & outputs live ──────────────────────────────
_SYSTEM_EXPLICIT = "XTBFIT_SYSTEM" in os.environ
SYSTEM  = _envstr("XTBFIT_SYSTEM", "msrb_50rep")           # display label + in-repo fallback subfolder
_INPUT  = _envstr("XTBFIT_INPUT",  "")                     # raw engrad dir (from -i or env)
_OUTPUT = _envstr("XTBFIT_OUTPUT", "")                     # working/output dir (from -o or env)

# path globals (set by _resolve, refreshed by set_io)
DATA_DIR = ENGRAD_DIR = TRAIN_DIR = VAL_DIR = FIT_DIR = None

def _resolve():
    global DATA_DIR, ENGRAD_DIR, TRAIN_DIR, VAL_DIR, FIT_DIR
    if _OUTPUT:                                            # everything generated goes under OUTPUT
        DATA_DIR  = _OUTPUT
        TRAIN_DIR = os.path.join(_OUTPUT, "train")
        VAL_DIR   = os.path.join(_OUTPUT, "val")
        FIT_DIR   = _OUTPUT
    else:                                                  # backward-compatible in-repo layout
        DATA_DIR  = os.path.join(ROOT, "data", SYSTEM)
        TRAIN_DIR = os.path.join(DATA_DIR, "train")
        VAL_DIR   = os.path.join(DATA_DIR, "val")
        FIT_DIR   = os.path.join(ROOT, "fit", SYSTEM)
    ENGRAD_DIR = _INPUT if _INPUT else os.path.join(ROOT, "data", SYSTEM, "engrad")


def set_io(input=None, output=None, system=None):
    """Point the pipeline at an input (engrad) dir and/or an output dir. Called from the
    CLI parse below, but also usable programmatically."""
    global _INPUT, _OUTPUT, SYSTEM
    if system:
        SYSTEM = system
    if input:
        _INPUT = os.path.abspath(os.path.expanduser(input))
    if output:
        _OUTPUT = os.path.abspath(os.path.expanduser(output))
        if not system and not _SYSTEM_EXPLICIT:            # friendly label from output dir name
            SYSTEM = os.path.basename(os.path.normpath(_OUTPUT)) or SYSTEM
    _resolve()


# parse CLI -i/-o/-s now (before any script reads the path globals) and resolve
_cli = _pop_cli_io()
set_io(_cli.get("input"), _cli.get("output"), _cli.get("system"))

# ── the fitter package (bundled for portability; override with $PARAMFITTER) ──
PARAMFITTER = _envstr("PARAMFITTER", os.path.join(ROOT, "deps", "gfn2-xtb_paramfitter"))

# ── structure / replica selection ─────────────────────────────────────────────
# f-index = replica snapshot; node index = reaction-coordinate position.
# TRAIN_REPLICAS / VAL_REPLICAS are lists of f-indices (disjoint). None = "use all
# available" (put everything in train, no validation split).
TRAIN_REPLICAS = _envlist("XTBFIT_TRAIN_REPLICAS", list(range(1, 11)))    # f1..f10
VAL_REPLICAS   = _envlist("XTBFIT_VAL_REPLICAS",   list(range(11, 21)))   # f11..f20 held out
# per-structure exclusions (bad reference data). node1_f1: bad B3LYP SCF state + no D4.
# node1_f19: stock GFN2-xTB fails SCF here even with Fermi smearing (etemp=500, maxiter=300)
# — verified by a full default-param scan (the only such failure); node1 is the fragile
# reactant geometry. Leaving it in makes the GA baseline non-finite. (Per-system; run
# check_data.py on a new dataset and set XTBFIT_EXCLUDE accordingly.)
EXCLUDE = [x for x in os.environ.get("XTBFIT_EXCLUDE", "node1_f1,node1_f19").split(",") if x]

# ── SCF robustness (small-gap barrier geometries need Fermi smearing to converge)─
XTB_ETEMP   = _envfloat("XTB_ETEMP", 500.0)      # electronic temperature, K
XTB_MAXITER = _envint("XTB_MAXITER", 300)        # SCF iterations

# ── GA (DDGA) knobs — raise POP/ITERS/N_CPUS on a big node (see RUN_ON_HPC.md) ──
POP         = _envint("XTBFIT_POP", 24)
ITERS       = _envint("XTBFIT_ITERS", 30)
BOUNDS_FRAC = _envfloat("XTBFIT_BOUNDS_FRAC", 0.02)      # +/- fraction around stock GFN2
N_CPUS      = _envint("XTBFIT_NCPUS", max(1, min(12, (os.cpu_count() or 4) - 2)))
MNIWI       = _envint("XTBFIT_MNIWI", 12)                # max iterations without improvement

# ── objective knobs (this reaction spans ~60-100 kcal/mol; see RUN_ON_HPC.md) ──
OBJ = dict(ref_ener=_envfloat("XTBFIT_REF_ENER", 60.0),
           ewall=1e299, cwall=1e299, gwall=1e299)

# Tunable GFN2 slots. The parameter set (HCONS=75 default, HCONSSe=95 with Selenium) is
# defined once in paramset.py and re-exported here, so config is the single import surface
# and PATTERNS/template/dimension/seed can never drift apart. Select via XTBFIT_PARAMSET.
sys.path.insert(0, CODE_DIR)
import paramset as _PS
PARAMSET   = _PS.PARAMSET       # "HCONS" or "HCONSSe"
PARM       = _PS.PARM           # selected parameter-file template string
PATTERNS   = _PS.PATTERNS       # VARIABLE1#..  (75 or 95)
DIMENSION  = _PS.DIMENSION      # len(PATTERNS)
SEED_EXTRA = _PS.SEED_EXTRA     # [] (HCONS) or the 20 Se stock values (HCONSSe)
# unit constants
HARTREE_TO_KCAL_MOL = 627.509391
ANGSTROM_TO_BOHR    = 1.8897259885789


def summary():
    return (f"SYSTEM={SYSTEM}\n  input (engrad)={ENGRAD_DIR}\n  output={FIT_DIR}\n"
            f"  train={TRAIN_DIR}\n  val={VAL_DIR}\n"
            f"  paramset={PARAMSET} ({DIMENSION} params)\n"
            f"  train_replicas={TRAIN_REPLICAS}  val_replicas={VAL_REPLICAS}  exclude={EXCLUDE}\n"
            f"  GA: POP={POP} ITERS={ITERS} bounds=±{BOUNDS_FRAC*100:.0f}% N_CPUS={N_CPUS} MNIWI={MNIWI}\n"
            f"  SCF: etemp={XTB_ETEMP}K maxiter={XTB_MAXITER}  ref_ener={OBJ['ref_ener']}\n"
            f"  PARAMFITTER={PARAMFITTER}")
