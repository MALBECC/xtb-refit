#!/usr/bin/env python3
"""Re-fit GFN2-xTB to the TRAIN curves with the published DDGA (full Energy-Charge-
Gradient objective). All knobs come from config.py (override via env for HPC scaling).
Outputs to FIT_DIR: param_gfn2-xtb.txt (the parameter set for MD), best_solution.txt,
fit_summary.txt, ga_logger.log (loss log for plot_loss.py).

Usage:  python fit.py
"""
import os, time
import numpy as np
import common as K          # sets env before xtb import
import config as C

from algorithms import DDGeneticAlgorithm as DDGA
from objectivefunctions import ECGFittingV3Json
from parameters import parm_HCONS, XTBParam

paths = K.curve_paths(C.TRAIN_DIR)
assert paths, f"No dataset_f*.json in {C.TRAIN_DIR} — run assemble_dataset.py first"
os.makedirs(C.FIT_DIR, exist_ok=True)
GA_OUT = os.path.join(C.FIT_DIR, "ga_work"); os.makedirs(GA_OUT, exist_ok=True)

print(C.summary())
print(f"\nfitting {len(paths)} train curve(s), objective E+C+G  {C.OBJ}")
orig = K.default_gfn2_params()
cvo = [[True, True, True] for _ in paths]
funct = ECGFittingV3Json(patterns=C.PATTERNS, parm=parm_HCONS, paths=paths,
                         cv_objectives=cvo, out_folder=GA_OUT, **C.OBJ)
def objective(inp): return funct.evaluate(solution=inp[0], thr_id=inp[1])

s0 = np.array(funct.evaluate(solution=list(orig), thr_id=0))
print(f"baseline (default GFN2) SCORE={s0[0]:.3e}  GRAD={s0[1]:.3f} CHRG={s0[2]:.3f} E={s0[3]:.3e}")

vb = np.array([[v - C.BOUNDS_FRAC*abs(v), v + C.BOUNDS_FRAC*abs(v)] for v in orig])
ap = {'max_num_iteration': C.ITERS, 'population_size': C.POP, 'mutation_probability': 0.9,
      'elit_ratio': 0.15, 'crossover_probability': 0.95, 'parents_portion': 0.4,
      'crossover_type': 'uniform', 'max_iteration_without_improv': C.MNIWI}
model = DDGA(function=objective, dimension=75, variable_boundaries=vb,
             algorithm_parameters=ap, convergence_curve=False, progress_bar=False)
t = time.time()
model.run(guess=list(orig), log_path=os.path.join(C.FIT_DIR, "ga_logger.log"), n_cpus=C.N_CPUS)
dt = time.time() - t

best = np.array(model.best_variable); best_score = float(model.best_function)
XTBParam(parm_HCONS, C.PATTERNS, list(best)).print_param_file(os.path.join(C.FIT_DIR, "param_gfn2-xtb.txt"))
np.savetxt(os.path.join(C.FIT_DIR, "best_solution.txt"), best, header="75 optimized GFN2 params")
with open(os.path.join(C.FIT_DIR, "fit_summary.txt"), "w") as f:
    f.write(f"system {C.SYSTEM}\n")
    f.write(f"baseline SCORE {s0[0]:.4e} -> best SCORE {best_score:.4e}\n")
    f.write(f"train_curves {len(paths)}  POP {C.POP} ITERS {C.ITERS} bounds {C.BOUNDS_FRAC} "
            f"N_CPUS {C.N_CPUS} etemp {C.XTB_ETEMP} ref_ener {C.OBJ['ref_ener']}\n")
    f.write(f"time_s {dt:.0f}\n")
    f.write(f"convergence {list(map(float, model.report))}\n")
print(f"\nDONE: SCORE {s0[0]:.3e} -> {best_score:.3e} in {dt:.0f}s")
print("exported", os.path.join(C.FIT_DIR, "param_gfn2-xtb.txt"))
