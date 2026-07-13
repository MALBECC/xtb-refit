#!/usr/bin/env python3
"""Validate the pipeline WITHOUT optimizing: reads TRAIN curves, runs xtb with default
params, computes the ECG objective, and checks that custom params take effect (XTBPATH
honored). Also reports a rough per-generation wall-time estimate. Usage: python dryrun.py"""
import os, time
import numpy as np
import common as K
import config as C

from objectivefunctions import ECGFittingV3Json

paths = K.curve_paths(C.TRAIN_DIR)
assert paths, f"No dataset_f*.json in {C.TRAIN_DIR} — run assemble_dataset.py first"
print(C.summary())
n_struct = sum(len(K.load_curve(p)["bases"]) for p in paths)
orig = K.default_gfn2_params()
funct = ECGFittingV3Json(patterns=C.PATTERNS, parm=C.PARM, paths=paths,
                         cv_objectives=[[True, True, True]] * len(paths),
                         out_folder=os.path.join(C.FIT_DIR, "dryrun_work"), **C.OBJ)
t = time.time(); s0 = np.array(funct.evaluate(solution=list(orig), thr_id=0)); t_eval = time.time() - t
pert = list(orig)
for i in range(0, C.DIMENSION, 7): pert[i] *= 1.05
s1 = np.array(funct.evaluate(solution=pert, thr_id=1))
print(f"\ntrain curves {len(paths)} / {n_struct} structures   paramset={C.PARAMSET} ({C.DIMENSION} params)")
print(f"default   SCORE={s0[0]:.3e}  GRAD={s0[1]:.3f} CHRG={s0[2]:.3f} E={s0[3]:.3e}")
print(f"perturbed SCORE={s1[0]:.3e}")
print("pipeline runs   :", "YES" if (np.isfinite(s0[0]) and s0[0] < 1e299) else "NO")
print("params take hold:", "YES" if not np.allclose(s0, s1) else "NO (XTBPATH ignored!)")
# dedicated proof that the EXTRA (Se) params are wired: perturb ONLY indices >=75 (Se),
# holding H/C/N/O/S at stock, and confirm the objective moves.
if C.DIMENSION > 75:
    se = list(orig)
    for i in range(75, C.DIMENSION): se[i] *= 1.05
    s2 = np.array(funct.evaluate(solution=se, thr_id=2))
    print("Se params take hold:", "YES" if not np.allclose(s0, s2) else "NO (Se tokens DEAD!)",
          f" (SCORE {s0[0]:.3e} -> {s2[0]:.3e})")
# rough estimate: one generation ~ ceil(POP/N_CPUS) * t_eval ; total ~ * ITERS
per_gen = np.ceil(C.POP / C.N_CPUS) * t_eval
print(f"\ntiming: 1 evaluation ~{t_eval:.0f}s  ->  ~{per_gen/60:.1f} min/generation  ->  "
      f"~{per_gen*C.ITERS/3600:.1f} h for {C.ITERS} generations  (POP={C.POP}, N_CPUS={C.N_CPUS})")
