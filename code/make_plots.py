#!/usr/bin/env python3
"""Generate all TRAIN-set plots from the fitted params (one xtb pass over the data):
  profile_3way.png       DFT vs xTB default vs re-fit reaction profiles (per replica)
  error_distributions.png  E / charge / force error box plots + energy parity
  parity_plots.png       predicted-vs-actual for energy, forces, charges (R²/RMSE)
  loss_curve.png         DDGA convergence (from ga_logger.log)
Usage: python make_plots.py    (needs FIT_DIR/param_gfn2-xtb.txt from fit.py)"""
import os
import common as K
import config as C
import plot_lib as P

ref = os.path.join(C.FIT_DIR, "param_gfn2-xtb.txt")
assert os.path.exists(ref), f"no {ref} — run fit.py first"
DEF = K.write_param_dir(K.default_gfn2_params(), os.path.join(C.FIT_DIR, "default_param"))
REF = C.FIT_DIR

print(f"running xtb (default + re-fit) over TRAIN set ({C.TRAIN_DIR}) ...")
data = P.collect(C.TRAIN_DIR, DEF, REF)
tag = f"{C.SYSTEM} (train)"
P.figure_profile(data, os.path.join(C.FIT_DIR, "profile_3way.png"), f"Reaction profile — {tag}")
P.figure_errors(data, os.path.join(C.FIT_DIR, "error_distributions.png"), f"Error distributions — {tag}")
P.figure_parity(data, os.path.join(C.FIT_DIR, "parity_plots.png"), f"Predicted vs actual — {tag}")
P.figure_loss(C.FIT_DIR, os.path.join(C.FIT_DIR, "loss_curve.png"), f"DDGA convergence — {C.SYSTEM}")

m = P.maes(data)
print(f"\nTRAIN MAEs (default -> re-fit):")
print(f"  energy  {m['E'][0]:.2f} -> {m['E'][1]:.2f} kcal/mol")
print(f"  charge  {m['q'][0]:.4f} -> {m['q'][1]:.4f} e")
print(f"  force   {m['F'][0]:.4f} -> {m['F'][1]:.4f} Eh/Bohr")
print(f"saved 4 plots to {C.FIT_DIR}")
