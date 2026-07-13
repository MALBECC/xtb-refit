#!/usr/bin/env python3
"""Held-out validation: apply the fitted params (default & re-fit) to the VALIDATION
replicas — structures NEVER seen during training — and report out-of-sample errors +
the generalization gap vs the training set. With 50 replicas this is a cheap, honest
alternative to the (expensive) leave-one-node-out CV: no re-fitting, just prediction.

Outputs FIT_DIR/val_parity.png, val_errors.png, val_summary.txt. Usage: python validate.py"""
import os
import numpy as np
import common as K
import config as C
import plot_lib as P

ref = os.path.join(C.FIT_DIR, "param_gfn2-xtb.txt")
assert os.path.exists(ref), f"no {ref} — run fit.py first"
if not K.curve_paths(C.VAL_DIR):
    raise SystemExit(f"No validation curves in {C.VAL_DIR}. Set XTBFIT_VAL_REPLICAS and "
                     f"re-run assemble_dataset.py, or skip validation.")

DEF = K.write_param_dir(K.default_gfn2_params(), os.path.join(C.FIT_DIR, "default_param"))
print(f"running xtb (default + re-fit) over held-out VAL set ({C.VAL_DIR}) ...")
val = P.collect(C.VAL_DIR, DEF, C.FIT_DIR)
mv = P.maes(val)

# training MAEs for the gap (re-run over train — cheap relative to a fit)
print(f"running xtb over TRAIN set for the gap comparison ...")
tr = P.collect(C.TRAIN_DIR, DEF, C.FIT_DIR)
mt = P.maes(tr)

P.figure_parity(val, os.path.join(C.FIT_DIR, "val_parity.png"),
                f"Predicted vs actual — {C.SYSTEM} (HELD-OUT validation)")
P.figure_errors(val, os.path.join(C.FIT_DIR, "val_errors.png"),
                f"Error distributions — {C.SYSTEM} (HELD-OUT validation)")

lines = []
lines.append(f"system {C.SYSTEM}")
lines.append(f"train replicas {C.TRAIN_REPLICAS}  |  val replicas {C.VAL_REPLICAS}")
lines.append("")
lines.append(f"{'quantity':8s} {'default':>12s} {'refit(train)':>14s} {'refit(val)':>14s}  gap(val-train)")
for key, unit in (("E", "kcal/mol"), ("q", "e"), ("F", "Eh/Bohr")):
    d, rt = mt[key]; _, rv = mv[key]
    lines.append(f"{key:8s} {d:12.4f} {rt:14.4f} {rv:14.4f}  {rv-rt:+.4f}  {unit}")
lines.append("")
lines.append("Read: small train->val gap = generalizes; large gap = overfit (won't transfer to MD).")
txt = "\n".join(lines)
open(os.path.join(C.FIT_DIR, "val_summary.txt"), "w").write(txt + "\n")
print("\n" + txt)
print(f"\nsaved val_parity.png, val_errors.png, val_summary.txt to {C.FIT_DIR}")
