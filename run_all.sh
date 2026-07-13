#!/usr/bin/env bash
# Full unattended pipeline. Point it at your data (INPUT) and a results dir (OUTPUT)
# that live OUTSIDE the repo, then run under nohup:
#   nohup bash run_all.sh > run_all.log 2>&1 &
#
# Configure the run by editing INPUT / OUTPUT / NCPUS below, or by exporting
# XTBFIT_INPUT / XTBFIT_OUTPUT / XTBFIT_NCPUS before launching. Everything generated
# (assembled train/ + val/, param file, plots, analysis/) goes under OUTPUT — the repo
# stays code-only.
#
# Stages (sequential, so total core use never exceeds NCPUS):
#   1. assemble Max dataset (all 50 replicas -> train, no held-out val)
#   2. production Max fit  (POP=80 ITERS=50)  -> $OUTPUT/param_gfn2-xtb.txt  [USE FOR MD]
#   3. train plots (report + built-in)
#   4. generalization analysis (analysis_cv.py): random split + per-node CV
#
# The fit self-limits via MNIWI=12 (stops after 12 generations without improvement),
# so wall time is usually well under the 50-iteration ceiling.
# (no `set -u`: env.sh intentionally probes optional unset vars like $XTBFIT_CONDA)
cd "$(dirname "$0")"
source code/env.sh
cd code

# ── EDIT THESE (or export XTBFIT_INPUT / XTBFIT_OUTPUT) ───────────────────────────
INPUT="${XTBFIT_INPUT:-$(cd .. && pwd)/data/msrb_50rep/engrad}"   # raw *_engrad.json
OUTPUT="${XTBFIT_OUTPUT:-$HOME/xtb_runs/msrb_50rep}"             # all results land here
NCPUS="${XTBFIT_NCPUS:-40}"
IO=(-i "$INPUT" -o "$OUTPUT")                                    # passed to every stage

stamp(){ date "+%Y-%m-%d %H:%M:%S"; }
log(){ echo "[$(stamp)] $*"; }
log "INPUT=$INPUT"
log "OUTPUT=$OUTPUT   NCPUS=$NCPUS"
mkdir -p "$OUTPUT"

# ── 1. production dataset: all 50 replicas as training, no validation split ──────
log "STAGE 1: assemble Max dataset (all 50 replicas)"
XTBFIT_TRAIN_REPLICAS=all XTBFIT_VAL_REPLICAS=none python assemble_dataset.py "${IO[@]}"

# ── 2. production Max fit ────────────────────────────────────────────────────────
log "STAGE 2: production Max fit (POP=80 ITERS=50 N_CPUS=$NCPUS)"
XTBFIT_TRAIN_REPLICAS=all XTBFIT_VAL_REPLICAS=none \
  XTBFIT_POP=80 XTBFIT_ITERS=50 XTBFIT_NCPUS=$NCPUS \
  python fit.py "${IO[@]}"
PARAM="$OUTPUT/param_gfn2-xtb.txt"
if [ ! -f "$PARAM" ]; then
  log "ERROR: production param file not produced ($PARAM). Aborting."
  exit 1
fi
log "STAGE 2 done -> $PARAM"

# ── 3. plots ─────────────────────────────────────────────────────────────────────
log "STAGE 3: plots (readable report plots + built-in train plots)"
XTBFIT_TRAIN_REPLICAS=all XTBFIT_VAL_REPLICAS=none XTBFIT_NCPUS=$NCPUS \
  python make_report_plots.py "${IO[@]}" || log "WARN: make_report_plots failed (non-fatal)"
XTBFIT_TRAIN_REPLICAS=all XTBFIT_VAL_REPLICAS=none \
  python make_plots.py "${IO[@]}" || log "WARN: make_plots failed (non-fatal)"

# ── 4. generalization / extrapolation analysis ───────────────────────────────────
#   (A) random 80/20 split over f1-20 (interpolation)
#   (B) per-node 6-fold CV over f1-10 (extrapolation -> per-node error map)
#   Lighter GA (POP=40 ITERS=40, 1 wave on 40 cores) since these measure generalization
#   patterns, not the production params.
log "STAGE 4: generalization analysis (random split + per-node CV)"
XTBFIT_ANALYSIS_POP=40 XTBFIT_ANALYSIS_ITERS=40 XTBFIT_ANALYSIS_NCPUS=$NCPUS \
  XTBFIT_ANALYSIS_RAND_REPLICAS="1-20" XTBFIT_ANALYSIS_NODE_REPLICAS="1-10" \
  XTBFIT_ANALYSIS_KFOLDS=6 XTBFIT_ANALYSIS_SEED=42 \
  python analysis_cv.py all "${IO[@]}" || log "WARN: analysis_cv failed"

log "ALL DONE. Production params: $PARAM"
log "Analysis outputs: $OUTPUT/analysis/"
