# Running xtb_refit on another machine (HPC / big-core node)

This project is self-contained: copy the whole `xtb_refit/` folder, create the conda
env, and run. The bundled fitter (`deps/gfn2-xtb_paramfitter`, includes the SCF patch)
and the data (`data/`) travel with it.

## 1. Copy it over
```bash
# ~55 MB (48 MB data + 6 MB bundled fitter). From your local machine:
rsync -az /home/jsemelak/2-Research/3-MSRA/xtb_refit/  user@hpc:/path/to/xtb_refit/
# (or: tar czf xtb_refit.tgz xtb_refit && scp xtb_refit.tgz user@hpc: && ssh … tar xzf …)
```

## 2. Create the environment (one-time)
```bash
cd /path/to/xtb_refit
mamba env create -f environment.yml     # or: conda env create -f environment.yml
```
`xtb-python` bundles the xtb engine — no separate xtb install needed.

**Shared-cluster gotcha:** if it fails with `cannot set permissions … pkgs/cache`, the
shared conda package cache is owned by another user. Use a cache dir you own:
```bash
mkdir -p $HOME/conda_pkgs
CONDA_PKGS_DIRS=$HOME/conda_pkgs mamba env create -f environment.yml
```
`code/env.sh` auto-detects conda in common locations. If yours is elsewhere:
```bash
export XTBFIT_CONDA=/path/to/conda/etc/profile.d/conda.sh   # and/or XTBFIT_ENV=<name>
```

## 3. Choose scale (env vars — no file edits)
```bash
source code/env.sh
export XTBFIT_NCPUS=$(( $(nproc) - 2 ))   # use the node's cores
# how many replicas to train on (thermal snapshots per node; more = better stats, slower):
export XTBFIT_TRAIN_REPLICAS="1-10"       # or "1-25", or "all"
export XTBFIT_VAL_REPLICAS="11-20"        # disjoint held-out set (or "none")
# bigger population exploits many cores AND improves the search:
export XTBFIT_POP=48 XTBFIT_ITERS=40
```
> Ranges like `"1-10"` are accepted; so are comma lists `"1,2,3"`. `all`/`none` also work.

## 4. Run
```bash
cd code
python check_data.py           # QC FIRST: duplicates / SCF fails / outliers (see README §4)
python assemble_dataset.py     # engrad/ -> train/ + val/
python dryrun.py               # sanity + prints a measured wall-time estimate for YOUR node
python fit.py                  # the fit -> ../fit/<system>/param_gfn2-xtb.txt   (run in tmux/nohup)
python make_report_plots.py    # readable result plots (averaged profile, per-node error)
python make_plots.py           # built-in train plots
python validate.py             # held-out validation (if a val split was kept)
```
Or run the whole thing unattended: `nohup bash ../run_all.sh > ../run_all.log 2>&1 &`.
Run the fit under `tmux`/`nohup` so it survives disconnects:
`nohup python fit.py > ../fit/$XTBFIT_SYSTEM/run.log 2>&1 &`

## 5. How it scales (important)
The DDGA parallelizes **across the population** (one core per parameter set). But each
single evaluation scores its structures **serially**, so there is a per-generation floor:

```
wall time  ≈  ceil(POP / N_CPUS) × ITERS × (N_structures × ~0.08 s)
```

Measured on a 20-core node (POP=24, N_CPUS=12):

| training set | structures | ~1 evaluation | ~full fit (30 gen) |
|---|---|---|---|
| 10 replicas | 479 | 39 s | ~0.6 h |
| 25 replicas | ~1200 | ~1.6 min | ~1.6 h |
| all 50 replicas | ~2399 | ~3.2 min | ~1.5–3 h |

Key point: **adding cores beyond `POP` does not lower per-generation time** (each eval is
serial internally). To exploit a big node, raise **`POP` together with `N_CPUS`** (bigger,
better-converged population at the same per-generation cost), and/or use more replicas.
`dryrun.py` prints an estimate measured on the actual node — trust that over the table.

## 6. Outputs (in `fit/<system>/`)
- **`param_gfn2-xtb.txt`** — the re-fitted parameters (use this for MD).
- `profile_3way.png`, `error_distributions.png`, `parity_plots.png`, `loss_curve.png` — train.
- `val_parity.png`, `val_errors.png`, `val_summary.txt` — **held-out validation** (the honest
  generalization check; small train→val gap = trustworthy for MD).
- `fit_summary.txt`, `best_solution.txt`, `ga_logger.log`.

## 7. Suggested prompt for Claude Code on the node
> "In this xtb_refit folder, create the conda env from environment.yml, then run the
> pipeline in code/ (assemble → dryrun → fit → make_plots → validate) training on
> replicas 1–25 with 11–20 held out for validation, using all available cores. Run the
> fit in the background and report the train and held-out MAEs plus the plots when done."
