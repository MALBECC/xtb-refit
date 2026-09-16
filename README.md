# xtb_refit — reproducible GFN2-xTB re-fitting to ORCA reference data

Re-fit GFN2-xTB parameters (energies, charges, gradients) to per-structure ORCA
`*_engrad.json` reference data, using the DDGA optimizer in
[`gfn2-xtb_paramfitter`](https://github.com/josevlibera2010/gfn2-xtb_paramfitter)
(bundled in `deps/`, with the SCF-robustness patch). Self-contained: drop in a new
system's JSON, run a data check, then the pipeline.

```
xtb_refit/                 the repo — code + bundled fitter ONLY
├── code/            all scripts (system-agnostic; read code/config.py)
├── deps/            bundled fitter (gfn2-xtb_paramfitter + SCF-robustness patch)
├── examples/        a small runnable dataset in the supported input format
├── run_all.sh       one-command pipeline (assemble → fit → plots → analysis)
└── environment.yml
```

**Data and results live OUTSIDE the repo — you point at them per run** (so the repo stays
clean/code-only for GitHub):
```
  -i INPUT     folder of raw ORCA *_engrad.json                       (read-only)
  -o OUTPUT    folder for EVERYTHING generated: assembled train/ + val/
               curves, param_gfn2-xtb.txt (for MD), plots, analysis/
```
Every script accepts `-i`/`-o` (and `-s LABEL`). `assemble`/`check_data`/`analysis_cv`
need `-i`; `dryrun`/`fit`/`make_plots`/`make_report_plots`/`validate` only need `-o`
(they read `train/` from OUTPUT). If omitted, falls back to `$XTBFIT_INPUT`/`$XTBFIT_OUTPUT`,
then the in-repo `data/<SYSTEM>/` + `fit/<SYSTEM>` layout (backward compatible).

The file you use for MD is **`OUTPUT/param_gfn2-xtb.txt`**.

## Input format — compact `*_engrad.json`

**This is the supported input format.** One JSON per structure, holding the reference
energy, per-atom charges, gradient and geometry. Point `-i` at a folder of them (any
subfolder layout — they are found recursively) and everything else follows.

A small, runnable example lives in **[`examples/msrb_sel_mini/`](examples/msrb_sel_mini/)**
— 26 real structures (MsrB selenocysteine, B3LYP-D4/def2-TZVP), enough to run the whole
pipeline in a few minutes. Its README documents the schema, which fields are actually
required, the two filename rules that will silently ruin a dataset if broken, and how to
produce these files from ORCA (the `EnGrad` keyword, and why a merge step is needed).

Start there when onboarding a new system: copy its layout, run `check_data.py`, then
proceed as below.

---

## What this is — and credit

**This is basically a copy of the original [`gfn2-xtb_paramfitter`](https://github.com/josevlibera2010/gfn2-xtb_paramfitter) repo, plus a few utilities to make it work with ORCA JSON files.** The actual fitting — the Dynamic Domain Genetic Algorithm (DDGA), the multiobjective Energy–Charge–Gradient objective functions, and the xtb runner — is entirely that package, by **Velázquez-Libera et al.**, *J. Chem. Theory Comput.* **2025**, 21, 5118–5131 (DOI [10.1021/acs.jctc.5c00247](https://doi.org/10.1021/acs.jctc.5c00247)). It is **bundled verbatim in `deps/gfn2-xtb_paramfitter/`**, with a single one-line patch that exposes the `XTB_ETEMP` / `XTB_MAXITER` env vars so small-gap barrier structures converge. If you use this, cite that paper and repo.

The original package fits to Gaussian `.log` reference files (via cclib). The **only thing added here** is a thin, system-agnostic layer in `code/` that feeds it **ORCA** data instead and wraps it in a reproducible, one-command pipeline:

- **`assemble_dataset.py`** — the core adapter: converts ORCA `! … EnGrad` per-structure `*_engrad.json` into the array `dataset_f*.json` curves the fitter's program-agnostic JSON objective (`ECGFittingV3Json`) already knows how to read (unit conversions, replica → curve grouping, train/val split, exclusions).
- **`config.py` / `common.py` / `env.sh`** — one-file configuration + portable environment setup.
- **`check_data.py`, `make_plots.py` / `make_report_plots.py`, `validate.py`, `analysis_cv.py`, `run_all.sh`** — data QC, result plots, held-out validation, generalization studies, and the driver script.

In short: **the bundled repo does the fitting; the code here just makes it speak ORCA JSON.** Everything below is how to drive that.

---

## 0. One-time: create the conda environment
```bash
cd xtb_refit
mamba env create -f environment.yml        # (or: conda env create -f environment.yml)
```
This installs `xtb-python`, which **bundles the xtb engine** — no separate xtb install.

> **Shared-cluster gotcha.** If `mamba env create` dies with
> `critical libmamba filesystem error: cannot set permissions … pkgs/cache`, the shared
> conda package cache is owned by another user. Point the cache at a dir you own and retry:
> ```bash
> mkdir -p $HOME/conda_pkgs
> CONDA_PKGS_DIRS=$HOME/conda_pkgs mamba env create -f environment.yml
> ```

Then activate it (also sets the fork/OpenMP + SCF safety flags):
```bash
source code/env.sh          # activates env 'xtbfit'; override XTBFIT_CONDA / XTBFIT_ENV if needed
```

---

## 1. The whole pipeline in one command
For the default system (`msrb_50rep`) on a big node, using 40 cores:
```bash
nohup bash run_all.sh > run_all.log 2>&1 &     # assemble → Max fit → plots → analysis
tail -f run_all.log                            # watch progress
```
`run_all.sh` runs everything sequentially (so it never exceeds `N_CPUS` cores). **Edit
`INPUT` / `OUTPUT` / `NCPUS` at the top of the script** (or export `XTBFIT_INPUT` /
`XTBFIT_OUTPUT`) — point them at your data and a results dir outside the repo. **The fit
self-limits** via `MNIWI=12` (stops after 12 generations without improvement).

If you'd rather run the steps yourself, or you have a **new system**, use the manual
steps below.

---

## 2. Manual steps (and what to run for a NEW system)

```bash
source code/env.sh
cd code
IN=/path/to/your/engrad         # folder of raw *_engrad.json (input)
OUT=$HOME/xtb_runs/mysystem      # where all results go (outside the repo)

# (a) DATA QC — ALWAYS run first on a new dataset. Finds duplicate/mislabeled files,
#     SCF-non-converging structures, and reference outliers; prints what to exclude.
python check_data.py -i $IN -o $OUT    # add --no-scf to skip the (parallel) xtb SCF scan

# (b) assemble compact engrad/ -> OUT/train/ + OUT/val/ curves
python assemble_dataset.py -i $IN -o $OUT

# (c) sanity + measured wall-time estimate on THIS machine
python dryrun.py -o $OUT

# (d) the re-fit  -> $OUT/param_gfn2-xtb.txt   (run under nohup/tmux)
nohup python fit.py -o $OUT > $OUT/run.log 2>&1 &

# (e) readable result plots (needs the param file from step d)
python make_report_plots.py -i $IN -o $OUT   # dataset_overview / profile_3way_averaged / …_per_node
python make_plots.py -o $OUT                 # built-in train plots (parity, errors, loss curve)

# (f) held-out validation (only if you kept a val split; see config)
python validate.py -o $OUT

# (g) generalization / extrapolation study (independent re-fits; see §5)
python analysis_cv.py all -i $IN -o $OUT
```
> `-s LABEL` optionally sets the display label (default: basename of `OUT`). Replica
> ranges / exclusions are still env/config (e.g. `XTBFIT_TRAIN_REPLICAS`, `XTBFIT_EXCLUDE`).

### For a brand-new system
1. Put its `*_engrad.json` in a folder anywhere (any subfolder layout; naming
   `node<N>_f<F>_engrad`, `N`=reaction-coordinate node, `F`=replica snapshot) → that's `-i`.
2. Pick a results dir outside the repo → that's `-o`. Set replica ranges to match its f-indices.
3. **Run `check_data.py -i … -o …` and act on what it reports** (quarantine duplicates,
   add SCF failures to `XTBFIT_EXCLUDE`) — see §4.
4. Run steps (b)–(g).

---

## 3. Configure (code/config.py, or env vars — no file edits needed)
| what | CLI / config | env override |
|---|---|---|
| **input** dir (raw engrad) | `-i INPUT` | `XTBFIT_INPUT` |
| **output** dir (all results) | `-o OUTPUT` | `XTBFIT_OUTPUT` |
| display label | `-s LABEL` / `SYSTEM` | `XTBFIT_SYSTEM` |
| training replicas (f-indices) | `TRAIN_REPLICAS` | `XTBFIT_TRAIN_REPLICAS="1-10"` / `"all"` |
| held-out validation replicas | `VAL_REPLICAS` | `XTBFIT_VAL_REPLICAS="11-20"` / `"none"` |
| drop bad structures | `EXCLUDE` | `XTBFIT_EXCLUDE="node1_f1,node1_f19"` |
| tunable parameter set | `paramset.py` | `XTBFIT_PARAMSET` (`HCONS` \| `HCONSSe`) |
| GA size / cores | `POP,ITERS,N_CPUS` | `XTBFIT_POP / _ITERS / _NCPUS` |
| SCF Fermi smearing | `XTB_ETEMP,XTB_MAXITER` | `XTB_ETEMP / XTB_MAXITER` |

Replica specs accept ranges (`"1-10"`), comma lists (`"1,2,3"`), and `all` / `none`.

**Which elements are tuned (`XTBFIT_PARAMSET`, default `HCONS`).** `HCONS` tunes H/C/N/O/S
(75 params) — use it for S-only systems. `HCONSSe` additionally tunes **Selenium** (`$Z=34`,
+20 params → 95), mirroring exactly how Sulfur is tuned — use it when the reactive center is
Se (e.g. a selenocysteine variant); with `HCONS`, Se stays frozen at stock GFN2 and the fit
cannot correct the reactive atom. The Se-extended template is built at import in
`code/paramset.py` (deps/ is not modified). Adding another element later = add its stock
values + field list there. **Re-derive `XTBFIT_EXCLUDE` per system** with `check_data.py`
(the defaults are msrb-specific).
Set `XTBFIT_TRAIN_REPLICAS=all XTBFIT_VAL_REPLICAS=none` to train on every replica.

---

## 4. Data quality control (important — read this)
The pipeline keys off each file's internal `label` field and needs stock GFN2 to converge
on every kept structure, so bad input silently breaks or biases the fit. `check_data.py`
catches three problems (all seen in `msrb_50rep`):

1. **Duplicate / mislabeled files** → a curve gets more energies than unique labels →
   objective returns `1e300` for *every* parameter set. *Fix:* quarantine the offending
   file out of `engrad/` (the script prints a `mv` command).
2. **SCF non-convergence with stock GFN2** → the GA baseline is non-finite, fit can't start.
   *Fix:* add those `nodeX_fY` to `XTBFIT_EXCLUDE` (or `config.py`).
3. **Reference outliers** (bad DFT SCF states = energy spikes; blown geometries = huge
   gradients; broken populations = net-charge drift) → reported conservatively for you to
   inspect. Skip only clear outliers — over-pruning biases the fit.

**Already applied to `msrb_50rep`:** `node6_f12_engrad.json` was a byte-identical duplicate
of `node6_f11` (quarantined to `data/msrb_50rep/_quarantine/`); `node1_f1` and `node1_f19`
are excluded (bad B3LYP SCF state / no-D4, and stock-GFN2 SCF failure).

---

## 5. Generalization analysis (`analysis_cv.py`)
Two independent cross-validation studies (each re-fits from scratch), separate from the
production run — outputs under `fit/<system>/analysis/`:
- **`random_split/`** — random 80/20 structure split (replicas of a node scattered across
  train/val) → **interpolation** quality; near-zero train→val gap = not overfit.
- **`node_cv/`** — K-fold over reaction-coordinate NODES, all replicas of a held-out node
  kept out → **extrapolation**. `per_node_error_map.png` shows the held-out error for every
  node = **where more DFT data would help**.

All metrics skip gross outliers (>10·MAD) and report medians. Knobs:
`XTBFIT_ANALYSIS_POP/_ITERS/_NCPUS/_KFOLDS/_RAND_REPLICAS/_NODE_REPLICAS/_SEED`.

---

## 6. Outputs (in `fit/<system>/`)
- **`param_gfn2-xtb.txt`** — the re-fitted parameters (**use this for MD**).
- `profile_3way_averaged.png` — mean DFT vs xTB-default vs re-fit reaction profile (+ residual).
- `profile_error_per_node.png` — per-node mean \|error\| (energy/charge/force), default vs re-fit.
- `dataset_overview.png` — DFT reference ensemble + per-node thermal spread.
- `parity_plots.png`, `error_distributions.png`, `loss_curve.png` — built-in train plots.
- `analysis/…` — the generalization study (§5).
- `fit_summary.txt`, `best_solution.txt`, `ga_logger.log`.

*(The built-in `profile_3way.png` draws one panel per replica and is unreadable for many
replicas; `profile_3way_averaged.png` / `profile_error_per_node.png` are the readable
replacements.)*

**Compact `*_engrad.json` schema** (one file per structure) — from ORCA
`! B3LYP D4 def2-TZVP EnGrad`:
```
label, n_atoms, charge, multiplicity, energy_hartree, orca_keywords,
atoms[]: {element, atomic_number, xyz_angstrom, mulliken_charge,
          loewdin_charge, gradient_hartree_per_bohr, force_hartree_per_bohr}
```

## Bigger machines / HPC
See **[RUN_ON_HPC.md](RUN_ON_HPC.md)** for scaling `POP`/`N_CPUS` and how wall-time scales.

## Notes on this dataset (msrb_50rep)
- MsrB capped QM region, 47 atoms (C₁₂H₂₇N₄O₂S₂), charge +1, B3LYP-D4/def2-TZVP.
- 48 nodes × 50 replicas; `node1_f1` (bad SCF/no-D4), `node1_f19` (stock-GFN2 SCF fail)
  excluded; `node6_f12` quarantined (duplicate). Usable: 2397 structures.
- Production params fit on **all 50 replicas** (POP=80): train energy MAE 3.86 → 1.66
  kcal/mol; averaged-profile systematic bias 3.57 → 0.55 kcal/mol.
