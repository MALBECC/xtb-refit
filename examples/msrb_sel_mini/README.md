# `msrb_sel_mini` — the supported input format, as a runnable example

This is the **reference example of the input this pipeline consumes**: per-structure
compact `*_engrad.json`. Everything here runs in a few minutes on a laptop, so it
doubles as a smoke test after changing `code/`.

**26 structures** — 13 reaction-coordinate nodes (1, 5, 9, … 45, 48) × 2 replicas
(`f10`, `f11`), subsampled from the full `msrb_sel_50rep` set (48 nodes × 50 replicas).
System: **MsrB selenocysteine variant**, 47 atoms (C₁₂H₂₇N₄O₂S₁**Se₁**), charge +1,
multiplicity 1. Reference level **B3LYP-D4/def2-TZVP** (see each file's `orca_keywords`).

`f10` lands in TRAIN and `f11` in VAL under the default replica split, so the commands
below work with no configuration at all.

## Run it

```bash
OUT=/tmp/msrb_sel_mini                    # anywhere outside the repo
export XTBFIT_PARAMSET=HCONSSe            # tune Se too (95 params); omit for HCONS (75)

python code/check_data.py       -i examples/msrb_sel_mini/engrad -o $OUT   # QC
python code/assemble_dataset.py -i examples/msrb_sel_mini/engrad -o $OUT   # -> train/ + val/
python code/dryrun.py                                            -o $OUT   # sanity + timing
python code/fit.py                                               -o $OUT   # ~6 min, POP=24 ITERS=30
```

Expected from `dryrun.py` (13 train structures, 12 cores):

```
train curves 1 / 13 structures   paramset=HCONSSe (95 params)
default   SCORE=9.078e+18  GRAD=109.010 CHRG=1.124 E=7.410e+16
pipeline runs      : YES
params take hold   : YES
Se params take hold: YES  (SCORE 9.078e+18 -> 3.887e+21)
timing: 1 evaluation ~2-4s -> ~0.1 min/generation -> minutes for 30 generations
```

A baseline SCORE around 1e18 is **normal** — the energy term is unbounded until the
parameters are close. See `common.warn_objective_scale()` for what an actually-broken
dataset looks like.

> These 26 structures are a demo, not a production training set. Parameters fitted here
> are for checking the machinery runs, not for MD.

## The format

One JSON per structure. `assemble_dataset.py` finds them recursively, so any subfolder
layout works (here: `engrad/node<N>/node<N>_f<F>_engrad.json`).

```
label, n_atoms, charge, multiplicity, energy_hartree, orca_keywords,
atoms[]: {element, atomic_number, xyz_angstrom, mulliken_charge,
          loewdin_charge, gradient_hartree_per_bohr, force_hartree_per_bohr}
```

**Required** (the pipeline reads nothing else): `charge`, `multiplicity`,
`energy_hartree`, `atoms[]`, and per atom `atomic_number`, `xyz_angstrom`,
`mulliken_charge`, `gradient_hartree_per_bohr`. `label` is optional and falls back to
the filename. The remaining fields are carried for provenance and ignored.

Units: energy **Hartree**, coordinates **Ångström**, gradient (dE/dx) **Eh/Bohr**,
charges **e**. `assemble_dataset.py` converts to the fitter's units (kcal/mol, Bohr).

### Two naming rules

1. The filename must end in **`_engrad.json`**.
2. The label must contain **`node<N>_f<F>`** — `N` = reaction-coordinate node,
   `F` = replica. This is not cosmetic: replicas are grouped into energy curves by `F`,
   and nodes ordered by `N`. **If the pattern does not match, every structure silently
   collapses into one curve `dataset_f1.json`** and the fit is meaningless.
   `check_data.py` catches label/filename mismatches — run it first on any new dataset.

## Producing these from ORCA

The reference calculation needs the **`EnGrad`** keyword — that is what makes ORCA write
the `.engrad` file holding the energy and gradient:

```
! B3LYP D4 def2-TZVP EnGrad
```

No single ORCA file has everything, so a conversion step is required:

| source | energy | gradient | geometry | charges |
|---|:--:|:--:|:--:|:--:|
| `.engrad` (from `EnGrad`) | ✓ | ✓ | ✓ (Bohr) | ✗ |
| `.out` | ✓ | ✓ | ✓ | ✓ Mulliken / Löwdin |
| `_property.txt` (ORCA 5.0.4) | ✓ | ✗ | ✓ | Mayer only |

So the converter merges the `.engrad` (energy + gradient + geometry) with the charge
block parsed from the `.out`. Mulliken charges need no extra keyword — ORCA prints them
by default — but do not suppress output printing, and note that the **last** such block
is the one to take.

On **ORCA 6** this is simpler: the property JSON carries `SCF_Energy`,
`Mulliken_Population_Analysis`, `Loewdin_Population_Analysis`, `Geometry` **and**
`Nuclear_Gradient` in one file, so no merge is needed. That is the route the bundled
upstream notebook uses (`deps/gfn2-xtb_paramfitter/Notebooks/Compile_Orca6_Dataset.ipynb`).
ORCA 5.0.4's `_property.txt` has no gradient section, which is why the two-file merge
exists here.
