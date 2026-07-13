#!/usr/bin/env python3
"""Assemble compact ORCA *_engrad.json files -> dataset_f*.json curve files, split
into a TRAIN set and (optionally) a held-out VALIDATION set by replica f-index.

Reads everything from config.py:
  ENGRAD_DIR      where the compact *_engrad.json live
  TRAIN_REPLICAS  f-indices for training  (None -> use ALL replicas, no val split)
  VAL_REPLICAS    f-indices held out for validation (disjoint from train)
  EXCLUDE         per-structure substrings to drop (bad reference data)
  TRAIN_DIR/VAL_DIR  output dirs

Compact-file schema (per structure): label, n_atoms, charge, multiplicity,
energy_hartree, atoms[]{atomic_number, xyz_angstrom, mulliken_charge,
gradient_hartree_per_bohr}. UNITS written to dataset_f*.json (JSONCurveHandler):
energies kcal/mol, coordinates Bohr, gradients dE/dx Eh/Bohr, charges e.

Usage:  python assemble_dataset.py        (uses config)
"""
import os, sys, json, re, glob
import config as C

LABEL_RE = re.compile(r"node(\d+)_f(\d+)", re.I)


def find_compact(root):
    hits = []
    for p in sorted(glob.glob(os.path.join(root, "**", "*_engrad.json"), recursive=True)):
        try:
            d = json.load(open(p))
        except (OSError, ValueError):
            continue
        if isinstance(d, dict) and "energy_hartree" in d and "atoms" in d:
            hits.append(p)
    return hits


def parse_one(path):
    d = json.load(open(path))
    label = d.get("label") or os.path.splitext(os.path.basename(path))[0]
    m = LABEL_RE.search(label)
    node = int(m.group(1)) if m else 0
    fidx = int(m.group(2)) if m else 1
    atoms = d["atoms"]
    return {
        "label": label, "node": node, "fidx": fidx,
        "chrg": int(d["charge"]), "mult": int(d["multiplicity"]),
        "energy_kcal": float(d["energy_hartree"]) * C.HARTREE_TO_KCAL_MOL,
        "Z":     [int(a["atomic_number"]) for a in atoms],
        "coords_bohr": [[c * C.ANGSTROM_TO_BOHR for c in a["xyz_angstrom"]] for a in atoms],
        "charges":     [float(a["mulliken_charge"]) for a in atoms],
        "gradient":    [list(map(float, a["gradient_hartree_per_bohr"])) for a in atoms],
        "qsum": sum(float(a["mulliken_charge"]) for a in atoms),
    }


def write_split(recs, out_dir, name):
    if not recs:
        print(f"  {name}: (no structures)"); return []
    curves = {}
    for r in recs:
        curves.setdefault(r["fidx"], []).append(r)
    os.makedirs(out_dir, exist_ok=True)
    for f in glob.glob(os.path.join(out_dir, "dataset_f*.json")):
        os.remove(f)                                  # clean stale curves
    written = []
    for fidx in sorted(curves):
        nodes = sorted(curves[fidx], key=lambda r: r["node"])
        ds = {
            "energies":       [r["energy_kcal"] for r in nodes],
            "atomic_numbers": {r["label"]: r["Z"] for r in nodes},
            "coordinates":    {r["label"]: r["coords_bohr"] for r in nodes},
            "charges":        {r["label"]: r["charges"] for r in nodes},
            "gradients":      {r["label"]: r["gradient"] for r in nodes},
            "system": {"chrg": {r["label"]: r["chrg"] for r in nodes},
                       "mult": {r["label"]: r["mult"] for r in nodes}},
        }
        out = os.path.join(out_dir, f"dataset_f{fidx}.json")
        json.dump(ds, open(out, "w")); written.append(out)
    nstruct = sum(len(v) for v in curves.values())
    bad = [f"{r['label']}({r['qsum']:+.2f})" for r in recs if abs(r["qsum"] - r["chrg"]) > 1e-2]
    print(f"  {name}: {len(curves)} curves / {nstruct} structures -> {out_dir}"
          + (f"   [WARN net-charge off: {bad[:5]}]" if bad else ""))
    return written


def main():
    print(C.summary(), "\n")
    files = find_compact(C.ENGRAD_DIR)
    if not files:
        sys.exit(f"No compact *_engrad.json under {C.ENGRAD_DIR}")
    recs = [parse_one(p) for p in files]
    n0 = len(recs)
    if C.EXCLUDE:
        # exact match on the node/f token (or full label) — NOT substring, so "node1_f1"
        # does not accidentally catch node1_f10..f19
        def is_excluded(r):
            tok = f"node{r['node']}_f{r['fidx']}"
            return any(x == tok or x == r["label"] for x in C.EXCLUDE)
        dropped = [r["label"] for r in recs if is_excluded(r)]
        recs = [r for r in recs if not is_excluded(r)]
        print(f"parsed {n0} structures; EXCLUDED {len(dropped)}: {dropped[:8]}"
              + (" ..." if len(dropped) > 8 else ""))

    if C.TRAIN_REPLICAS is None:                       # use everything for training
        train = recs; val = []
    else:
        tr, vr = set(C.TRAIN_REPLICAS), set(C.VAL_REPLICAS or [])
        overlap = tr & vr
        assert not overlap, f"train/val replica overlap: {overlap}"
        train = [r for r in recs if r["fidx"] in tr]
        val   = [r for r in recs if r["fidx"] in vr]
    print(f"replicas -> train {sorted({r['fidx'] for r in train})} | "
          f"val {sorted({r['fidx'] for r in val})}\n")
    write_split(train, C.TRAIN_DIR, "TRAIN")
    write_split(val,   C.VAL_DIR,   "VAL")


if __name__ == "__main__":
    main()
