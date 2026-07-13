"""Single source of truth for the tunable GFN2-xTB parameter set.

  HCONS    (default, 75 params)  — H,C,N,O,S; the stock bundled template.
  HCONSSe  (95 params)           — additionally tunes Selenium ($Z=34): its 20 element
                                   fields become VARIABLE76#..VARIABLE95#, mirroring
                                   exactly how Sulfur ($Z=16 -> VARIABLE56#..75#) is tuned.

Select with env  XTBFIT_PARAMSET  in {"HCONS","HCONSSe"}  (default "HCONS").

The Se-extended template is built by INJECTING tokens into a COPY of the bundled
`parm_HCONS` at import time, so deps/ stays byte-for-byte unmodified. Template, PATTERNS,
DIMENSION and the seed extension are ALL derived here so the lengths can never drift
(XTBParam requires len(patterns)==len(values), and the DDGA dimension must match too).
This module imports only `parameters` (pure strings, no numpy/xtb), so importing it during
config import — before common.py sets the OMP env — is safe.
"""
import os, re, sys

# repo root, computed independently of config (config imports THIS module -> avoid a cycle)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMFITTER = os.environ.get("PARAMFITTER", os.path.join(_ROOT, "deps", "gfn2-xtb_paramfitter"))
if PARAMFITTER not in sys.path:
    sys.path.insert(0, PARAMFITTER)

from parameters import parm_HCONS      # 75-token template (H,C,N,O,S); no heavy imports

# Stock $Z=34 (Se) values in the SAME field order Sulfur uses for VARIABLE56#..75#:
#   lev(3) exp(3) GAM GAM3 KCNS KCNP KCND DPOL QPOL REPA REPB POLYS POLYP POLYD LPARP LPARD
SE_STOCK = [-20.584732, -10.910799,  -0.110636,      # lev  (s,p,d)
              2.230969,   2.150656,   1.317549,      # exp  (s,p,d)
              0.235052,                              # GAM
              0.913725,                              # GAM3
             -0.061654,                              # KCNS
             -0.435018,                              # KCNP
              2.768559,                              # KCND
             -0.288648,                              # DPOL
              0.085728,                              # QPOL
              1.230284,                              # REPA
             27.426779,                              # REPB
            -24.506414,                              # POLYS
            -13.765750,                              # POLYP
             29.611132,                              # POLYD
              1.192113,                              # LPARP
             -2.500000]                              # LPARD
assert len(SE_STOCK) == 20

# (label, n_values) in file order; 3+3+14 = 20 fields
_SE_FIELDS = [("lev", 3), ("exp", 3), ("GAM", 1), ("GAM3", 1), ("KCNS", 1),
              ("KCNP", 1), ("KCND", 1), ("DPOL", 1), ("QPOL", 1), ("REPA", 1),
              ("REPB", 1), ("POLYS", 1), ("POLYP", 1), ("POLYD", 1),
              ("LPARP", 1), ("LPARD", 1)]
_FLOAT = re.compile(r"-?\d+\.\d+")


def _build_hconsse():
    """Return a copy of parm_HCONS with the $Z=34 (Se) block's 20 fields tokenized
    VARIABLE76#..VARIABLE95# (mirroring S). Heavily asserted so a future deps change
    fails loudly instead of silently producing a wrong template."""
    assert len(re.findall(r"\$Z=34\b", parm_HCONS)) == 1, "expected exactly one $Z=34 block"
    m = re.search(r"\$Z=34\b.*?\$end", parm_HCONS, re.S)
    assert m, "Se block $Z=34..$end not found"
    block = m.group(0)
    lines = block.split("\n")

    tok = 76
    replaced = 0
    for label, n in _SE_FIELDS:
        for li, line in enumerate(lines):
            if re.match(rf"\s*{re.escape(label)}=", line):      # '=' anchor: GAM != GAM3
                floats = _FLOAT.findall(line)
                assert len(floats) == n, f"{label}: expected {n} floats, saw {len(floats)} in {line!r}"
                for _ in range(n):
                    line = _FLOAT.sub(f"VARIABLE{tok}#", line, count=1)
                    tok += 1
                lines[li] = line
                replaced += 1
                break
        else:
            raise AssertionError(f"Se field '{label}=' not found in $Z=34 block")

    assert replaced == len(_SE_FIELDS), f"expected {len(_SE_FIELDS)} Se labels, matched {replaced}"
    assert tok == 96, f"expected to consume 20 values -> VARIABLE76..95, ended at {tok}"
    new_block = "\n".join(lines)

    assert parm_HCONS.count(block) == 1, "Se block not uniquely locatable for substitution"
    template = parm_HCONS.replace(block, new_block)

    toks = re.findall(r"VARIABLE\d+#", template)
    assert set(toks) == {f"VARIABLE{i}#" for i in range(1, 96)}, "token set is not exactly 1..95"
    for i in range(76, 96):
        assert new_block.count(f"VARIABLE{i}#") == 1, f"VARIABLE{i}# not placed exactly once"
    assert template != parm_HCONS, "Se injection produced no change"
    return template


PARAMSET = os.environ.get("XTBFIT_PARAMSET", "HCONS")
if PARAMSET == "HCONS":
    PARM, DIMENSION, SEED_EXTRA = parm_HCONS, 75, []
elif PARAMSET == "HCONSSe":
    PARM, DIMENSION, SEED_EXTRA = _build_hconsse(), 95, list(SE_STOCK)
else:
    raise ValueError(f"XTBFIT_PARAMSET must be 'HCONS' or 'HCONSSe', got {PARAMSET!r}")

PATTERNS = [f"VARIABLE{i}#" for i in range(1, DIMENSION + 1)]
