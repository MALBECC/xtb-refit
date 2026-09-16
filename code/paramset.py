"""Single source of truth for the tunable GFN2-xTB parameter set.

  HCONS     (default, 75 params)  — H,C,N,O,S; the stock bundled template.
  HCONSSe   (95 params)           — additionally tunes Selenium ($Z=34): its 20 element
                                    fields become VARIABLE76#..VARIABLE95#, mirroring S.
  HCONSPMg  (115 params)          — additionally tunes Phosphorus ($Z=15 -> VARIABLE76#..95#)
                                    and Magnesium ($Z=12 -> VARIABLE96#..115#), for the RNA
                                    Mg-phosphate QM region. Both are $ao=3s3p3d elements with
                                    the identical 20-field layout as S/Se.
  HCONSPMgSe(135 params)          — P, Mg AND Se (VARIABLE76#..135#), all mirroring S.

Select with env  XTBFIT_PARAMSET  in {HCONS,HCONSSe,HCONSPMg,HCONSPMgSe}  (default HCONS).

Each extension INJECTS tokens into a COPY of the bundled `parm_HCONS` at import time, so
deps/ stays byte-for-byte unmodified. Template, PATTERNS, DIMENSION and the seed extension
are ALL derived here so lengths can never drift (XTBParam requires len(patterns)==len(values),
and the DDGA dimension must match). Imports only `parameters` (pure strings; no numpy/xtb),
so importing during config import — before common.py sets the OMP env — is safe.
"""
import os, re, sys

# repo root, computed independently of config (config imports THIS module -> avoid a cycle)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMFITTER = os.environ.get("PARAMFITTER", os.path.join(_ROOT, "deps", "gfn2-xtb_paramfitter"))
if PARAMFITTER not in sys.path:
    sys.path.insert(0, PARAMFITTER)

from parameters import parm_HCONS      # 75-token template (H,C,N,O,S); no heavy imports

# Every $ao=3s3p3d element (S, Se, P, Mg, ...) exposes the SAME 20 tunable fields in this
# file order:  lev(s,p,d) exp(s,p,d) GAM GAM3 KCNS KCNP KCND DPOL QPOL REPA REPB POLYS POLYP
#              POLYD LPARP LPARD  = 3+3+14 = 20
_FIELDS20 = [("lev", 3), ("exp", 3), ("GAM", 1), ("GAM3", 1), ("KCNS", 1),
             ("KCNP", 1), ("KCND", 1), ("DPOL", 1), ("QPOL", 1), ("REPA", 1),
             ("REPB", 1), ("POLYS", 1), ("POLYP", 1), ("POLYD", 1),
             ("LPARP", 1), ("LPARD", 1)]
_FLOAT = re.compile(r"-?\d+\.\d+")

# Stock element values in _FIELDS20 order (read from deps parm; keep in sync if deps changes).
SE_STOCK = [-20.584732, -10.910799,  -0.110636,   2.230969,   2.150656,   1.317549,
              0.235052,   0.913725,  -0.061654,  -0.435018,   2.768559,  -0.288648,
              0.085728,   1.230284,  27.426779, -24.506414, -13.765750,  29.611132,
              1.192113,  -2.500000]                                     # $Z=34
P_STOCK  = [-17.518756,  -9.842286,  -0.444893,   1.816945,   1.903247,   1.167533,
              0.297739,   0.711291,   0.547610,  -0.489930,   2.429507,   2.110225,
              0.028679,   1.143343,  19.683502, -19.831771,  -5.515577,  26.397535,
             -1.558060,  -3.500000]                                     # $Z=15
MG_STOCK = [ -6.339908,  -0.697688,  -1.458197,   1.184203,   0.717769,   1.300000,
              0.344822,   2.349164,   1.164444,  -0.079924,   1.192409,  -0.082005,
             -0.005516,   0.917975,  18.083164, -11.167374,  39.076962,  12.691061,
             14.000000,  -0.500000]                                     # $Z=12
for _s in (SE_STOCK, P_STOCK, MG_STOCK):
    assert len(_s) == 20


def _inject_block(template, Z, start_tok):
    """Tokenize the $Z=<Z> block's 20 fields as VARIABLE<start_tok>#.. in a COPY of `template`.
    Returns (new_template, next_tok). Heavily asserted so a deps change fails loudly."""
    assert len(re.findall(rf"\$Z={Z}\b", template)) == 1, f"expected one $Z={Z} block"
    m = re.search(rf"\$Z={Z}\b.*?\$end", template, re.S)
    assert m, f"$Z={Z}..$end not found"
    block = m.group(0); lines = block.split("\n")
    tok = start_tok
    for label, n in _FIELDS20:
        for li, line in enumerate(lines):
            if re.match(rf"\s*{re.escape(label)}=", line):     # '=' anchor: GAM != GAM3
                floats = _FLOAT.findall(line)
                assert len(floats) == n, f"Z={Z} {label}: expected {n} floats, saw {len(floats)}"
                for _ in range(n):
                    line = _FLOAT.sub(f"VARIABLE{tok}#", line, count=1); tok += 1
                lines[li] = line; break
        else:
            raise AssertionError(f"Z={Z} field '{label}=' not found")
    new_block = "\n".join(lines)
    assert template.count(block) == 1, f"$Z={Z} block not uniquely locatable"
    return template.replace(block, new_block), tok


def _build(elements):
    """elements = ordered [(Z, stock), ...]. Inject each block starting at VARIABLE76#,
    contiguously. Returns (template, dimension, seed_extra)."""
    template, tok, seed = parm_HCONS, 76, []
    for Z, stock in elements:
        template, tok = _inject_block(template, Z, tok)
        seed += list(stock)
    dim = tok - 1
    toks = set(re.findall(r"VARIABLE\d+#", template))
    assert toks == {f"VARIABLE{i}#" for i in range(1, dim + 1)}, f"token set != 1..{dim}"
    assert len(seed) == dim - 75, "seed length mismatch"
    return template, dim, seed


_SETS = {
    "HCONS":      ([],                                 ),
    "HCONSSe":    ([(34, SE_STOCK)],                   ),
    "HCONSPMg":   ([(15, P_STOCK), (12, MG_STOCK)],    ),
    "HCONSPMgSe": ([(15, P_STOCK), (12, MG_STOCK), (34, SE_STOCK)],),
}

PARAMSET = os.environ.get("XTBFIT_PARAMSET", "HCONS")
if PARAMSET not in _SETS:
    raise ValueError(f"XTBFIT_PARAMSET must be one of {sorted(_SETS)}, got {PARAMSET!r}")

if PARAMSET == "HCONS":
    PARM, DIMENSION, SEED_EXTRA = parm_HCONS, 75, []
else:
    PARM, DIMENSION, SEED_EXTRA = _build(*_SETS[PARAMSET])

PATTERNS = [f"VARIABLE{i}#" for i in range(1, DIMENSION + 1)]

# Which elements the EXTRA tokens (VARIABLE76#..) belong to, in token order. Derived from
# the same _SETS entry that built the template, so a label can never disagree with what was
# actually injected. HCONS tunes nothing beyond the stock H/C/N/O/S -> empty list.
_SYMBOL = {12: "Mg", 15: "P", 16: "S", 34: "Se"}
EXTRA_ELEMENTS = [_SYMBOL[Z] for Z, _ in _SETS[PARAMSET][0]]
EXTRA_LABEL    = "+".join(EXTRA_ELEMENTS) if EXTRA_ELEMENTS else "none"
