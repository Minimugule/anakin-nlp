"""Chart palette + a Python port of the five computable colour checks.

Colour choices in this project are validated, not eyeballed. The checks below are
a faithful port of the reference JS validator: OKLab/OKLCH conversions, the
Machado-Oliveira-Fernandes (2009) colour-vision-deficiency transforms at severity
1.0, and WCAG contrast against the chart surface.

Run ``python -m anakin_nlp.palette`` to print the report for this project's palette.
"""

from __future__ import annotations

import math
from itertools import combinations

# --- thresholds (identical to the reference validator) -----------------------
BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}  # OKLCH L
CHROMA_FLOOR = 0.10
CVD_TARGET, CVD_FLOOR = 8.0, 6.0  # OKLab dE x100, min(protan, deutan)
NORMAL_FLOOR = 15.0  # OKLab dE x100, unsimulated vision - hard gate
CONTRAST_MIN = 3.0
SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}

MACHADO = {
    "protan": ((0.152286, 1.052583, -0.204868),
               (0.114503, 0.786281, 0.099216),
               (-0.003882, -0.048116, 1.051998)),
    "deutan": ((0.367322, 0.860646, -0.227968),
               (0.280085, 0.672501, 0.047413),
               (-0.011820, 0.042940, 0.968881)),
    "tritan": ((1.255528, -0.076749, -0.178779),
               (-0.078411, 0.930809, 0.147602),
               (0.004733, 0.691367, 0.303900)),
}

# --- project palette ---------------------------------------------------------
# Slots are drawn from the reference categorical theme. The assignment is
# semantic and fixed: a character keeps its colour in every figure of the report.
INK = {
    "surface": "#fcfcfb",
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
}

# Validated as a set with pairs="all": worst CVD dE 13.0, worst normal-vision dE
# 16.3 (both clear the gates). Yellow sits below 3:1 on the light surface, so the
# relief rule applies - every series is direct-labelled, never colour-alone.
CHARACTER_COLORS = {
    "ANAKIN": "#e34948",   # reference slot 8 (red) - the subject
    "OBI-WAN": "#2a78d6",  # slot 1 (blue)   - light-side control
    "PADME": "#eda100",    # slot 4 (yellow) - light-side control
    "PALPATINE": "#4a3aa7",  # slot 7 (violet) - dark-side control
}

# P(dark side) is a polarity around 0.5, so it takes the documented diverging
# pair: blue <-> red with a neutral grey midpoint.
DIVERGING = ("#2a78d6", "#f0efec", "#e34948")

LIGHT_SIDE = "#2a78d6"
DARK_SIDE = "#e34948"


# --- colour maths ------------------------------------------------------------
def _srgb(hex_color: str) -> tuple[float, float, float]:
    h = hex_color.strip().lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def _to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _lin(hex_color: str) -> tuple[float, float, float]:
    return tuple(_to_linear(c) for c in _srgb(hex_color))


def _oklab_from_lin(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    r, g, b = rgb
    l = math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m = math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s = math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)


def oklch(hex_color: str) -> tuple[float, float]:
    L, a, b = _oklab_from_lin(_lin(hex_color))
    return L, math.hypot(a, b)


def contrast(a: str, b: str) -> float:
    def lum(h):
        r, g, bl = _lin(h)
        return 0.2126 * r + 0.7152 * g + 0.0722 * bl
    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _simulate(hex_color: str, kind: str) -> tuple[float, float, float]:
    rgb, M = _lin(hex_color), MACHADO[kind]
    return tuple(min(1.0, max(0.0, sum(M[i][k] * rgb[k] for k in range(3)))) for i in range(3))


def delta_e(h1: str, h2: str, kind: str | None = None) -> float:
    a = _oklab_from_lin(_simulate(h1, kind) if kind else _lin(h1))
    b = _oklab_from_lin(_simulate(h2, kind) if kind else _lin(h2))
    return 100 * math.dist(a, b)


def validate(palette: list[str], mode: str = "light", pairs: str = "all") -> tuple[list, bool]:
    """Return (report rows, ok). ``pairs='all'`` is the strict list used for
    scatter/small-multiples; 'adjacent' is enough for stacked bars and lines."""
    surface = SURFACE[mode]
    lo, hi = BAND[mode]
    report, ok = [], True

    offband = [(c, round(oklch(c)[0], 3)) for c in palette if not lo <= oklch(c)[0] <= hi]
    ok &= not offband
    report.append(("Lightness band", "FAIL" if offband else "pass",
                   f"outside {lo}-{hi}: {offband}" if offband else f"all {len(palette)} inside"))

    lowc = [(c, round(oklch(c)[1], 3)) for c in palette if oklch(c)[1] < CHROMA_FLOOR]
    ok &= not lowc
    report.append(("Chroma floor", "FAIL" if lowc else "pass",
                   f"reads grey: {lowc}" if lowc else f"all >= {CHROMA_FLOOR}"))

    idx = list(combinations(range(len(palette)), 2)) if pairs == "all" \
        else [(i, i + 1) for i in range(len(palette) - 1)]

    worst = min(((delta_e(palette[i], palette[j], k), k, palette[i], palette[j])
                 for k in ("protan", "deutan") for i, j in idx), default=(99, "-", "", ""))
    tri = min((delta_e(palette[i], palette[j], "tritan") for i, j in idx), default=99)
    state = "pass" if worst[0] >= CVD_TARGET else "WARN" if worst[0] >= CVD_FLOOR else "FAIL"
    ok &= state != "FAIL"
    report.append(("CVD separation", state,
                   f"worst {pairs} {worst[2]}<->{worst[3]} dE {worst[0]:.1f} ({worst[1]}) - tritan {tri:.1f}"))

    nworst = min(((delta_e(palette[i], palette[j]), palette[i], palette[j]) for i, j in idx),
                 default=(99, "", ""))
    nstate = "pass" if nworst[0] >= NORMAL_FLOOR else "FAIL"
    ok &= nstate != "FAIL"
    report.append(("Normal-vision floor", nstate,
                   f"worst {nworst[1]}<->{nworst[2]} dE {nworst[0]:.1f} (floor {NORMAL_FLOOR:.0f})"))

    low = [(c, round(contrast(c, surface), 2)) for c in palette if contrast(c, surface) < CONTRAST_MIN]
    report.append(("Contrast vs surface", "relief" if low else "pass",
                   f"sub-{CONTRAST_MIN}:1, needs visible labels: {low}" if low
                   else f"all >= {CONTRAST_MIN}:1"))
    return report, ok


if __name__ == "__main__":
    pal = list(CHARACTER_COLORS.values())
    print(f"Palette: {', '.join(f'{k} {v}' for k, v in CHARACTER_COLORS.items())}\n")
    for mode in ("light",):
        rows, ok = validate(pal, mode=mode, pairs="all")
        print(f"--- mode={mode}, pairs=all ---")
        for name, state, detail in rows:
            print(f"  {state:<6} {name:<22} {detail}")
        print(f"  => {'OK' if ok else 'FAILED'}\n")
