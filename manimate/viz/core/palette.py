"""Colours for every viz component: the repo theme, plus teal, colormaps and phase colours.

Blue is the primary signal, yellow marks a derived quantity or "look here", teal is structure and secondary signal, red
is error / gradient / fault, green is success and hit. Existing ``theme`` names are reused, never redefined.
"""
from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass

from .. import theme_bridge as theme


@dataclass(frozen=True)
class Palette:
    bg: str = theme.BG
    fg: str = theme.FG
    muted: str = theme.MUTED
    panel: str = theme.PANEL
    blue: str = theme.ACCENT
    yellow: str = theme.YELLOW
    teal: str = "#2dd4bf"
    red: str = theme.DANGER
    green: str = theme.GOOD
    purple: str = theme.PURPLE
    orange: str = theme.WARN

    @property
    def text_colors(self) -> tuple[str, ...]:
        """Every colour that may be used for text; the style test checks their contrast against the background."""
        return (self.fg, self.muted, self.blue, self.yellow, self.teal, self.red, self.green, self.purple, self.orange)


PALETTE = Palette()

# Sequential and diverging maps as evenly spaced stops. The low end is lifted off black so a "zero" cell is still visible.
_MAPS: dict[str, list[str]] = {
    "viridis": ["#46327e", "#365c8d", "#277f8e", "#1fa187", "#4ac16d", "#a0da39", "#fde725"],
    "magma": ["#3b0f70", "#641a80", "#8c2981", "#b73779", "#de4968", "#f7705c", "#fe9f6d", "#fecf92"],
    "coolwarm": ["#4f7fd6", "#8fb0ea", "#cbd8ee", "#e8d8d0", "#f0a98c", "#e0694f", "#b8352d"],
    "blue_yellow": ["#1d4f91", "#58a6ff", "#8fc9c0", "#d9d38a", "#f5e27a"],
}
COLORMAPS = tuple(_MAPS)


def hex_to_rgb(c: str) -> tuple[float, float, float]:
    c = c.lstrip("#")
    if len(c) != 6:
        raise ValueError(f"expected a #rrggbb colour, got {c!r}")
    return tuple(int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{round(max(0.0, min(1.0, v)) * 255):02x}" for v in rgb)


def lerp_color(a: str, b: str, t: float) -> str:
    """Linear blend of two hex colours; ``t`` is clamped to [0, 1]."""
    t = max(0.0, min(1.0, float(t)))
    ra, rb = hex_to_rgb(a), hex_to_rgb(b)
    return rgb_to_hex(tuple(x + (y - x) * t for x, y in zip(ra, rb)))  # type: ignore[arg-type]


def colormap(name: str, v: float, vmin: float = 0.0, vmax: float = 1.0) -> str:
    """Hex colour for ``v`` on the named map. Out-of-range values clamp; NaN gives the muted colour."""
    if name not in _MAPS:
        raise ValueError(f"unknown colormap {name!r}; choose one of {list(COLORMAPS)}")
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return PALETTE.muted
    span = vmax - vmin
    u = 0.5 if span == 0 else (float(v) - vmin) / span
    u = max(0.0, min(1.0, u))
    stops = _MAPS[name]
    pos = u * (len(stops) - 1)
    i = min(int(pos), len(stops) - 2)
    return lerp_color(stops[i], stops[i + 1], pos - i)


def phase_to_color(theta: float, saturation: float = 0.65, value: float = 0.95) -> str:
    """Hue encodes phase angle (radians): used for complex amplitudes so interference is visible."""
    hue = (theta % (2 * math.pi)) / (2 * math.pi)
    return rgb_to_hex(colorsys.hsv_to_rgb(hue, saturation, value))


def relative_luminance(c: str) -> float:
    """WCAG relative luminance of a hex colour."""
    def lin(u: float) -> float:
        return u / 12.92 if u <= 0.03928 else ((u + 0.055) / 1.055) ** 2.4

    r, g, b = (lin(u) for u in hex_to_rgb(c))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(fg: str, bg: str = PALETTE.bg) -> float:
    """WCAG contrast ratio, from 1 (identical) to 21 (black on white)."""
    la, lb = relative_luminance(fg), relative_luminance(bg)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)
