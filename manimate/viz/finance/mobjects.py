"""Small Mobject builders shared by the finance scenes.

Each builds a complete static picture (so it can fade in as a whole); animation is done by the scenes.
Intuition: a plot is a window onto a curve, and everything else (dots, tangents, readouts) is a marker pinned to it.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from manim import DOWN, LEFT, ORIGIN, RIGHT, UP, DashedLine, Dot, Line, Rectangle, Text, VGroup, VMobject

from ... import theme
from ...ctx import label


def nice_ticks(lo: float, hi: float, target: int = 5) -> list[float]:
    """Round tick values covering [lo, hi] (steps of 1, 2, 2.5, 5 times a power of ten)."""
    span = max(hi - lo, 1e-12)
    raw = span / target
    mag = 10 ** np.floor(np.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if raw <= m * mag)
    first = np.ceil(lo / step - 1e-9) * step
    out, v = [], first
    while v <= hi + 1e-9:
        out.append(float(round(v, 10)))
        v += step
    return out


def live_text(key_fn: Callable[[], object], build_fn: Callable[[object], Text]) -> VGroup:
    """Text rebuilt only when its key changes (creating Text every frame is slow)."""
    g = VGroup(build_fn(key_fn()))
    g._key = key_fn()

    def upd(m):
        k = key_fn()
        if k != m._key:
            m._key = k
            m.become(VGroup(build_fn(k)))

    g.add_updater(upd)
    return g


class Plot(VGroup):
    """Labelled axes with data-to-screen mapping ``p(x, y)``. The frame is an L (left and bottom edges) plus ticks.

    Intuition: a window onto a curve; markers pinned through ``p`` stay glued to the data when the window moves.
    """

    def __init__(self, x_range: tuple[float, float], y_range: tuple[float, float], *, width: float = 8.5,
                 height: float = 3.6, x_label: str = "", y_label: str = "", x_fmt: str = "{:g}", y_fmt: str = "{:g}",
                 x_ticks: Sequence[float] | None = None, y_ticks: Sequence[float] | None = None,
                 x_tick_labels: Sequence[str] | None = None, zero_line: bool = True, font: int = 18):
        super().__init__()
        if x_range[1] <= x_range[0] or y_range[1] <= y_range[0]:
            raise ValueError("plot ranges must have max > min")
        self.x0, self.x1 = x_range
        self.y0, self.y1 = y_range
        self._box = Rectangle(width=width, height=height, stroke_opacity=0, fill_opacity=0)
        parts: list = [self._box]
        left, bottom = self._box.get_left()[0], self._box.get_bottom()[1]
        parts.append(Line((left, bottom, 0), (left, bottom + height, 0), color=theme.MUTED, stroke_width=2.5))
        parts.append(Line((left, bottom, 0), (left + width, bottom, 0), color=theme.MUTED, stroke_width=2.5))
        xt = list(x_ticks) if x_ticks is not None else nice_ticks(self.x0, self.x1, 5)
        yt = list(y_ticks) if y_ticks is not None else nice_ticks(self.y0, self.y1, 4)
        for i, v in enumerate(xt):
            pt = self.p(v, self.y0)
            parts.append(Line(pt, pt + DOWN * 0.08, color=theme.MUTED, stroke_width=2))
            txt = x_tick_labels[i] if x_tick_labels is not None else x_fmt.format(0.0 if abs(v) < 1e-12 else v)
            parts.append(label(txt, font, theme.MUTED).next_to(pt, DOWN, buff=0.12))
        for v in yt:
            pt = self.p(self.x0, v)
            parts.append(Line(pt, pt + LEFT * 0.08, color=theme.MUTED, stroke_width=2))
            parts.append(label(y_fmt.format(0.0 if abs(v) < 1e-12 else v), font, theme.MUTED).next_to(pt, LEFT, buff=0.12))
        if zero_line and self.y0 < 0 < self.y1:
            parts.append(Line(self.p(self.x0, 0), self.p(self.x1, 0), color=theme.MUTED, stroke_width=1.5, stroke_opacity=0.7))
        if x_label:
            parts.append(label(x_label, 20, theme.MUTED).next_to(self._box.get_bottom(), DOWN, buff=0.5))
        if y_label:
            t = label(y_label, 20, theme.MUTED)
            t.next_to(self._box.get_top() + LEFT * (width / 2), UP, buff=0.12).align_to(self._box, LEFT)
            parts.append(t)
        self.add(*parts)

    def p(self, x: float, y: float, clamp: bool = True) -> np.ndarray:
        if clamp:
            y = min(max(y, self.y0 - 0.02 * (self.y1 - self.y0)), self.y1 + 0.02 * (self.y1 - self.y0))
        b = self._box
        fx = (x - self.x0) / (self.x1 - self.x0)
        fy = (y - self.y0) / (self.y1 - self.y0)
        return np.array([b.get_left()[0] + fx * b.width, b.get_bottom()[1] + fy * b.height, 0.0])

    def curve(self, points: Sequence[tuple[float, float]], color: str = theme.ACCENT, width: float = 4,
              opacity: float = 1.0) -> VMobject:
        m = VMobject(color=color, stroke_width=width, stroke_opacity=opacity)
        m.set_points_as_corners([self.p(x, y) for x, y in points])
        return m

    def dot(self, x: float, y: float, color: str = theme.YELLOW, radius: float = 0.09) -> Dot:
        return Dot(self.p(x, y), radius=radius, color=color)

    def hline(self, y: float, color: str = theme.MUTED, dashed: bool = True) -> Line:
        cls = DashedLine if dashed else Line
        return cls(self.p(self.x0, y), self.p(self.x1, y), color=color, stroke_width=2)

    def vline(self, x: float, color: str = theme.MUTED, dashed: bool = True) -> Line:
        cls = DashedLine if dashed else Line
        return cls(self.p(x, self.y0), self.p(x, self.y1), color=color, stroke_width=2)


def pad_range(lo: float, hi: float, frac: float = 0.12) -> tuple[float, float]:
    if hi - lo < 1e-9:
        lo, hi = lo - 1.0, hi + 1.0
    pad = (hi - lo) * frac
    return lo - pad, hi + pad
