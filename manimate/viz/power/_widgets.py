"""Small drawing helpers shared by the power visuals: schematic symbols and a mini oscilloscope."""
from __future__ import annotations

import numpy as np
from manim import Arc, Circle, Dot, Line, Polygon, Rectangle, Text, VGroup, VMobject

from ..core.palette import PALETTE

LEAD = 0.18  # fraction of a symbol's length used for the leads at each end


def _frame(a, b):
    """Unit vector, length and rotation angle of the segment a -> b."""
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    v = b - a
    length = float(np.linalg.norm(v))
    return a, v / length, length, float(np.arctan2(v[1], v[0]))


def wire(a, b, color: str = PALETTE.muted, width: float = 3.0) -> Line:
    return Line(a, b, color=color, stroke_width=width)


def resistor(a, b, color: str = PALETTE.muted, n: int = 6) -> VGroup:
    a, u, L, _ = _frame(a, b)
    nrm = np.array([-u[1], u[0], 0.0])
    lead = LEAD * L
    pts = [a, a + u * lead]
    body = L - 2 * lead
    for k in range(n):
        pts.append(a + u * (lead + body * (k + 0.5) / n) + nrm * (0.11 if k % 2 == 0 else -0.11))
    pts += [a + u * (L - lead), a + u * L]
    return VGroup(VMobject(color=color, stroke_width=3).set_points_as_corners(pts))


def inductor(a, b, color: str = PALETTE.muted, loops: int = 4) -> VGroup:
    a, u, L, ang = _frame(a, b)
    lead = LEAD * L
    r = (L - 2 * lead) / (2 * loops)
    g = VGroup(Line(a, a + u * lead, color=color, stroke_width=3), Line(a + u * (L - lead), a + u * L, color=color, stroke_width=3))
    for k in range(loops):
        c = a + u * (lead + r * (2 * k + 1))
        arc = Arc(radius=r, start_angle=np.pi, angle=-np.pi, color=color, stroke_width=3).move_to(c + np.array([-u[1], u[0], 0.0]) * 0 )
        arc.rotate(ang, about_point=arc.get_center()).shift(np.array([-u[1], u[0], 0.0]) * r * 0.0)
        # semicircle bulging to the left of travel direction
        arc.move_to(c + np.array([-u[1], u[0], 0.0]) * (r * 0.64))
        g.add(arc)
    return g


def capacitor(a, b, color: str = PALETTE.muted) -> VGroup:
    a, u, L, _ = _frame(a, b)
    nrm = np.array([-u[1], u[0], 0.0])
    m1, m2 = a + u * (L / 2 - 0.07), a + u * (L / 2 + 0.07)
    return VGroup(Line(a, m1, color=color, stroke_width=3), Line(m2, a + u * L, color=color, stroke_width=3),
                  Line(m1 - nrm * 0.3, m1 + nrm * 0.3, color=color, stroke_width=3), Line(m2 - nrm * 0.3, m2 + nrm * 0.3, color=color, stroke_width=3))


def diode(a, b, color: str = PALETTE.muted) -> VGroup:
    """Anode at ``a``, cathode at ``b``."""
    a, u, L, _ = _frame(a, b)
    nrm = np.array([-u[1], u[0], 0.0])
    t0, t1 = a + u * (L / 2 - 0.16), a + u * (L / 2 + 0.16)
    tri = Polygon(t0 + nrm * 0.17, t0 - nrm * 0.17, t1, color=color, stroke_width=3, fill_opacity=0)
    return VGroup(Line(a, t0, color=color, stroke_width=3), tri, Line(t1 - nrm * 0.17, t1 + nrm * 0.17, color=color, stroke_width=3),
                  Line(t1, a + u * L, color=color, stroke_width=3))


def switch(a, b, closed: bool, color: str = PALETTE.muted) -> VGroup:
    """A lever from ``a`` towards ``b``; open tilts it away from the contact."""
    a, u, L, _ = _frame(a, b)
    nrm = np.array([-u[1], u[0], 0.0])
    p0, p1 = a + u * 0.22 * L, a + u * 0.78 * L
    tip = p1 if closed else p0 + (p1 - p0) * 0.9 + nrm * 0.32
    return VGroup(Line(a, p0, color=color, stroke_width=3), Line(p0, tip, color=color, stroke_width=3), Line(p1, a + u * L, color=color, stroke_width=3),
                  Dot(p0, radius=0.06, color=color), Dot(p1, radius=0.06, color=color))


def source(center, r: float = 0.3, color: str = PALETTE.muted, label: str = "") -> VGroup:
    c = np.array(center, dtype=float)
    g = VGroup(Circle(radius=r, color=color, stroke_width=3).move_to(c),
               Text("+", font_size=20, color=color).move_to(c + (0, r * 0.45, 0)), Text("-", font_size=20, color=color).move_to(c - (0, r * 0.45, 0)))
    if label:
        g.add(Text(label, font_size=18, color=color).next_to(g[0], np.array([-1.0, 0, 0]), buff=0.12))
    return g


class Scope(VGroup):
    """A framed mini plot: ``set_data(x, y)`` maps data into the box. Used for waveforms in every power scene."""

    def __init__(self, label: str = "", width: float = 3.4, height: float = 1.0, color: str = PALETTE.green):
        super().__init__()
        self.w, self.h = width, height
        self.frame = Rectangle(width=width, height=height, stroke_color=PALETTE.muted, stroke_width=1.5)
        self.zero = Line(self.frame.get_left(), self.frame.get_right(), color=PALETTE.muted, stroke_width=1, stroke_opacity=0.6)
        self.trace = VMobject(color=color, stroke_width=2.5)
        self.caption = Text(label, font_size=16, color=PALETTE.muted).next_to(self.frame, np.array([0, 1.0, 0]), buff=0.05).align_to(self.frame, np.array([-1.0, 0, 0]))
        self._x = (0.0, 1.0)
        self._y = (-1.0, 1.0)
        self.add(self.frame, self.zero, self.trace, self.caption)

    def to_point(self, x: float, y: float) -> np.ndarray:
        x0, x1 = self._x
        y0, y1 = self._y
        cx, cy, _ = self.frame.get_center()
        return np.array([cx + ((x - x0) / (x1 - x0 or 1.0) - 0.5) * self.frame.width, cy + ((y - y0) / (y1 - y0 or 1.0) - 0.5) * self.frame.height * 0.84, 0.0])

    def set_data(self, x, y, ylim: tuple[float, float] | None = None) -> "Scope":
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        self._x = (float(x.min()), float(x.max()))
        lo, hi = ylim if ylim else (float(y.min()), float(y.max()))
        if hi - lo < 1e-9:
            lo, hi = lo - 1.0, hi + 1.0
        self._y = (lo, hi)
        self.trace.set_points_as_corners([self.to_point(a, b) for a, b in zip(x, y)])
        zero_y = self.to_point(0, min(max(0.0, lo), hi))[1]
        self.zero.put_start_and_end_on(np.array([self.frame.get_left()[0], zero_y, 0.0]), np.array([self.frame.get_right()[0], zero_y, 0.0]))
        return self
