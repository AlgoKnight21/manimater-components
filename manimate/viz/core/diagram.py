"""Block-diagram parts: labelled blocks with ports and orthogonal wires (README section 5.3)."""
from __future__ import annotations

import numpy as np
from manim import Arrow, RoundedRectangle, Text, VGroup, VMobject

from .palette import PALETTE

PORTS = ("left", "right", "top", "bottom")


def orthogonal_route(a, b, via: list | None = None) -> list[np.ndarray]:
    """Right-angle path from ``a`` to ``b`` (optionally through ``via`` points): horizontal first, then vertical."""
    pts = [np.array(p, dtype=float)[:3] if len(p) == 3 else np.array([*p, 0.0]) for p in [a, *(via or []), b]]
    out = [pts[0]]
    for p in pts[1:]:
        q = out[-1]
        if abs(p[0] - q[0]) > 1e-9 and abs(p[1] - q[1]) > 1e-9:
            out.append(np.array([p[0], q[1], 0.0]))
        out.append(p)
    return out


class Block(VGroup):
    """Intuition: a box that does one job. Signals enter at named ports and leave at others."""

    def __init__(self, label: str, *, w: float = 1.6, h: float = 0.9, color: str = PALETTE.blue, size: int = 22):
        super().__init__()
        self.box = RoundedRectangle(width=w, height=h, corner_radius=0.1, color=color, stroke_width=3)
        self.text = Text(label, font_size=size, color=PALETTE.fg)
        if self.text.width > w * 0.9:
            self.text.scale_to_fit_width(w * 0.9)
        self.add(self.box, self.text.move_to(self.box))

    def port(self, name: str) -> np.ndarray:
        if name not in PORTS:
            raise ValueError(f"unknown port '{name}'; choose one of {PORTS}")
        d = {"left": (-1, 0, 0), "right": (1, 0, 0), "top": (0, 1, 0), "bottom": (0, -1, 0)}[name]
        return self.box.get_edge_center(np.array(d, dtype=float))


class Wire(VGroup):
    """Intuition: a connection that only turns at right angles, like a schematic or a bus."""

    def __init__(self, points, *, arrow: bool = False, color: str = PALETTE.muted, width: float = 3.0):
        super().__init__()
        pts = [np.array(p, dtype=float) for p in points]
        if len(pts) < 2:
            raise ValueError("a wire needs at least two points")
        self.points_list = pts
        body = pts[:-1] if arrow else pts
        if len(body) >= 2:
            line = VMobject(color=color, stroke_width=width).set_points_as_corners(body)
            self.add(line)
        if arrow:
            self.add(Arrow(pts[-2], pts[-1], buff=0, color=color, stroke_width=width, max_tip_length_to_length_ratio=0.35))

    @classmethod
    def route(cls, a, b, *, via: list | None = None, **kw) -> "Wire":
        return cls(orthogonal_route(a, b, via), **kw)
