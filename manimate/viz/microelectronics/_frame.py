"""Local coordinates for Mobjects that rebuild their geometry every frame.

A group that redraws itself from a tracker must know where it currently is on screen, even after ``ctx.fit`` has scaled it
or the scene has moved it. ``Framed`` adds two invisible anchors (at local (0, 0) and (1, 0)); ``pt(x, y)`` maps local
coordinates through them, so rebuilt shapes always land in the right place at the right size.
"""
from __future__ import annotations

import numpy as np
from manim import ORIGIN, RIGHT, Dot


def _anchor(p) -> Dot:
    return Dot(p, radius=0.001, fill_opacity=0.0, stroke_opacity=0.0, stroke_width=0)


class Framed:
    """Intuition: a ruler printed on the group itself, so it can always tell where its own origin and scale are."""

    def _init_frame(self) -> None:
        self._a0, self._a1 = _anchor(ORIGIN), _anchor(RIGHT)
        self.add(self._a0, self._a1)

    @property
    def unit(self) -> float:
        """Screen length of one local unit (1.0 until the group is scaled)."""
        return float(np.linalg.norm(self._a1.get_center() - self._a0.get_center()))

    def pt(self, x: float, y: float = 0.0) -> np.ndarray:
        """Screen position of local point ``(x, y)``."""
        return self._a0.get_center() + self.unit * np.array([x, y, 0.0])

    def local_x(self, screen_x: float) -> float:
        return (screen_x - self._a0.get_center()[0]) / max(self.unit, 1e-9)
