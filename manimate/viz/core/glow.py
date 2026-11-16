"""Cheap glow without shaders (works in the Cairo renderer): stacked, fading strokes and discs."""
from __future__ import annotations

import numpy as np
from manim import Circle, Mobject, VGroup, VMobject

from .palette import PALETTE

MAX_LAYERS = 6


class Glow(VGroup):
    """Intuition: light bleeding past the edge of a bright object. Use it only to direct attention."""

    def __init__(self, mobject: VMobject, *, color: str = PALETTE.yellow, layers: int = 4, spread: float = 0.12):
        super().__init__()
        if not 1 <= layers <= MAX_LAYERS:
            raise ValueError(f"layers must be between 1 and {MAX_LAYERS}")
        base = max(mobject.get_stroke_width(), 2.0)
        for i in range(layers, 0, -1):  # widest, faintest layer first so brighter ones sit on top
            layer = mobject.copy().set_fill(opacity=0)
            layer.set_stroke(color=color, width=base + i * spread * 30, opacity=0.5 / (i + 1))
            self.add(layer)
        self.source = mobject


class GlowDot(VGroup):
    """Intuition: a point of light. Electrons, pulse heads and state-vector tips are all this."""

    def __init__(self, point=(0.0, 0.0, 0.0), radius: float = 0.1, *, color: str = PALETTE.yellow, layers: int = 4):
        super().__init__()
        if radius <= 0:
            raise ValueError("radius must be positive")
        for i in range(layers, 0, -1):
            disc = Circle(radius=radius * (1 + 0.8 * i), stroke_width=0, fill_color=color, fill_opacity=0.35 / (i + 1))
            self.add(disc)
        self.add(Circle(radius=radius, stroke_width=0, fill_color=color, fill_opacity=1.0))
        self.move_to(np.array(point, dtype=float))
