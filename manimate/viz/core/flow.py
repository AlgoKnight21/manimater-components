"""Things that move along or around: pulses, drifting charge, particle clouds (README section 5.2)."""
from __future__ import annotations

import numpy as np
from manim import AnimationGroup, Circle, Dot, FadeOut, ShowPassingFlash, Succession, VGroup, VMobject, Wait

from .numeric import rng
from .palette import PALETTE

MAX_PARTICLES = 80  # performance budget, README section 14.2


def pulse_along(path: VMobject, color: str = PALETTE.yellow, run_time: float = 1.2, *, width: float = 0.12,
                trail: float = 0.25):
    """Intuition: a bright blip travelling a wire. The primitive behind signals, current and gradients."""
    if run_time <= 0:
        raise ValueError("run_time must be positive")
    head = path.copy().set_fill(opacity=0).set_stroke(color=color, width=max(2.0, width * 40), opacity=1.0)
    return ShowPassingFlash(head, time_width=min(max(trail, 0.05), 1.0), run_time=run_time)


class PulseTrain(AnimationGroup):
    """Intuition: several blips sent one after another, optionally running backward (blame in backprop)."""

    def __init__(self, paths: list[VMobject], delays: list[float] | None = None, color: str = PALETTE.yellow, *,
                 run_time: float = 1.2, reverse: bool = False, **kw):
        if not paths:
            raise ValueError("PulseTrain needs at least one path")
        delays = list(delays) if delays is not None else [0.0] * len(paths)
        if len(delays) != len(paths):
            raise ValueError("delays must have one entry per path")
        anims = []
        for p, d in zip(paths, delays):
            q = p.copy().reverse_points() if reverse else p
            pulse = pulse_along(q, color, run_time)
            anims.append(Succession(Wait(max(0.0, d)), pulse) if d > 0 else pulse)
        super().__init__(*anims, **kw)


class FlowField(VGroup):
    """Intuition: charge or data drifting along a wire. Particles loop forever; speed is a ValueTracker-friendly number."""

    def __init__(self, path: VMobject, *, n: int = 24, speed: float = 1.0, color: str = PALETTE.blue,
                 radius: float = 0.05):
        super().__init__()
        if not 1 <= n <= MAX_PARTICLES:
            raise ValueError(f"n must be between 1 and {MAX_PARTICLES}")
        self.path, self.speed, self.n = path, speed, n
        self._phase = 0.0
        for _ in range(n):
            self.add(Dot(radius=radius, color=color))
        self._place()
        self.add_updater(lambda m, dt: m._advance(dt))

    def _place(self) -> None:
        for i, d in enumerate(self.submobjects):
            d.move_to(self.path.point_from_proportion((i / self.n + self._phase) % 1.0))

    def _advance(self, dt: float) -> None:
        self._phase = (self._phase + 0.2 * self.speed * dt) % 1.0
        self._place()


class ParticleField(VGroup):
    """Intuition: a crowd of carriers or samples. Seeded, so two renders match; can lose or pair up members."""

    def __init__(self, n: int = 30, bounds: tuple[float, float] = (4.0, 2.0), *, kind: str = "dot",
                 drift: tuple[float, float] | None = None, seed: int = 7, color: str = PALETTE.teal):
        super().__init__()
        if not 1 <= n <= MAX_PARTICLES:
            raise ValueError(f"n must be between 1 and {MAX_PARTICLES}")
        if kind not in ("dot", "ring"):
            raise ValueError("kind must be 'dot' or 'ring'")
        self.bounds, self.drift = bounds, drift
        g = rng(seed)
        for _ in range(n):
            x, y = (g.random(2) - 0.5) * np.array(bounds)
            p = Dot(radius=0.05, color=color) if kind == "dot" else Circle(radius=0.07, color=color, stroke_width=3)
            self.add(p.move_to((x, y, 0.0)))
        if drift:
            self.add_updater(lambda m, dt: m._drift(dt))

    def _drift(self, dt: float) -> None:
        w, h = self.bounds
        for p in self.submobjects:
            x, y, _ = p.get_center()
            x = (x + self.drift[0] * dt + w / 2) % w - w / 2
            y = (y + self.drift[1] * dt + h / 2) % h - h / 2
            p.move_to((x, y, 0.0))

    def remove_particles(self, ids: list[int], run_time: float = 0.8):
        """Fade out the particles at ``ids`` (e.g. carriers swept out of a depletion region)."""
        return AnimationGroup(*(FadeOut(self.submobjects[i]) for i in ids), run_time=run_time)

    def recombine(self, pairs: list[tuple[int, int]], run_time: float = 1.0):
        """Slide each pair to its midpoint and fade it out (electron meets hole)."""
        anims = []
        for a, b in pairs:
            mid = (self.submobjects[a].get_center() + self.submobjects[b].get_center()) / 2
            anims += [self.submobjects[a].animate.move_to(mid), self.submobjects[b].animate.move_to(mid)]
        return Succession(AnimationGroup(*anims, run_time=run_time * 0.7),
                          AnimationGroup(*(FadeOut(self.submobjects[i]) for p in pairs for i in p), run_time=run_time * 0.3))
