"""Scene plumbing: per-reveal timing, a standalone runner for example scripts, and a dry-run harness for tests."""
from __future__ import annotations

from typing import Callable

import numpy as np
from manim import Animation, Scene, config

from ... import theme
from ...ctx import STAGE_CENTER, STAGE_H, STAGE_W, Ctx

DEFAULT_DURATION = 20.0  # README section 2.1: scenes stay under about 20 seconds


class Beats:
    """Intuition: a metronome. Hands out reveal durations so ``n`` reveals share what is left of the narration."""

    def __init__(self, ctx: Ctx, n: int, *, cap: float = 1.0, floor: float = 0.3):
        if n < 1:
            raise ValueError("n must be at least 1")
        self.ctx, self.left, self.cap, self.floor = ctx, n, cap, floor

    def next(self) -> float:
        d = Ctx.beat(self.ctx.remaining, self.left, self.cap, self.floor)
        self.left = max(0, self.left - 1)
        return d


class VizScene(Scene):
    """Run any ``play_*(ctx, ...)`` function as a plain Manim scene: subclass and override ``script``.

    (Named ``script`` rather than ``play`` because ``Scene.play`` already exists and ``Ctx`` calls it.)
    """

    duration: float = DEFAULT_DURATION

    def script(self, ctx: Ctx) -> None:
        raise NotImplementedError("override script(self, ctx) and call a play_* function")

    def construct(self) -> None:
        self.camera.background_color = theme.BG
        self.script(Ctx(self, self.duration))


def _animation_names(anims) -> list[str]:
    out: list[str] = []
    for a in anims:
        out.append(type(a).__name__)
        if isinstance(a, Animation) and hasattr(a, "animations"):
            out += _animation_names(a.animations)
    return out


class RecordingCtx(Ctx):
    """A Ctx that remembers what was played, so tests can check style rules without rendering."""

    def __init__(self, scene: Scene, duration: float = DEFAULT_DURATION):
        super().__init__(scene, duration)
        self.log: list[tuple[list[str], float]] = []

    def play(self, *anims, run_time: float = 1.0, **kw) -> None:
        self.log.append((_animation_names(anims), max(0.1, float(run_time))))
        super().play(*anims, run_time=run_time, **kw)

    @property
    def animation_names(self) -> set[str]:
        return {n for names, _ in self.log for n in names}

    def fade_in_time(self) -> float:
        return sum(t for names, t in self.log if names and names[0] == "FadeIn")

    def on_stage(self, tol: float = 0.05) -> bool:
        """True when every visible mobject lies inside the 12.4 x 5.8 stage (caption strip excluded)."""
        cx, cy, _ = STAGE_CENTER
        for m in self.scene.mobjects:
            if not len(m.get_all_points()):
                continue
            lo, hi = m.get_corner(np.array([-1, -1, 0])), m.get_corner(np.array([1, 1, 0]))
            if lo[0] < cx - STAGE_W / 2 - tol or hi[0] > cx + STAGE_W / 2 + tol \
                    or lo[1] < cy - STAGE_H / 2 - tol or hi[1] > cy + STAGE_H / 2 + tol:
                return False
        return True


def dry_run(script: Callable[[Ctx], None], duration: float = 10.0) -> RecordingCtx:
    """Run ``script(ctx)`` on a real Scene with rendering switched off; returns the RecordingCtx."""
    keys = ("dry_run", "progress_bar", "verbosity")
    saved = {k: config[k] for k in keys}
    config.dry_run, config.progress_bar, config.verbosity = True, "none", "ERROR"
    try:
        scene = Scene()
        scene.camera.background_color = theme.BG
        ctx = RecordingCtx(scene, duration)
        script(ctx)
        return ctx
    finally:
        for k, v in saved.items():
            config[k] = v
