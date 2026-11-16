"""Scenes for ``viz.signals`` control (README section 8): each ``play_*`` runs as a storyboard component or a standalone Scene.

Poles and zeros arrive from the planner as flat ``[re, im, re, im, ...]`` lists. Every scene fades the finished frame in
whole, then moves the parts that carry the meaning, and ends by filling the narration time that is left (``ctx.finish``).
"""
from __future__ import annotations

import numpy as np
from manim import DOWN, VGroup

from ...ctx import Ctx
from ..core.focus import fade_in_whole
from . import _math as sm
from .control import BlockDiagram, FeedbackLoop, PIDController, StateSpaceDiagram
from .plots import BodePlot, NyquistPlot, PoleZeroPlot, RootLocus
from .response import PIDTuningDemo, StepResponsePlot

BLOCK_PRESETS = ("feedback", "pid", "state_space", "cascade")
PLANT_CHOICES = ("first_order", "second_order", "integrator")
RESPONSE_KINDS = ("step", "impulse")


def _slots(ctx: Ctx, *weights: float, floor: float = 0.5):
    """Intuition: cutting the time that is left into shares, so the picture ends when the narration does."""
    left = ctx.remaining
    total = float(sum(weights)) or 1.0
    return [max(floor, left * w / total) for w in weights]


def complex_roots(flat) -> list[complex]:
    """Intuition: ``[-1, 2, -1, -2]`` means the two numbers ``-1+2j`` and ``-1-2j``: pairs of (real, imaginary)."""
    vals = [float(v) for v in (flat or [])]
    if len(vals) % 2:
        raise ValueError("give poles and zeros as pairs: real, imaginary, real, imaginary, ...")
    return [complex(vals[i], vals[i + 1]) for i in range(0, len(vals), 2)]


def _real_poly(roots: list[complex]) -> np.ndarray:
    return np.real(np.poly(roots)) if roots else np.array([1.0])


def _need(poles: list[complex]) -> list[complex]:
    if not poles:
        raise ValueError("give at least one pole")
    return poles


def play_block_diagram(ctx: Ctx, preset: str = "feedback", show_scopes: bool = True) -> None:
    """Intuition: a flow chart for signals. Each box transforms what flows through it, and feedback compares the result with the goal."""
    if preset not in BLOCK_PRESETS:
        raise ValueError(f"preset must be one of {list(BLOCK_PRESETS)}")
    if preset == "feedback":
        loop = FeedbackLoop(["Controller", "Plant"], ["Sensor"])
        mob = loop
        run = lambda t: loop.animate_signal(pulses=2, run_time=t)  # noqa: E731
    elif preset == "pid":
        pid = PIDController(2.0, 1.0, 0.5)
        mob = pid
        run = lambda t: pid.animate_response(run_time=t)  # noqa: E731
    elif preset == "state_space":
        mob = StateSpaceDiagram([[0, 1], [-2, -3]], [[0], [1]], [[1, 0]])
        run = None
    else:
        mob = BlockDiagram({"r": {"label": "r", "kind": "io"}, "s": {"kind": "sum"}, "c": "Controller", "p": "Plant",
                            "y": {"label": "y", "kind": "io"}}, [("r", "s"), ("s", "c"), ("c", "p"), ("p", "y"), ("p", "s")])
        run = (lambda t: mob.animate_time(run_time=t)) if show_scopes else None
    ctx.fit(mob)
    fade_in_whole(ctx, mob, 1.0)
    if run is not None:
        (t,) = _slots(ctx, 1)
        ctx.play(run(max(1.5, t * 0.92)), run_time=max(1.5, t * 0.92))
    ctx.finish()


def play_pid_response(ctx: Ctx, plant: str = "second_order", kp: float = 1.0, ki: float = 0.0, kd: float = 0.0) -> None:
    """Intuition: P pushes in proportion to the error, I remembers the past error, D brakes on how fast it is changing."""
    if plant not in PLANT_CHOICES:
        raise ValueError(f"plant must be one of {list(PLANT_CHOICES)}")
    demo = PIDTuningDemo(plant, kp=max(0.0, kp), ki=0.0, kd=0.0)
    ctx.fit(demo)
    fade_in_whole(ctx, demo, 1.0)
    steps = [("kp", kp), ("ki", ki), ("kd", kd)]
    steps = [(k, v) for k, v in steps if v > 0 and k != "kp"] or [("ki", 0.0)]
    first, *rest = _slots(ctx, 1, *([1.0] * len(steps)))
    ctx.play(demo.animate_gains(kp=max(kp, 0.1) * 3, run_time=first), run_time=first)
    for (k, v), t in zip(steps, rest):
        ctx.play(demo.animate_gains(**{k: v}, run_time=t), run_time=t)
    ctx.finish()


def play_time_response(ctx: Ctx, poles=None, zeros=None, gain: float = 1.0, kind: str = "step") -> None:
    """Intuition: poke the system once and watch how it settles: rise time, overshoot and settling are all set by where the poles sit."""
    if kind not in RESPONSE_KINDS:
        raise ValueError(f"kind must be one of {list(RESPONSE_KINDS)}")
    p, z = _need(complex_roots(poles if poles else [-1, 2, -1, -2])), complex_roots(zeros)
    tf = sm.tf_from_pz(p, z, gain)
    plot = StepResponsePlot(tf, kind=kind, width=6.0, height=3.0)
    pz = PoleZeroPlot(p, z, size=3.0)
    row = VGroup(pz, plot).arrange(buff=0.6)
    ctx.fit(row)
    fade_in_whole(ctx, row, 1.0)
    a, b = _slots(ctx, 3, 1)
    ctx.play(plot.animate_draw(run_time=a), run_time=a)
    ctx.play(plot.animate_annotate(run_time=b), run_time=b)
    ctx.finish()


def play_bode_plot(ctx: Ctx, poles=None, zeros=None, gain: float = 1.0, asymptotes: bool = True) -> None:
    """Intuition: play a pure tone at every frequency and note how loud and how late it comes back."""
    p, z = _need(complex_roots(poles if poles else [0, 0, -1, 0, -2, 0])), complex_roots(zeros)
    bode = BodePlot(p, z, gain, asymptotes=bool(asymptotes))
    ctx.fit(bode)
    fade_in_whole(ctx, bode, 1.0)
    a, b = _slots(ctx, 3, 1)
    ctx.play(bode.animate_sweep(run_time=a), run_time=a)
    ctx.play(bode.animate_margins(run_time=b), run_time=b)
    ctx.finish()


def play_nyquist_plot(ctx: Ctx, poles=None, zeros=None, gain: float = 1.0) -> None:
    """Intuition: trace the loop's response around the critical point minus one; circling it is how instability shows itself."""
    p, z = _need(complex_roots(poles if poles else [0, 0, -1, 0, -2, 0])), complex_roots(zeros)
    nyq = NyquistPlot(p, z, gain)
    ctx.fit(nyq)
    fade_in_whole(ctx, nyq, 1.0)
    a, b = _slots(ctx, 3, 1)
    ctx.play(nyq.animate_trace(run_time=a), run_time=a)
    ctx.play(nyq.animate_mirror(run_time=b), run_time=b)
    ctx.finish()


def play_root_locus(ctx: Ctx, poles=None, zeros=None, k_max: float = 10.0) -> None:
    """Intuition: turn the gain knob and watch every closed-loop pole walk; stability ends when one crosses the yellow line."""
    p, z = _need(complex_roots(poles if poles else [0, 0, -1, 0, -2, 0])), complex_roots(zeros)
    num, den = _real_poly(z), _real_poly(p)
    locus = RootLocus(num, den, max(0.5, float(k_max)))
    ctx.fit(locus)
    fade_in_whole(ctx, locus, 1.0)
    (t,) = _slots(ctx, 1)
    ctx.play(locus.animate_k(run_time=max(1.5, t * 0.92)), run_time=max(1.5, t * 0.92))
    ctx.finish()


def play_pole_zero_map(ctx: Ctx, poles=None, zeros=None, domain: str = "s") -> None:
    """Intuition: poles are where the system rings; left of the line they fade away, right of it they blow up."""
    dom = domain if domain in ("s", "z") else "s"
    default = [-0.5, 2, -0.5, -2, -3, 0] if dom == "s" else [0.5, 0.5, 0.5, -0.5]
    p, z = _need(complex_roots(poles if poles else default)), complex_roots(zeros)
    pz = PoleZeroPlot(p, z, dom, size=4.2)
    ctx.fit(pz)
    fade_in_whole(ctx, pz, 1.0)
    first = p[0]
    target = complex(first.real * 0.4, first.imag) if dom == "s" else complex(first.real * 1.5, first.imag * 1.5)
    (t,) = _slots(ctx, 1)
    ctx.play(pz.animate_move_pole(0, target, run_time=max(1.5, t * 0.92)), run_time=max(1.5, t * 0.92))
    ctx.finish()
