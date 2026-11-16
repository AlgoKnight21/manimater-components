"""Control-system adapters (viz README sections 8 and 12.1): thin wrappers over ``viz.signals`` scenes.

Poles and zeros are flat ``[re, im, re, im, ...]`` lists: ``[-1, 2, -1, -2]`` is the pair ``-1+2j`` and ``-1-2j``.
"""
from __future__ import annotations

from ..ctx import Ctx
from ..registry import Param as P, component
from ..viz.signals import scenes as S

_D = "control-systems"
_ROOTS = "as pairs real, imaginary, real, imaginary (up to 4 values); [-1, 0] is the single pole at -1"


def _in_range(lo: float, hi: float, what: str = "value"):
    return lambda v: None if lo <= v <= hi else f"{what} must be between {lo:g} and {hi:g}"


def _roots(v: list) -> str | None:
    if len(v) % 2:
        return "give pairs: real, imaginary, real, imaginary"
    return None if all(abs(x) <= 50 for x in v) else "values must be between -50 and 50"


@component(
    "block_diagram",
    "Control block diagrams: a feedback loop with signal pulses, a PID controller's three parallel paths, a state-space "
    "diagram, or a cascade with a summing junction and live scopes.",
    {"preset": P("string", "which diagram", default="feedback", choices=list(S.BLOCK_PRESETS)),
     "show_scopes": P("boolean", "animate signals through the cascade diagram", default=True)},
    {"preset": "feedback", "show_scopes": True},
    domain=_D,
)
def block_diagram(ctx: Ctx, preset: str = "feedback", show_scopes: bool = True) -> None:
    S.play_block_diagram(ctx, preset, show_scopes)


@component(
    "pid_response",
    "PID tuning on a plant's step response: proportional gain alone leaves an offset and ringing, integral action closes "
    "the gap, derivative action damps the overshoot.",
    {"plant": P("string", "the system being controlled", default="second_order", choices=list(S.PLANT_CHOICES)),
     "kp": P("number", "proportional gain, 0 to 20", default=1, validator=_in_range(0, 20, "kp")),
     "ki": P("number", "integral gain, 0 to 20", default=1, validator=_in_range(0, 20, "ki")),
     "kd": P("number", "derivative gain, 0 to 20", default=1, validator=_in_range(0, 20, "kd"))},
    {"plant": "second_order", "kp": 4, "ki": 2, "kd": 1.5},
    domain=_D,
)
def pid_response(ctx: Ctx, plant: str = "second_order", kp: float = 1, ki: float = 1, kd: float = 1) -> None:
    S.play_pid_response(ctx, plant, kp, ki, kd)


@component(
    "time_response",
    "Step or impulse response of a system from its poles and zeros, with rise time, overshoot and settling time marked "
    "and the pole-zero map beside it.",
    {"poles": P("number[]", f"poles {_ROOTS}", default=[-1, 2, -1, -2], validator=_roots),
     "zeros": P("number[]", f"zeros {_ROOTS}", default=[], validator=_roots),
     "gain": P("number", "overall gain", default=1, validator=_in_range(0.01, 1000, "gain")),
     "kind": P("string", "input signal", default="step", choices=list(S.RESPONSE_KINDS))},
    {"poles": [-1, 2, -1, -2], "zeros": [], "gain": 5, "kind": "step"},
    domain=_D,
)
def time_response(ctx: Ctx, poles: list | None = None, zeros: list | None = None, gain: float = 1, kind: str = "step") -> None:
    S.play_time_response(ctx, poles, zeros, gain, kind)


@component(
    "bode_plot",
    "Bode plot (magnitude and phase against frequency) with straight-line asymptotes, a frequency sweep, and the gain and phase margins.",
    {"poles": P("number[]", f"poles {_ROOTS}", default=[0, 0, -1, 0, -2, 0], validator=_roots),
     "zeros": P("number[]", f"zeros {_ROOTS}", default=[], validator=_roots),
     "gain": P("number", "loop gain", default=1, validator=_in_range(0.01, 1000, "gain")),
     "asymptotes": P("boolean", "draw the straight-line approximation", default=True)},
    {"poles": [0, 0, -1, 0, -2, 0], "zeros": [], "gain": 10, "asymptotes": True},
    domain=_D,
)
def bode_plot(ctx: Ctx, poles: list | None = None, zeros: list | None = None, gain: float = 1, asymptotes: bool = True) -> None:
    S.play_bode_plot(ctx, poles, zeros, gain, asymptotes)


@component(
    "nyquist_plot",
    "Nyquist plot of a loop transfer function traced around the critical point minus one, with its mirror image and the encirclement count that decides stability.",
    {"poles": P("number[]", f"poles {_ROOTS}", default=[0, 0, -1, 0, -2, 0], validator=_roots),
     "zeros": P("number[]", f"zeros {_ROOTS}", default=[], validator=_roots),
     "gain": P("number", "loop gain", default=1, validator=_in_range(0.01, 1000, "gain"))},
    {"poles": [0, 0, -1, 0, -2, 0], "zeros": [], "gain": 10},
    domain=_D,
)
def nyquist_plot(ctx: Ctx, poles: list | None = None, zeros: list | None = None, gain: float = 1) -> None:
    S.play_nyquist_plot(ctx, poles, zeros, gain)


@component(
    "root_locus",
    "Root locus: the closed-loop poles walking across the s-plane as the gain increases from zero, with the critical gain where they cross into instability.",
    {"poles": P("number[]", f"open-loop poles {_ROOTS}", default=[0, 0, -1, 0, -2, 0], validator=_roots),
     "zeros": P("number[]", f"open-loop zeros {_ROOTS}", default=[], validator=_roots),
     "k_max": P("number", "largest gain to sweep, 0.5 to 200", default=10, validator=_in_range(0.5, 200, "k_max"))},
    {"poles": [0, 0, -1, 0, -2, 0], "zeros": [], "k_max": 10},
    domain=_D,
)
def root_locus(ctx: Ctx, poles: list | None = None, zeros: list | None = None, k_max: float = 10) -> None:
    S.play_root_locus(ctx, poles, zeros, k_max)


@component(
    "pole_zero_map",
    "Pole-zero map in the s-plane (stable on the left) or the z-plane (stable inside the unit circle), with a pole sliding to show how stability changes.",
    {"poles": P("number[]", f"poles {_ROOTS}", default=[-0.5, 2, -0.5, -2, -3, 0], validator=_roots),
     "zeros": P("number[]", f"zeros {_ROOTS}", default=[], validator=_roots),
     "domain": P("string", "s for continuous time, z for discrete time", default="s", choices=["s", "z"])},
    {"poles": [-0.5, 2, -0.5, -2, -3, 0], "zeros": [-1, 0], "domain": "s"},
    domain=_D,
)
def pole_zero_map(ctx: Ctx, poles: list | None = None, zeros: list | None = None, domain: str = "s") -> None:
    S.play_pole_zero_map(ctx, poles, zeros, domain)
