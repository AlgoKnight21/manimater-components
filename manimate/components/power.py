"""Power pack: transformer, power triangle, motor torque-speed curves."""
from __future__ import annotations

import math

import numpy as np
from manim import (
    DOWN, LEFT, RIGHT, UP, Arc, Arrow, Axes, Create, Dot, FadeIn, Line, Polygon, Rectangle, Text, ValueTracker, VGroup,
    Write, always_redraw,
)

from .. import theme
from ..ctx import Ctx, label
from ..registry import Param as P, component

CENTER = np.array([0.0, 0.5, 0.0])


def _range(lo: float, hi: float):
    return lambda v: None if lo <= v <= hi else f"must be between {lo:g} and {hi:g}"


def _fmt(v: float) -> str:
    return f"{v:.3g}" if abs(v) < 1000 else f"{v:,.0f}"


# --------------------------------------------------------------------------------------------------
def _winding(x: float, turns: int, color: str) -> VGroup:
    """Horizontal turns wrapped around a vertical core limb centred at ``x``."""
    top, bottom = CENTER[1] + 1.05, CENTER[1] - 1.05
    ys = np.linspace(top, bottom, turns)
    return VGroup(*[Line((x - 0.75, y, 0), (x + 0.75, y, 0), color=color, stroke_width=5) for y in ys])


@component(
    "transformer",
    "Two windings on an iron core: flux links them and the voltage scales by the turns ratio Vs/Vp = Ns/Np. Shows step-up or step-down.",
    {"turns_ratio": P("number", "Ns/Np, between 0.1 and 10 (above 1 steps up, below 1 steps down)", default=2,
                      validator=_range(0.1, 10)),
     "primary_volts": P("number", "Primary voltage in volts, 1 to 1000000", default=230, validator=_range(1, 1_000_000)),
     "title": P("string", "Heading", default="")},
    {"turns_ratio": 2, "primary_volts": 230, "title": "Step-up transformer"},
    domain="electrical",
)
def transformer(ctx: Ctx, turns_ratio: float = 2, primary_volts: float = 230, title: str = "") -> None:
    secondary_volts = primary_volts * turns_ratio
    np_draw = 6
    ns_draw = max(2, min(12, round(np_draw * turns_ratio)))
    core = VGroup(
        Rectangle(width=5.0, height=3.3, fill_color=theme.MUTED, fill_opacity=0.55, stroke_color=theme.FG, stroke_width=3),
        Rectangle(width=2.6, height=1.5, fill_color=theme.BG, fill_opacity=1, stroke_color=theme.FG, stroke_width=3),
    ).move_to(CENTER)
    prim = _winding(-1.9, np_draw, theme.ACCENT)
    sec = _winding(1.9, ns_draw, theme.WARN)
    kind = "step-up" if turns_ratio > 1 else "step-down" if turns_ratio < 1 else "1:1 isolation"
    left = VGroup(label("Primary", 28, theme.ACCENT, weight="BOLD"), label(f"Vp = {_fmt(primary_volts)} V", 26),
                  label("Np turns", 22, theme.MUTED)).arrange(DOWN, buff=0.15).move_to((-4.9, CENTER[1], 0))
    right = VGroup(label("Secondary", 28, theme.WARN, weight="BOLD"), label(f"Vs = {_fmt(secondary_volts)} V", 26),
                   label("Ns turns", 22, theme.MUTED)).arrange(DOWN, buff=0.15).move_to((4.9, CENTER[1], 0))
    formula = label(f"Vs / Vp = Ns / Np = {turns_ratio:g}   ({kind})", 28, theme.YELLOW).move_to((0, -2.35, 0))

    path = Rectangle(width=3.8, height=2.4).move_to(CENTER)
    t = ValueTracker(0)
    flux = always_redraw(lambda: Dot(path.point_from_proportion(t.get_value() % 1.0), radius=0.11, color=theme.YELLOW))
    flux_tag = label("magnetic flux", 20, theme.YELLOW).move_to(CENTER)

    if title:
        ctx.play(Write(ctx.heading(title)), run_time=0.7)
    ctx.play(Create(core), run_time=0.9)
    ctx.play(Create(prim), FadeIn(left), run_time=1.0)
    ctx.play(Create(sec), FadeIn(right), run_time=1.0)
    ctx.play(FadeIn(formula), run_time=0.6)
    ctx.add(flux)
    ctx.play(FadeIn(flux_tag), run_time=0.4)
    loops = max(1, int(ctx.remaining / 2.5))
    ctx.play(t.animate.set_value(float(loops)), run_time=max(1.0, ctx.remaining), rate_func=lambda x: x)


# --------------------------------------------------------------------------------------------------
@component(
    "power_triangle",
    "Right triangle of real power P, reactive power Q and apparent power S with the power-factor angle. Use for power factor and reactive power.",
    {"real_power_kw": P("number", "Real power P in kW, above 0", default=4, validator=_range(0.001, 1_000_000)),
     "power_factor": P("number", "Power factor cos(phi), between 0.1 and 1", default=0.8, validator=_range(0.1, 1)),
     "lagging": P("boolean", "True for an inductive (lagging) load, false for a capacitive (leading) load", default=True),
     "title": P("string", "Heading", default="")},
    {"real_power_kw": 4, "power_factor": 0.8, "lagging": True, "title": "Power triangle"},
    domain="electrical",
)
def power_triangle(ctx: Ctx, real_power_kw: float = 4, power_factor: float = 0.8, lagging: bool = True, title: str = "") -> None:
    p = real_power_kw
    s = p / power_factor
    q = math.sqrt(max(s * s - p * p, 0.0))
    sign = 1.0 if lagging else -1.0
    phi = math.acos(power_factor)
    # Scale so the triangle fits a 5.2 x 3.2 box.
    k = min(5.2 / p, 3.2 / q) if q > 1e-9 else 5.2 / p
    origin = np.array([-3.6, 0.5 - sign * q * k / 2, 0.0])  # centre the triangle vertically on the stage
    a = origin
    b = origin + np.array([p * k, 0.0, 0.0])
    c = origin + np.array([p * k, sign * q * k, 0.0])
    arrow_p = Arrow(a, b, buff=0, color=theme.GOOD, stroke_width=7, max_tip_length_to_length_ratio=0.12)
    arrow_q = Arrow(b, c, buff=0, color=theme.WARN, stroke_width=7, max_tip_length_to_length_ratio=0.2) if q > 1e-9 else VGroup()
    arrow_s = Arrow(a, c, buff=0, color=theme.ACCENT, stroke_width=7, max_tip_length_to_length_ratio=0.12)
    fill = Polygon(a, b, c, color=theme.PANEL, fill_opacity=0.6, stroke_width=0)
    angle = Arc(radius=0.9, start_angle=0, angle=sign * phi, arc_center=a, color=theme.YELLOW, stroke_width=4)
    phi_tag = Text("φ", font_size=30, color=theme.YELLOW).move_to(a + np.array([1.25, sign * 0.3 * min(1.0, phi * 2), 0]))
    tag_p = label("P", 30, theme.GOOD, weight="BOLD").next_to(Line(a, b), DOWN if sign > 0 else UP, buff=0.2)
    tag_q = label("Q", 30, theme.WARN, weight="BOLD").next_to(Line(b, c), RIGHT, buff=0.2)
    tag_s = label("S", 30, theme.ACCENT, weight="BOLD").move_to((a + c) / 2 + np.array([-0.35, sign * 0.35, 0]))
    kind = "lagging" if lagging else "leading"
    panel = VGroup(
        label(f"P = {_fmt(p)} kW", 28, theme.GOOD),
        label(f"Q = {_fmt(q)} kvar", 28, theme.WARN),
        label(f"S = {_fmt(s)} kVA", 28, theme.ACCENT),
        label(f"pf = cos φ = {power_factor:g} {kind}", 26, theme.YELLOW),
        label("S² = P² + Q²", 24, theme.MUTED),
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to((3.9, 0.4, 0))

    if title:
        ctx.play(Write(ctx.heading(title)), run_time=0.7)
    ctx.play(FadeIn(fill), Create(arrow_p), FadeIn(tag_p), run_time=1.0)
    ctx.play(Create(arrow_q), FadeIn(tag_q), run_time=1.0)
    ctx.play(Create(arrow_s), FadeIn(tag_s), run_time=1.0)
    ctx.play(Create(angle), FadeIn(phi_tag), run_time=0.8)
    ctx.play(FadeIn(panel), run_time=0.8)
    ctx.finish()


# --------------------------------------------------------------------------------------------------
def _motor_path(kind: str):
    """Return u in [0, 1] -> (speed, torque), speed as a fraction of no-load speed, torque on a 0..3 scale."""
    if kind == "dc_shunt":      # nearly constant speed, drooping slightly with load
        return lambda u: (1.0 - 0.08 * u, 2.4 * u)
    if kind == "dc_series":     # speed falls steeply as torque rises
        return lambda u: (min(1.0, (0.5 / (0.2 + 2.8 * u)) ** 0.7), 0.2 + 2.8 * u)
    if kind == "synchronous":   # one speed, any torque
        return lambda u: (1.0, 3.0 * u)

    def induction(u: float) -> tuple[float, float]:  # classic breakdown-torque curve against speed (slip s = 1 - n)
        n = 0.99 * u
        s, sm = max(1.0 - n, 1e-3), 0.2
        return n, 2.5 * 2 / (s / sm + sm / s)
    return induction


_TITLES = {"induction": "Induction motor", "dc_shunt": "DC shunt motor", "dc_series": "DC series motor",
           "synchronous": "Synchronous motor"}
_NOTES = {
    "induction": "Starts with torque, peaks at breakdown, runs just below synchronous speed",
    "dc_shunt": "Speed stays nearly constant from no load to full load",
    "dc_series": "Very high starting torque, speed rises sharply as load falls",
    "synchronous": "Runs at exactly synchronous speed at any load, no starting torque of its own",
}


@component(
    "torque_speed_curve",
    "Torque versus speed characteristic of a motor with a marker sliding along it. Use for comparing how motors behave under load.",
    {"motor": P("string", "Motor type", default="induction", choices=["induction", "dc_shunt", "dc_series", "synchronous"]),
     "title": P("string", "Heading", default="")},
    {"motor": "induction", "title": "Torque-speed curve"},
    domain="electrical",
)
def torque_speed_curve(ctx: Ctx, motor: str = "induction", title: str = "") -> None:
    axes = Axes(x_range=[0, 1.05, 0.25], y_range=[0, 3.1, 1], x_length=7.4, y_length=3.9,
                axis_config={"color": theme.MUTED, "include_numbers": False, "include_tip": True}).move_to((-2.0, 0.25, 0))
    xl = Text("speed", font_size=22, color=theme.MUTED).next_to(axes.x_axis, DOWN, buff=0.25).align_to(axes.x_axis, RIGHT)
    yl = Text("torque", font_size=22, color=theme.MUTED).next_to(axes.y_axis, UP, buff=0.15).align_to(axes.y_axis, LEFT)
    path = _motor_path(motor)
    curve = axes.plot_parametric_curve(lambda u: np.array([*path(u), 0.0]), t_range=[0, 1, 0.01], color=theme.ACCENT, stroke_width=6)
    u = ValueTracker(0.0)
    dot = always_redraw(lambda: Dot(axes.c2p(*path(u.get_value())), radius=0.13, color=theme.YELLOW))
    panel = VGroup(label(_TITLES[motor], 30, theme.ACCENT, weight="BOLD"), label(_NOTES[motor], 24, theme.FG, width=22)
                   ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to((4.3, 0.6, 0))

    if title:
        ctx.play(Write(ctx.heading(title)), run_time=0.7)
    ctx.play(Create(axes), FadeIn(xl), FadeIn(yl), run_time=0.9)
    ctx.play(Create(curve), run_time=min(2.5, max(1.2, ctx.remaining / 3)))
    ctx.play(FadeIn(panel), run_time=0.6)
    ctx.add(dot)
    ctx.play(u.animate.set_value(1.0), run_time=max(1.0, ctx.remaining), rate_func=lambda x: x)
