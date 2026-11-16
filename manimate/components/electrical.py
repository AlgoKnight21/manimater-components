"""Electrical engineering pack: AC waveforms, phasors, circuits, rotating field, synchronous motor."""
from __future__ import annotations

import math

import numpy as np
from manim import (
    DOWN, LEFT, PI, RIGHT, TAU, UP, Annulus, AnnularSector, Arc, Arrow, Axes, Circle, Create, DashedLine, Dot, FadeIn,
    LaggedStart, Line, ParametricFunction, Rectangle, Text, ValueTracker, VGroup, VMobject, Write, always_redraw,
)

from .. import theme
from ..ctx import Ctx, label
from ..registry import Param as P, component

CENTER = np.array([0.0, 0.5, 0.0])
SIDE_CENTER = np.array([-2.6, 0.5, 0.0])
PANEL_X = 3.9


def _polar(r: float, a: float) -> np.ndarray:
    return np.array([r * math.cos(a), r * math.sin(a), 0.0])


def _smooth(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def _pole_group(poles: int, center: np.ndarray, r_in: float, r_out: float, opacity: float, letters: bool) -> VGroup:
    """Alternating N (red) / S (blue) pole faces, built at angle 0. Rotate the whole group afterwards."""
    arc = TAU / poles * 0.78
    g = VGroup()
    for k in range(poles):
        a = k * TAU / poles
        north = k % 2 == 0
        g.add(AnnularSector(inner_radius=r_in, outer_radius=r_out, angle=arc, start_angle=a - arc / 2,
                            color=theme.DANGER if north else theme.ACCENT, fill_opacity=opacity, stroke_width=0).shift(center))
        if letters:
            g.add(Text("N" if north else "S", font_size=26, color=theme.FG, weight="BOLD")
                  .move_to(center + _polar((r_in + r_out) / 2, a)))
    return g


def _spin(mob: VGroup, center: np.ndarray, angle_fn) -> None:
    """Keep ``mob`` rotated to ``angle_fn()`` radians about ``center`` (cheap: no re-creation per frame)."""
    mob._cur = 0.0

    def upd(m):
        a = angle_fn()
        m.rotate(a - m._cur, about_point=center)
        m._cur = a

    mob.add_updater(upd)


def _live_text(key_fn, build_fn) -> VGroup:
    """Text that is only rebuilt when its key changes (Text creation is slow, so not every frame)."""
    g = VGroup(build_fn(key_fn()))
    g._key = key_fn()

    def upd(m):
        k = key_fn()
        if k != m._key:
            m._key = k
            m.become(VGroup(build_fn(k)))

    g.add_updater(upd)
    return g


def _stator(center: np.ndarray, show_coils: bool = True) -> VGroup:
    ring = Annulus(inner_radius=2.05, outer_radius=2.45, color=theme.MUTED, fill_opacity=0.3, stroke_width=2).shift(center)
    g = VGroup(ring)
    if show_coils:
        # Opposite pairs are one phase: A/A' at 90/270, B/B' at 210/30, C/C' at 330/150 degrees.
        coils = [("A", 90, 0), ("C'", 150, 2), ("B", 210, 1), ("A'", 270, 0), ("C", 330, 2), ("B'", 30, 1)]
        for name, deg, ph in coils:
            p = center + _polar(2.25, math.radians(deg))
            g.add(Dot(p, radius=0.17, color=theme.PHASE_COLORS[ph]))
            g.add(Text(name, font_size=16, color=theme.BG, weight="BOLD").move_to(p))
    return g


def _rev_seconds(poles: int, f: float) -> float:
    return min(12.0, max(1.5, 3.0 * (poles / 2) * (50.0 / max(f, 1.0))))


def _speed_panel(poles: int, f: float, y: float = 1.9) -> VGroup:
    ns = 120.0 * f / poles
    g = VGroup(
        label(f"Poles: {poles}", 26, theme.FG),
        label(f"Supply: {f:g} Hz", 26, theme.FG),
        label(f"Ns = 120 × {f:g} / {poles}", 28, theme.YELLOW),
        label(f"= {ns:g} rpm", 34, theme.YELLOW, weight="BOLD"),
        label("(animation slowed down)", 18, theme.MUTED),
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
    g.move_to((PANEL_X, y, 0))
    return g


# --------------------------------------------------------------------------------------------------
@component(
    "ac_waveform", "Animated sine wave(s) with a moving marker. Use 3 phases to show a balanced three-phase supply.",
    {"phases": P("integer", "1 for single-phase, 3 for three-phase", default=1, choices=[1, 3]),
     "frequency_hz": P("number", "Supply frequency, used for the period note", default=50),
     "title": P("string", "Heading", default="")},
    {"phases": 3, "frequency_hz": 50, "title": "Three-phase supply"},
    domain="electrical",
)
def ac_waveform(ctx: Ctx, phases: int = 1, frequency_hz: float = 50.0, title: str = "") -> None:
    axes = Axes(x_range=[0, 4 * PI, PI], y_range=[-1.3, 1.3, 1], x_length=10.5, y_length=4.0,
                axis_config={"color": theme.MUTED, "include_numbers": False, "include_tip": True}).move_to((0, 0.2, 0))
    names = ["A", "B", "C"] if phases == 3 else ["v(t)"]
    colors = theme.PHASE_COLORS if phases == 3 else [theme.ACCENT]
    shifts = [k * TAU / 3 for k in range(phases)]
    curves = [axes.plot(lambda x, s=s: math.sin(x - s), color=colors[k], x_range=[0, 4 * PI, 0.05], stroke_width=5, use_smoothing=False)
              for k, s in enumerate(shifts)]
    tags = [Text(names[k], font_size=24, color=colors[k]).next_to(axes.c2p(4 * PI, math.sin(4 * PI - s)), RIGHT, buff=0.15)
            for k, s in enumerate(shifts)]
    tl = axes.get_x_axis_label(Text("time", font_size=22, color=theme.MUTED), edge=RIGHT, direction=DOWN, buff=0.3)
    period = 1000.0 / max(frequency_hz, 0.1)
    note = Text(f"{frequency_hz:g} Hz  →  one cycle every {period:g} ms" + ("   |   phases 120° apart" if phases == 3 else ""),
                font_size=24, color=theme.YELLOW).move_to((0, -2.45, 0))
    if title:
        ctx.play(Write(ctx.heading(title)), run_time=0.7)
    ctx.play(Create(axes), FadeIn(tl), run_time=0.8)
    ctx.play(*[Create(c) for c in curves], run_time=min(3.0, max(1.5, ctx.remaining / 3)))
    ctx.play(*[FadeIn(t) for t in tags], FadeIn(note), run_time=0.5)
    t = ValueTracker(0)
    scan = always_redraw(lambda: Line(axes.c2p(t.get_value(), -1.3), axes.c2p(t.get_value(), 1.3), color=theme.MUTED, stroke_width=2))
    dots = [always_redraw(lambda k=k, s=s: Dot(axes.c2p(t.get_value(), math.sin(t.get_value() - s)), color=colors[k], radius=0.1))
            for k, s in enumerate(shifts)]
    ctx.add(scan, *dots)
    ctx.play(t.animate.set_value(4 * PI), run_time=max(1.0, ctx.remaining), rate_func=lambda x: x)


# --------------------------------------------------------------------------------------------------
def _phasors(raw: list) -> list[tuple[str, float, float]]:
    out = []
    for d in raw[:6]:
        try:
            out.append((str(d.get("label", "V"))[:6], abs(float(d.get("magnitude", 1))), float(d.get("angle_deg", 0))))
        except (TypeError, ValueError):
            continue
    return out or [("A", 1.0, 0.0), ("B", 1.0, -120.0), ("C", 1.0, 120.0)]


@component(
    "phasor_diagram", "Rotating phasor arrows on a circle. Shows magnitude and phase angle of voltages or currents (empty list = balanced three-phase set).",
    {"phasors": P("object[]", "Up to 6 phasors", default=[], item_hint="{label: string, magnitude: number, angle_deg: number}"),
     "rotate": P("boolean", "Spin the phasors counter-clockwise", default=True),
     "title": P("string", "Heading", default="")},
    {"phasors": [{"label": "Va", "magnitude": 1, "angle_deg": 0}, {"label": "Vb", "magnitude": 1, "angle_deg": -120},
                 {"label": "Vc", "magnitude": 1, "angle_deg": 120}], "rotate": True, "title": "Phasors"},
    domain="electrical",
)
def phasor_diagram(ctx: Ctx, phasors: list | None = None, rotate: bool = True, title: str = "") -> None:
    ph = _phasors(phasors or [])
    top = max(m for _, m, _ in ph) or 1.0
    scale = 2.0 / top
    c = CENTER
    ring = Circle(radius=2.3, color=theme.MUTED, stroke_width=2).move_to(c)
    xa = Line(c + LEFT * 2.6, c + RIGHT * 2.6, color=theme.MUTED, stroke_width=1.5)
    ya = Line(c + DOWN * 2.6, c + UP * 2.6, color=theme.MUTED, stroke_width=1.5)
    theta = ValueTracker(0)
    palette = theme.PHASE_COLORS + [theme.GOOD, theme.PURPLE, theme.WARN]
    mobs = []
    for i, (name, mag, ang) in enumerate(ph):
        def make(name=name, mag=max(mag, 0.15 * top), ang=ang, color=palette[i % len(palette)]):
            a = math.radians(ang) + theta.get_value()
            tip = c + _polar(scale * mag, a)
            return VGroup(Arrow(c, tip, buff=0, color=color, stroke_width=6, max_tip_length_to_length_ratio=0.2),
                          Text(name, font_size=26, color=color).move_to(c + _polar(scale * mag + 0.42, a)))
        mobs.append(always_redraw(make))
    if title:
        ctx.play(Write(ctx.heading(title)), run_time=0.7)
    ctx.play(Create(ring), Create(xa), Create(ya), run_time=0.8)
    ctx.play(LaggedStart(*[FadeIn(m) for m in mobs], lag_ratio=0.3), run_time=1.2)
    if rotate:
        ctx.play(theta.animate.set_value(TAU), run_time=max(1.5, ctx.remaining), rate_func=lambda x: x)


# --------------------------------------------------------------------------------------------------
def _lead(x0: float, x1: float) -> Line:
    return Line((x0, 0, 0), (x1, 0, 0), color=theme.FG, stroke_width=4)


def _mask() -> Rectangle:
    return Rectangle(width=1.5, height=1.0, fill_color=theme.BG, fill_opacity=1, stroke_width=0)


def _sym_resistor() -> VGroup:
    z = VMobject(color=theme.YELLOW, stroke_width=4)
    z.set_points_as_corners([np.array(p + (0,)) for p in [(-0.6, 0), (-0.42, 0), (-0.32, 0.25), (-0.12, -0.25), (0.08, 0.25),
                                                        (0.28, -0.25), (0.38, 0), (0.6, 0)]])
    return VGroup(_mask(), z)


def _sym_capacitor() -> VGroup:
    return VGroup(_mask(), _lead(-0.6, -0.12), _lead(0.12, 0.6),
                  Line((-0.12, -0.38, 0), (-0.12, 0.38, 0), color=theme.ACCENT, stroke_width=6),
                  Line((0.12, -0.38, 0), (0.12, 0.38, 0), color=theme.ACCENT, stroke_width=6))


def _sym_inductor() -> VGroup:
    arcs = [Arc(radius=0.1, start_angle=PI, angle=-PI, color=theme.PURPLE, stroke_width=4).move_arc_center_to(np.array([x, 0, 0]))
            for x in (-0.3, -0.1, 0.1, 0.3)]
    return VGroup(_mask(), _lead(-0.6, -0.4), _lead(0.4, 0.6), *arcs)


def _sym_battery() -> VGroup:
    return VGroup(_mask(), _lead(-0.6, -0.1), _lead(0.1, 0.6),
                  Line((-0.1, -0.4, 0), (-0.1, 0.4, 0), color=theme.GOOD, stroke_width=6),
                  Line((0.1, -0.22, 0), (0.1, 0.22, 0), color=theme.GOOD, stroke_width=9))


def _sym_ac() -> VGroup:
    wave = ParametricFunction(lambda t: np.array([t, 0.14 * math.sin(t * 4 * PI), 0]), t_range=[-0.25, 0.25, 0.01],
                              color=theme.ACCENT, stroke_width=4)
    return VGroup(_mask(), _lead(-0.6, -0.38), _lead(0.38, 0.6), Circle(radius=0.38, color=theme.FG, stroke_width=4), wave)


def _sym_lamp() -> VGroup:
    return VGroup(_mask(), _lead(-0.6, -0.35), _lead(0.35, 0.6), Circle(radius=0.35, color=theme.YELLOW, stroke_width=4),
                  Line((-0.25, -0.25, 0), (0.25, 0.25, 0), color=theme.YELLOW, stroke_width=3),
                  Line((-0.25, 0.25, 0), (0.25, -0.25, 0), color=theme.YELLOW, stroke_width=3))


def _sym_switch() -> VGroup:
    return VGroup(_mask(), _lead(-0.6, -0.35), _lead(0.35, 0.6), Dot((-0.35, 0, 0), radius=0.06, color=theme.FG),
                  Dot((0.35, 0, 0), radius=0.06, color=theme.FG),
                  Line((-0.35, 0, 0), (0.28, 0.32, 0), color=theme.WARN, stroke_width=4))


def _sym_generic() -> VGroup:
    return VGroup(_mask(), _lead(-0.6, -0.4), _lead(0.4, 0.6),
                  Rectangle(width=0.8, height=0.45, color=theme.FG, stroke_width=4))


_SYMBOLS = {"resistor": _sym_resistor, "capacitor": _sym_capacitor, "inductor": _sym_inductor, "battery": _sym_battery,
            "ac_source": _sym_ac, "lamp": _sym_lamp, "switch": _sym_switch, "generic": _sym_generic}
_ALIASES = {"r": "resistor", "res": "resistor", "c": "capacitor", "cap": "capacitor", "l": "inductor", "coil": "inductor",
            "v": "battery", "dc": "battery", "dc_source": "battery", "cell": "battery", "ac": "ac_source", "source": "ac_source",
            "ac_supply": "ac_source", "bulb": "lamp", "s": "switch"}
# (x, y, vertical?) on a rectangular loop of 8 x 3.6, relative to the loop centre
_W, _H = 8.0, 3.6
_SLOTS = {"L": (-_W / 2, 0, True), "R": (_W / 2, 0, True), "T": (0, _H / 2, False), "B": (0, -_H / 2, False),
          "TL": (-2, _H / 2, False), "TR": (2, _H / 2, False), "BL": (-2, -_H / 2, False), "BR": (2, -_H / 2, False)}
_ORDER = {1: ["L"], 2: ["L", "R"], 3: ["L", "T", "R"], 4: ["L", "T", "R", "B"], 5: ["L", "TL", "TR", "R", "B"],
          6: ["L", "TL", "TR", "R", "BR", "BL"]}


@component(
    "circuit", "A closed loop circuit with 1-6 elements and animated current flow. Element format 'type:label'; types: battery, ac_source, resistor, capacitor, inductor, lamp, switch.",
    {"elements": P("string[]", "e.g. ['battery:12 V', 'resistor:R = 4 Ω', 'capacitor:C']", required=True),
     "title": P("string", "Heading", default="")},
    {"elements": ["ac_source:230 V", "resistor:R", "inductor:L"], "title": "Series RL circuit"},
    domain="electrical",
)
def circuit(ctx: Ctx, elements: list[str], title: str = "") -> None:
    els = [e for e in elements if e.strip()][:6] or ["resistor:R"]
    center = CENTER + np.array([0, -0.1, 0])
    loop = Rectangle(width=_W, height=_H, color=theme.FG, stroke_width=4).move_to(center)
    syms = VGroup()
    labels = VGroup()
    for slot, raw in zip(_ORDER[len(els)], els):
        kind_raw, _, text = raw.partition(":")
        key = kind_raw.strip().lower().replace(" ", "_")
        kind = _ALIASES.get(key, key)
        if kind not in _SYMBOLS:
            kind, text = "generic", text or kind_raw
        x, y, vertical = _SLOTS[slot]
        s = _SYMBOLS[kind]()
        if vertical:
            s.rotate(PI / 2)
        s.move_to(center + np.array([x, y, 0]))
        syms.add(s)
        if text.strip():
            lab = label(text.strip(), 24, theme.FG, width=16)
            if slot in ("L",):
                lab.next_to(s, LEFT, buff=0.15)
            elif slot in ("R",):
                lab.next_to(s, RIGHT, buff=0.15)
            elif slot.startswith("B"):
                lab.next_to(s, DOWN, buff=0.15)
            else:
                lab.next_to(s, UP, buff=0.15)
            labels.add(lab)
    if title:
        ctx.play(Write(ctx.heading(title)), run_time=0.7)
    ctx.play(Create(loop), run_time=1.0)
    ctx.play(LaggedStart(*[FadeIn(s, scale=0.8) for s in syms], lag_ratio=0.3), run_time=1.2)
    ctx.play(FadeIn(labels), run_time=0.6)
    u = ValueTracker(0)
    dots = []
    for k in range(8):
        d = Dot(radius=0.07, color=theme.YELLOW)
        d.add_updater(lambda m, k=k: m.move_to(loop.point_from_proportion((u.get_value() + k / 8) % 1.0)))
        dots.append(d)
    ctx.add(*dots)
    ctx.play(u.animate.set_value(max(1.0, ctx.remaining / 4.0)), run_time=max(1.0, ctx.remaining), rate_func=lambda x: x)


# --------------------------------------------------------------------------------------------------
@component(
    "rotating_field", "Stator cross-section showing how three-phase currents create a rotating magnetic field, with the synchronous speed formula Ns = 120 f / P.",
    {"poles": P("integer", "Number of poles", default=2, choices=[2, 4, 6]),
     "frequency_hz": P("number", "Supply frequency in Hz", default=50)},
    {"poles": 2, "frequency_hz": 50},
    domain="electrical",
)
def rotating_field(ctx: Ctx, poles: int = 2, frequency_hz: float = 50.0) -> None:
    c = SIDE_CENTER
    stator = _stator(c)
    field = _pole_group(poles, c, 0.7, 1.95, 0.55, letters=False)
    hub = Dot(c, radius=0.09, color=theme.FG)
    panel = _speed_panel(poles, frequency_hz)
    note = label("Three AC phases 120° apart\nmake the field rotate", 22, theme.MUTED).move_to((PANEL_X, -1.3, 0))
    T = ValueTracker(0)
    omega = TAU / _rev_seconds(poles, frequency_hz)
    _spin(field, c, lambda: omega * T.get_value())
    ctx.play(Create(stator[0]), run_time=0.7)
    ctx.play(FadeIn(VGroup(*stator[1:])), run_time=0.7)
    ctx.play(FadeIn(field), FadeIn(hub), run_time=0.7)
    ctx.play(FadeIn(panel), FadeIn(note), run_time=0.8)
    ctx.play(T.animate.set_value(max(1.0, ctx.remaining)), run_time=max(1.0, ctx.remaining), rate_func=lambda x: x)


# --- synchronous motor ----------------------------------------------------------------------------------------------
# Layout: the machine on the left, a fixed left-aligned panel on the right. Colours carry meaning here:
#   yellow = the rotating STATOR field (what the supply makes), red / blue = the ROTOR magnet (N / S).
_SM_X = 0.85                       # left edge of the panel
_SM_DIAL = np.array([1.85, -0.5, 0.0])
_SM_MAX_DELTA = 90.0               # pull-out angle, electrical degrees
_SM_STATUS = [
    "Light load: the rotor trails the field by a small angle δ",
    "Load rising: δ grows, but the speed stays exactly Ns",
    "Heavy load: still locked in step, as long as δ stays under 90°",
]


def sm_delta_deg(t: float, ta: float, tb: float) -> float:
    """Load angle (electrical degrees) at time ``t``: light load until ``ta``, ramping to heavy load by ``tb``."""
    return 15.0 + 55.0 * _smooth((t - ta) / max(tb - ta, 1e-6))


def sm_torque_fraction(delta_deg: float) -> float:
    """Torque as a fraction of the pull-out torque: T = T_max * sin(delta), so 1.0 at 90 degrees."""
    return math.sin(math.radians(min(max(delta_deg, 0.0), _SM_MAX_DELTA)))


def _left(mob, x: float, y: float):
    """Place ``mob`` with its left edge at ``x`` and its vertical centre at ``y`` (so a panel never jitters as text changes)."""
    return mob.move_to((x + mob.width / 2, y, 0))


def _sm_legend() -> VGroup:
    rotor_chip = VGroup(Rectangle(width=0.17, height=0.3, color=theme.DANGER, fill_opacity=0.95, stroke_width=0),
                        Rectangle(width=0.17, height=0.3, color=theme.ACCENT, fill_opacity=0.95, stroke_width=0)).arrange(RIGHT, buff=0)
    stator_chip = Rectangle(width=0.34, height=0.3, color=theme.YELLOW, fill_opacity=0.6, stroke_width=0)
    rows = VGroup()
    for chip, text, y in ((rotor_chip, "Rotor: DC-excited magnet", 2.78),
                          (stator_chip, "Stator: rotating field", 2.34)):
        chip.move_to((_SM_X + 0.17, y, 0))
        t = _left(label(text, 20, theme.FG), _SM_X + 0.55, y)
        rows.add(VGroup(chip, t))
    return rows


def _sm_speed_block(poles: int, f: float) -> VGroup:
    ns = 120.0 * f / poles
    a = _left(label(f"Ns = 120 × {f:g} / {poles}", 26, theme.YELLOW), _SM_X, 1.62)
    b = _left(label(f"= {ns:g} rpm", 36, theme.YELLOW, weight="BOLD"), _SM_X, 1.1)
    note = _left(label("(slowed down)", 16, theme.MUTED), b.get_right()[0] + 0.3, 1.04)
    return VGroup(a, b, note)


@component(
    "synchronous_motor", "Cross-section of a synchronous motor: the rotor locks onto the rotating stator field and turns at exactly synchronous speed; a dial and torque meter show the load angle growing as load increases.",
    {"poles": P("integer", "Number of poles", default=2, choices=[2, 4, 6]),
     "frequency_hz": P("number", "Supply frequency in Hz", default=50)},
    {"poles": 2, "frequency_hz": 50},
    domain="electrical",
)
def synchronous_motor(ctx: Ctx, poles: int = 2, frequency_hz: float = 50.0) -> None:
    c = SIDE_CENTER
    mech = 2.0 / poles                                     # mechanical degrees per electrical degree
    omega = TAU / _rev_seconds(poles, frequency_hz)
    T = ValueTracker(0)

    # ---- the machine -------------------------------------------------------------------------------------------
    stator = _stator(c)
    field = _pole_group(poles, c, 1.72, 2.0, 0.5, letters=False)
    for seg in field:
        seg.set_color(theme.YELLOW)
    gap_ring = Circle(radius=1.43, color=theme.MUTED, stroke_width=1.5, stroke_opacity=0.5).move_to(c)
    body = Circle(radius=1.4, color=theme.MUTED, fill_color=theme.PANEL, fill_opacity=1, stroke_width=2).move_to(c)
    rotor = _pole_group(poles, c, 0.45, 1.36, 0.95, letters=False)
    shaft = VGroup(Circle(radius=0.45, color=theme.MUTED, fill_color=theme.BG, fill_opacity=1, stroke_width=2).move_to(c),
                   Dot(c, radius=0.08, color=theme.FG))
    letters = VGroup()
    for k in range(poles):                                 # upright N / S that ride on the rotor but never turn upside-down
        letters.add(Text("N" if k % 2 == 0 else "S", font_size=28, color=theme.FG, weight="BOLD"))

    total = max(3.0, ctx.remaining - 3.0)
    ta, tb = 0.3 * total, 0.8 * total
    rot_a = lambda: omega * T.get_value() - math.radians(sm_delta_deg(T.get_value(), ta, tb)) * mech   # rotor N axis
    fld_a = lambda: omega * T.get_value()                                                               # stator-field N axis
    for k, ch in enumerate(letters):
        ch.add_updater(lambda m, k=k: m.move_to(c + _polar(0.92, rot_a() + k * TAU / poles)))
        ch.move_to(c + _polar(0.92, k * TAU / poles))
    _spin(field, c, fld_a)
    _spin(rotor, c, rot_a)

    # pointers across the air gap, and the angle between them
    fld_tick = always_redraw(lambda: Line(c + _polar(1.46, fld_a()), c + _polar(1.74, fld_a()), color=theme.YELLOW, stroke_width=6))
    rot_tick = always_redraw(lambda: Line(c + _polar(1.2, rot_a()), c + _polar(1.62, rot_a()), color=theme.FG, stroke_width=6))
    gap_arc = always_redraw(lambda: Arc(radius=1.56, start_angle=rot_a(), angle=fld_a() - rot_a(), arc_center=c,
                                        color=theme.YELLOW, stroke_width=9))

    # ---- the panel ---------------------------------------------------------------------------------------------
    legend = _sm_legend()
    speed = _sm_speed_block(poles, frequency_hz)
    dial_ring = Circle(radius=0.88, color=theme.MUTED, fill_color=theme.PANEL, fill_opacity=0.9, stroke_width=2).move_to(_SM_DIAL)
    bs = Arrow(_SM_DIAL, _SM_DIAL + _polar(0.8, PI / 2), buff=0, color=theme.YELLOW, stroke_width=6, max_tip_length_to_length_ratio=0.3)
    br = always_redraw(lambda: Arrow(_SM_DIAL, _SM_DIAL + _polar(0.8, PI / 2 - math.radians(sm_delta_deg(T.get_value(), ta, tb))),
                                     buff=0, color=theme.DANGER, stroke_width=6, max_tip_length_to_length_ratio=0.3))
    d_arc = always_redraw(lambda: Arc(radius=0.46, start_angle=PI / 2 - math.radians(sm_delta_deg(T.get_value(), ta, tb)),
                                      angle=math.radians(sm_delta_deg(T.get_value(), ta, tb)), arc_center=_SM_DIAL,
                                      color=theme.YELLOW, stroke_width=5))
    stator_tag = label("stator", 16, theme.YELLOW).move_to(_SM_DIAL + (-0.58, 1.0, 0.0))
    rotor_tag = label("rotor", 16, theme.DANGER)
    rotor_tag.add_updater(lambda m: m.move_to(_SM_DIAL + _polar(1.2, PI / 2 - math.radians(sm_delta_deg(T.get_value(), ta, tb)))))
    dial_group = VGroup(dial_ring, bs, stator_tag)

    col = 3.75                                              # the readout column, right of the dial
    readout = _live_text(lambda: round(sm_delta_deg(T.get_value(), ta, tb)),
                         lambda d: _left(label(f"δ = {d}°", 32, theme.YELLOW, weight="BOLD"), col, -0.05))
    t_label = _left(label("Torque ∝ sin δ", 20, theme.MUTED), col, -0.62)
    track = Rectangle(width=3.0, height=0.2, color=theme.MUTED, fill_color=theme.PANEL, fill_opacity=1, stroke_width=2)
    track.move_to((col + 1.5, -1.0, 0))
    meter = always_redraw(lambda: _left(
        Rectangle(width=max(0.02, 3.0 * sm_torque_fraction(sm_delta_deg(T.get_value(), ta, tb))), height=0.2, stroke_width=0,
                  fill_opacity=1, color=theme.GOOD if sm_delta_deg(T.get_value(), ta, tb) < 60 else theme.WARN),
        col, -1.0))
    pull = _left(label("pull-out at δ = 90°", 16, theme.MUTED), col, -1.38)

    def status_text(k: int):
        return _left(label(_SM_STATUS[k], 21, theme.FG, width=36, align="left"), _SM_X, -2.05)

    status = _live_text(lambda: 0 if T.get_value() < ta else (1 if T.get_value() < tb else 2), status_text)

    # ---- the story ---------------------------------------------------------------------------------------------
    # One reveal (as on main): the machine, the panel and the readouts fade in together, so nothing appears piece by
    # piece and the right half of the screen is not left empty while the machine settles.
    ctx.play(FadeIn(VGroup(*stator, body, gap_ring, field, rotor, shaft, letters, legend, speed, fld_tick, rot_tick, gap_arc,
                           dial_group, br, d_arc, rotor_tag, readout, t_label, track, meter, pull, status)), run_time=1.4)
    ctx.add(fld_tick, rot_tick, gap_arc, br, d_arc, meter, letters)
    ctx.play(T.animate.set_value(total), run_time=max(1.0, ctx.remaining), rate_func=lambda x: x)
