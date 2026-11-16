"""Physics pack, built on the community plugin manim-physics (https://github.com/Matheart/manim-physics).

Covers electrostatics, standing waves, pendulums, rigid-body collisions and thin lenses. The plugin's 3D
components (wave surfaces, magnetic fields) need a ThreeDScene, which the storyboard renderer does not use, so they
are not wrapped here.
"""
from __future__ import annotations

import math

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, Circle, Create, FadeIn, Line, Square, Triangle, VGroup, Write

from .. import theme
from ..ctx import STAGE_CENTER, Ctx, label
from ..registry import Param as P, component
from ._plugin import fit, require

_PIP = "manim-physics"
_CY = STAGE_CENTER[1]


# --------------------------------------------------------------------------------------------------------------
# Electric field
# --------------------------------------------------------------------------------------------------------------
def _charges_ok(v: list) -> str | None:
    if not 1 <= len(v) <= 4:
        return "needs 1 to 4 charges"
    if any(x == 0 or abs(x) > 3 for x in v):
        return "each charge must be non-zero and between -3 and 3"
    return None


@component(
    "electric_field",
    "Field arrows around 1-4 point charges.",
    {"charges": P("number[]", "signed strengths, 1-4 values in -3..3, e.g. [1, -1]", validator=_charges_ok)},
    {"charges": [2, -1]},
    domain="physics",
    plugin="manim-physics",
)
def electric_field(ctx: Ctx, charges: list[float] | None = None) -> None:
    mp = require("manim_physics", _PIP)
    charges = charges or [1, -1]
    n = len(charges)
    xs = [0.0] if n == 1 else list(np.linspace(-3.0, 3.0, n))
    items = [mp.Charge(float(q), np.array([x, _CY, 0.0]), add_glow=False) for q, x in zip(charges, xs)]
    field = mp.ElectricField(
        *items, x_range=[-6.2, 6.2, 0.8], y_range=[_CY - 2.4, _CY + 1.8, 0.8],
        colors=[theme.ACCENT, theme.YELLOW, theme.DANGER], length_func=lambda norm: 0.5 * (1 - math.exp(-2 * norm)),
    )
    head = ctx.heading("Electric field of point charges")
    legend = label("red = positive charge, blue = negative charge", 22, theme.MUTED)
    legend.next_to(head, DOWN, buff=0.15)
    ctx.play(Write(head), FadeIn(legend), run_time=0.8)
    ctx.play(FadeIn(VGroup(*items)), run_time=0.8)
    ctx.play(Create(field), run_time=min(2.5, max(1.0, ctx.remaining * 0.4)))
    ctx.wait(ctx.remaining)


# --------------------------------------------------------------------------------------------------------------
# Standing waves
# --------------------------------------------------------------------------------------------------------------
def _harmonics_ok(v: list) -> str | None:
    if not 1 <= len(v) <= 4:
        return "needs 1 to 4 harmonics"
    if any(not float(x).is_integer() or not 1 <= x <= 6 for x in v):
        return "each harmonic must be a whole number from 1 to 6"
    return None


@component(
    "standing_wave",
    "Animated standing waves for chosen harmonics.",
    {"harmonics": P("number[]", "1-4 numbers from 1-6, e.g. [1, 2, 3]", validator=_harmonics_ok)},
    {"harmonics": [1, 2, 3]},
    domain="physics",
    plugin="manim-physics",
)
def standing_wave(ctx: Ctx, harmonics: list[float] | None = None) -> None:
    mp = require("manim_physics", _PIP)
    harmonics = [int(h) for h in (harmonics or [1, 2, 3])]
    amp = min(0.6, 2.0 / len(harmonics))
    colors = [theme.ACCENT, theme.GOOD, theme.YELLOW, theme.PURPLE]
    tags, waves = VGroup(), VGroup()
    for k, n in enumerate(harmonics):
        waves.add(mp.StandingWave(n, length=7.0, period=3.0 / n, amplitude=amp, color=colors[k % len(colors)], stroke_width=5))
        tags.add(label(f"n = {n}" + (" (fundamental)" if n == 1 else ""), 24, theme.FG, align="left"))
    waves.arrange(DOWN, buff=0.35)
    for tag, wave in zip(tags, waves):  # one label column, so every wave starts at the same x
        tag.move_to(wave.get_center())
    tags.arrange(DOWN, buff=0.0)
    for tag, wave in zip(tags, waves):
        tag.set_y(wave.get_y())
    rows = VGroup(tags, waves).arrange(RIGHT, buff=0.5, aligned_edge=UP)
    for tag, wave in zip(tags, waves):  # re-align vertically after the side-by-side arrangement
        tag.set_y(wave.get_y())
    fit(rows, max_w=12.0, max_h=4.6, center=(0, 0.2, 0), max_scale=1.0)
    head = ctx.heading("Standing waves")
    ctx.play(Write(head), run_time=0.7)
    ctx.play(FadeIn(rows), run_time=0.8)
    for wave in waves:
        wave.start_wave()  # time-based updater: the waves oscillate during the waits below
    ctx.wait(ctx.remaining)


# --------------------------------------------------------------------------------------------------------------
# Pendulum and collisions (pymunk rigid-body simulation)
# --------------------------------------------------------------------------------------------------------------
class _SpaceHost:
    """Lets manim-physics' ``SpaceScene`` helpers run inside the storyboard's plain ``Scene``.

    ``SpaceScene`` is a Scene subclass, but its methods only need ``add``, ``add_body`` and a ``space`` attribute, so
    they are borrowed here instead of changing the scene class. The space is stepped in small sub-steps because
    low-quality renders run at 15 fps, and one big step per frame lets fast bodies tunnel through thin walls.
    """

    SUBSTEPS = 10

    def __init__(self, scene, gravity: tuple[float, float] = (0.0, -9.81)):
        rm = require("manim_physics.rigid_mechanics.rigid_mechanics", _PIP)
        self._cls = rm.SpaceScene
        self.scene = scene
        self.space = rm.Space(gravity=gravity)
        scene.add(self.space)
        self.space.add_updater(self._step)

    def _step(self, space, dt: float) -> None:
        for _ in range(self.SUBSTEPS):
            space.space.step(dt / self.SUBSTEPS)

    def add(self, *mobjects) -> None:
        self.scene.add(*mobjects)

    def add_body(self, body) -> None:
        self._cls.add_body(self, body)

    def make_rigid_body(self, *mobs, **kw) -> None:
        self._cls.make_rigid_body(self, *mobs, **kw)

    def make_static_body(self, *mobs, **kw) -> None:
        self._cls.make_static_body(self, *mobs, **kw)

    def stop_rigidity(self, *mobs) -> None:
        self._cls.stop_rigidity(self, *mobs)

    def release(self, *mobs) -> None:
        """Detach the simulation. pymunk objects hold locks that ``deepcopy`` cannot copy, and the scene copies every
        mobject when it fades the component out, so the physics state must be gone before the component returns."""
        self.space.clear_updaters()
        self.scene.remove(self.space)
        for mob in mobs:
            for sub in mob.get_family():
                sub.clear_updaters(recursive=False)
                for attr in ("body", "shape", "spacescene"):
                    sub.__dict__.pop(attr, None)


@component(
    "pendulum",
    "Simulated pendulum; 2-3 bobs are chaotic with a traced path.",
    {"bobs": P("integer", "number of bobs", default=1, choices=[1, 2, 3]),
     "length": P("number", "length in metres, 1-3.4", default=3.0,
                 validator=lambda v: None if 1 <= v <= 3.4 else "must be between 1 and 3.4"),
     "angle_degrees": P("number", "start angle in degrees, 5-90", default=35,
                        validator=lambda v: None if 5 <= v <= 90 else "must be between 5 and 90")},
    {"bobs": 1, "length": 3.0, "angle_degrees": 35},
    domain="physics",
    plugin="manim-physics",
)
def pendulum(ctx: Ctx, bobs: int = 1, length: float = 3.0, angle_degrees: float = 35) -> None:
    mp = require("manim_physics", _PIP)
    from manim import ORANGE, TracedPath  # noqa: PLC0415
    host = _SpaceHost(ctx.scene)
    # A multi-bob rig can swing all the way over the top, so it hangs from the middle of the stage and is shorter.
    if bobs > 1:
        length = min(length, 2.3)
    pivot = np.array([0.0, 2.3 if bobs == 1 else 0.6, 0.0])
    theta = math.radians(angle_degrees)
    seg = length / bobs
    points, p = [], pivot.copy()
    for _ in range(bobs):
        p = p + seg * np.array([math.sin(theta), -math.cos(theta), 0.0])
        points.append(p.copy())
    bob_style = {"radius": 0.24 if bobs == 1 else 0.15, "color": ORANGE, "fill_opacity": 1}
    rod_style = {"color": theme.FG, "stroke_width": 4}
    if bobs == 1:
        rig = mp.Pendulum(length=length, initial_theta=theta, pivot_point=pivot, rod_style=rod_style, bob_style=bob_style)
    else:
        rig = mp.MultiPendulum(*points, pivot_point=pivot, rod_style=rod_style, bob_style=bob_style)
    ceiling = Line(pivot + LEFT * 1.2, pivot + RIGHT * 1.2, color=theme.MUTED, stroke_width=6)
    head = ctx.heading({1: "Simple pendulum", 2: "Double pendulum", 3: "Triple pendulum"}[bobs])
    if bobs == 1:
        period = 2 * math.pi * math.sqrt(length / 9.81)
        note = label(f"L = {length:g} m, small-angle period about {period:.1f} s", 22, theme.MUTED)
    else:
        note = label("Tiny changes in the start give a completely different path (chaos)", 22, theme.MUTED)
    note.next_to(head, DOWN, buff=0.15)
    ctx.play(Write(head), FadeIn(note), FadeIn(ceiling), FadeIn(rig), run_time=1.0)
    host.make_rigid_body(*rig.bobs)
    rig.start_swinging()
    trace = None
    if bobs > 1:
        trace = TracedPath(rig.bobs[-1].get_center, stroke_color=theme.ACCENT, stroke_width=3, dissipating_time=4)
        ctx.add(trace)
    try:
        ctx.wait(ctx.remaining)
    finally:
        if trace is not None:
            ctx.scene.remove(trace)
        host.release(rig)


@component(
    "falling_objects",
    "Shapes dropped under gravity, bouncing off floor and walls.",
    {"objects": P("integer", "shapes, 1-5", default=3,
                  validator=lambda v: None if 1 <= v <= 5 else "must be between 1 and 5"),
     "bounciness": P("number", "0 (dead) to 0.95 (very bouncy)", default=0.7,
                     validator=lambda v: None if 0 <= v <= 0.95 else "must be between 0 and 0.95")},
    {"objects": 3, "bounciness": 0.7},
    domain="physics",
    plugin="manim-physics",
)
def falling_objects(ctx: Ctx, objects: int = 3, bounciness: float = 0.7) -> None:
    require("manim_physics", _PIP)
    host = _SpaceHost(ctx.scene)
    floor_y, half_w = -2.2, 5.0
    floor = Line([-half_w, floor_y, 0], [half_w, floor_y, 0], color=theme.MUTED)
    left = Line([-half_w, floor_y, 0], [-half_w, 3.0, 0], color=theme.MUTED)
    right = Line([half_w, floor_y, 0], [half_w, 3.0, 0], color=theme.MUTED)
    makers = [lambda: Circle(radius=0.35), lambda: Square(side_length=0.6), lambda: Triangle().scale(0.45)]
    colors = [theme.DANGER, theme.YELLOW, theme.ACCENT, theme.GOOD, theme.PURPLE]
    xs = np.linspace(-3.2, 3.2, objects) if objects > 1 else [0.0]
    shapes = VGroup()
    for k, x in enumerate(xs):
        s = makers[k % 3]()
        s.set_fill(colors[k % len(colors)], 1).set_stroke(theme.FG, 2)
        s.move_to([x + 0.2 * (k % 2), 1.0 + 0.7 * k, 0])
        if k % 3 == 1:
            s.rotate(0.4)
        shapes.add(s)
    head = ctx.heading("Falling and bouncing")
    note = label(f"bounciness (elasticity) = {bounciness:g}", 22, theme.MUTED).next_to(head, DOWN, buff=0.15)
    ctx.play(Write(head), FadeIn(note), Create(floor), Create(left), Create(right), FadeIn(shapes), run_time=1.0)
    host.make_static_body(floor, left, right)  # elasticity multiplies with the shapes', so 1 keeps theirs unchanged
    host.make_rigid_body(*shapes, elasticity=float(bounciness), friction=0.6)
    try:
        ctx.wait(ctx.remaining)
    finally:
        host.release(floor, left, right, shapes)


# --------------------------------------------------------------------------------------------------------------
# Lens and light rays
# --------------------------------------------------------------------------------------------------------------
@component(
    "lens_rays",
    "Parallel rays through a convex or concave lens.",
    {"kind": P("string", "lens type", default="converging", choices=["converging", "diverging"]),
     "rays": P("integer", "rays, 3-9", default=5,
               validator=lambda v: None if 3 <= v <= 9 else "must be between 3 and 9")},
    {"kind": "converging", "rays": 5},
    domain="physics",
    plugin="manim-physics",
)
def lens_rays(ctx: Ctx, kind: str = "converging", rays: int = 5) -> None:
    mp = require("manim_physics", _PIP)
    # Small |f| and thickness keep the lens (and the rays' bend) inside the frame; the plugin sizes the lens from them.
    f = 3.5 if kind == "converging" else -3.5
    lens = mp.Lens(f, 0.6, fill_opacity=0.5, color=theme.ACCENT).move_to([-0.8, _CY, 0])
    span = 1.0 if kind == "converging" else 1.5  # keep every ray inside the lens: the convex one is shorter
    beams = [mp.Ray(np.array([-5.5, _CY + y, 0.0]), RIGHT, 9, [lens], color=theme.WARN, stroke_width=4)
             for y in np.linspace(-span, span, rays)]
    head = ctx.heading("Convex lens: rays converge" if kind == "converging" else "Concave lens: rays spread out")
    ctx.play(Write(head), FadeIn(lens), run_time=0.9)
    ctx.play(*[Create(b) for b in beams], run_time=min(2.0, max(1.0, ctx.remaining * 0.4)))
    ctx.wait(ctx.remaining)
