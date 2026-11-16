"""Semiconductor physics pictures (README section 7.2): the PN junction, its band diagram, the diode curve, the MOSFET."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from manim import (
    DOWN, LEFT, RIGHT, UP, AnimationGroup, Arrow, Axes, DashedLine, DashedVMobject, Dot, DoubleArrow, Flash, GrowArrow, FadeOut, Line,
    Rectangle, Succession, ValueTracker, VGroup, VMobject,
)

from ...ctx import label
from ..core.flow import ParticleField
from ..core.palette import PALETTE, colormap
from ._frame import Framed
from ._math import (DEFAULT_VBI, channel_profile, depletion_split, depletion_width, diode_current, drain_current, mos_region,
                    pinch_point)

RW, RH = 3.4, 2.2  # half-width of each doped region and the height of the PN picture, in local units
DEP_SCALE = 1.2  # local length of the depletion region at zero bias (for equal doping)


def _set_vis(c, ring: bool, opacity: float) -> None:
    c.set_stroke(opacity=opacity) if ring else c.set_fill(opacity=opacity)


def _sign(plus: bool, color: str) -> VGroup:
    bar = Line((-0.07, 0, 0), (0.07, 0, 0), color=color, stroke_width=3)
    return VGroup(bar, Line((0, -0.07, 0), (0, 0.07, 0), color=color, stroke_width=3)) if plus else VGroup(bar)


@dataclass(frozen=True)
class PNData:
    bias: float
    width: float  # normalised depletion width (1.0 at zero bias)
    xp: float  # local half-widths of the depletion region on the P and N sides
    xn: float
    vbi: float
    regime: str  # "forward", "reverse" or "equilibrium"


class PNJunction(Framed, VGroup):
    """Intuition: two crowds mixing until a charged moat stops them. Reverse bias widens the moat; forward bias narrows it.

    ``bias`` is in volts, positive for forward. Holes are open rings, electrons are filled dots; carriers that fall inside the
    depletion region fade out and the fixed ions left behind (+ on the N side, - on the P side) set up the built-in field.
    """

    def __init__(self, bias: float = 0.0, n_carriers: int = 24, show_ions: bool = True, seed: int = 7, *,
                 na: float = 1.0, nd: float = 1.0, vbi: float = DEFAULT_VBI):
        super().__init__()
        if not 1 <= n_carriers <= 40:
            raise ValueError("n_carriers must be between 1 and 40")
        depletion_width(0.0, na, nd, vbi=vbi)  # validates the doping and vbi
        self.bias, self.na, self.nd, self.vbi, self.show_ions = ValueTracker(bias), na, nd, vbi, show_ions
        self._init_frame()
        self.p_region = Rectangle(width=RW, height=RH, stroke_color=PALETTE.orange, stroke_width=2,
                                  fill_color=PALETTE.orange, fill_opacity=0.10).move_to((-RW / 2, 0, 0))
        self.n_region = Rectangle(width=RW, height=RH, stroke_color=PALETTE.blue, stroke_width=2,
                                  fill_color=PALETTE.blue, fill_opacity=0.10).move_to((RW / 2, 0, 0))
        self.depletion = VMobject(stroke_color=PALETTE.muted, stroke_width=2, fill_color=PALETTE.muted, fill_opacity=0.25)
        self.holes = ParticleField(n_carriers, (RW - 0.4, RH - 0.4), kind="ring", seed=seed, color=PALETTE.orange)
        self.electrons = ParticleField(n_carriers, (RW - 0.4, RH - 0.4), kind="dot", seed=seed + 1, color=PALETTE.blue)
        self.holes.move_to((-RW / 2, 0, 0))
        self.electrons.move_to((RW / 2, 0, 0))
        self.ions: list[tuple[VGroup, bool, float, float]] = []  # (mobject, on N side, column fraction, row y)
        self._ion_group = VGroup()
        if show_ions:
            for frac in (0.25, 0.6, 0.95):
                for y in (-0.9, -0.45, 0.45, 0.9):
                    for n_side in (True, False):
                        ion = _sign(n_side, PALETTE.teal)
                        self.ions.append((ion, n_side, frac, y))
                        self._ion_group.add(ion)
        self.field = Arrow(RIGHT, LEFT, buff=0, color=PALETTE.yellow, stroke_width=5, max_tip_length_to_length_ratio=0.25)
        self.field_label = label("E", 24, PALETTE.yellow)
        self.caption_p = label("P-type", 24, PALETTE.orange).move_to((-RW / 2, -RH / 2 - 0.3, 0))
        self.caption_n = label("N-type", 24, PALETTE.blue).move_to((RW / 2, -RH / 2 - 0.3, 0))
        self.caption_dep = label("depletion region", 22, PALETTE.muted).move_to((0, RH / 2 + 0.3, 0))
        self.add(self.p_region, self.n_region, self.depletion, self._ion_group, self.holes, self.electrons, self.field,
                 self.field_label, self.caption_p, self.caption_n, self.caption_dep)
        self._sync()
        self.add_updater(lambda m: m._sync())

    # -- geometry ---------------------------------------------------------------------------------------------------
    def edges(self, v: float | None = None) -> tuple[float, float]:
        """Local half-widths ``(x_p, x_n)`` of the depletion region at bias ``v`` (default: the current bias)."""
        v = float(self.bias.get_value()) if v is None else v
        w, _ = depletion_width(v, self.na, self.nd, vbi=self.vbi)
        return depletion_split(w * DEP_SCALE, self.na, self.nd)

    def _sync(self) -> None:
        xp, xn = self.edges()
        h = RH / 2
        corners = [self.pt(-xp, -h), self.pt(xn, -h), self.pt(xn, h), self.pt(-xp, h), self.pt(-xp, -h)]
        self.depletion.set_points_as_corners(corners)
        for ion, n_side, frac, y in self.ions:
            ion.move_to(self.pt(frac * xn if n_side else -frac * xp, y))
        self.field.put_start_and_end_on(self.pt(0.7 * xn, 0), self.pt(-0.7 * xp, 0))
        self.field_label.move_to(self.pt(0.35 * (xn - xp), -0.3))
        for group, ring in ((self.holes, True), (self.electrons, False)):
            for c in group.submobjects:
                if getattr(c, "_managed", False):
                    continue
                inside = -xp <= self.local_x(c.get_center()[0]) <= xn
                want = 0.0 if inside else 1.0
                if getattr(c, "_vis", None) != want:
                    _set_vis(c, ring, want)
                    c._vis = want

    # -- the Mobject contract -----------------------------------------------------------------------------------------
    @property
    def data(self) -> PNData:
        v = float(self.bias.get_value())
        w, _ = depletion_width(v, self.na, self.nd, vbi=self.vbi)
        xp, xn = self.edges(v)
        return PNData(v, w, xp, xn, self.vbi, "equilibrium" if abs(v) < 1e-9 else ("forward" if v > 0 else "reverse"))

    def bind(self, tracker: ValueTracker, param: str = "bias") -> "PNJunction":
        """Drive the junction from ``tracker`` (volts). The only parameter is ``"bias"``."""
        if param != "bias":
            raise ValueError("PNJunction can only bind 'bias'")
        self.bias = tracker
        self._sync()
        return self

    def set_bias(self, v: float) -> "PNJunction":
        self.bias.set_value(v)
        self._sync()
        return self

    def animate_bias(self, v_from: float, v_to: float, *, run_time: float = 3.0):
        """The moat breathes: forward bias narrows it, reverse bias widens it. Starts from ``v_from``."""
        self.bias.set_value(v_from)
        return self.bias.animate.set_value(v_to).build().set_run_time(run_time)

    def animate_diffusion(self, *, run_time: float = 2.0):
        """Electrons and holes spill across the junction: two arrows grow toward the depletion edges, then fade."""
        xp, xn = self.edges()
        e_arrow = Arrow(self.pt(xn + 1.2, 0.8), self.pt(xn + 0.15, 0.8), buff=0, color=PALETTE.blue, stroke_width=5)
        h_arrow = Arrow(self.pt(-xp - 1.2, -0.8), self.pt(-xp - 0.15, -0.8), buff=0, color=PALETTE.orange, stroke_width=5)
        return Succession(AnimationGroup(GrowArrow(e_arrow), GrowArrow(h_arrow), run_time=run_time * 0.5),
                          AnimationGroup(FadeOut(e_arrow), FadeOut(h_arrow), run_time=run_time * 0.5))

    def animate_recombination(self, *, run_time: float = 2.0):
        """The electron and hole nearest the junction meet, flash and vanish."""
        xp, xn = self.edges()

        def nearest(group, side_sign):
            free = [c for c in group.submobjects if not getattr(c, "_managed", False) and getattr(c, "_vis", 1.0) == 1.0]
            if not free:
                raise ValueError("no free carrier is left to recombine")
            return min(free, key=lambda c: abs(self.local_x(c.get_center()[0])))

        e, h = nearest(self.electrons, 1), nearest(self.holes, -1)
        for c in (e, h):
            c._managed = True
        y_mid = 0.5 * (self.local_y(e) + self.local_y(h))
        mid = self.pt(0.0, y_mid)
        return Succession(
            AnimationGroup(e.animate.move_to(mid), h.animate.move_to(mid), run_time=run_time * 0.55),
            Flash(mid, color=PALETTE.yellow, line_length=0.15, flash_radius=0.25, run_time=run_time * 0.25),
            AnimationGroup(e.animate.set_opacity(0), h.animate.set_opacity(0), run_time=run_time * 0.2))

    def local_y(self, mob) -> float:
        return float((mob.get_center()[1] - self._a0.get_center()[1]) / max(self.unit, 1e-9))


EV = 0.8  # local units of band height per volt
EG = 1.0  # band gap, in local units
EC_N = 0.12  # conduction-band edge on the N side (the N side is the fixed reference)


class PNBandDiagram(Framed, VGroup):
    """Intuition: the junction as a hill. Electrons on the N side must climb the barrier to reach the P side; forward bias lowers it.

    Built from a ``PNJunction`` and reading its bias tracker every frame, so the bands bend in step with the moat above
    them. ``Ec`` and ``Ev`` are the band edges; the two flat grey lines are the Fermi levels, which separate by ``qV`` under bias.
    """

    def __init__(self, junction: PNJunction):
        super().__init__()
        self.junction = junction
        self._init_frame()
        self.shade = VMobject(stroke_width=0, fill_color=PALETTE.muted, fill_opacity=0.15)
        self.ec = VMobject(stroke_color=PALETTE.blue, stroke_width=4)
        self.ev = VMobject(stroke_color=PALETTE.orange, stroke_width=4)
        self.ef_p = VMobject(stroke_color=PALETTE.muted, stroke_width=2)
        self.ef_n = VMobject(stroke_color=PALETTE.muted, stroke_width=2)
        self.ref = VMobject(stroke_color=PALETTE.muted, stroke_width=1.5, stroke_opacity=0.6)
        self.barrier = DoubleArrow(DOWN, UP, buff=0, color=PALETTE.yellow, stroke_width=4, tip_length=0.15)
        self.barrier_label = label("barrier", 22, PALETTE.yellow)
        self.tags = VGroup(label("Ec", 20, PALETTE.blue).move_to((RW + 0.4, EC_N, 0)),
                           label("Ev", 20, PALETTE.orange).move_to((RW + 0.4, EC_N - EG, 0)),
                           label("Ef", 20, PALETTE.muted).move_to((RW + 0.4, 0.0, 0)),
                           label("electron energy", 22, PALETTE.muted).move_to((-RW + 1.1, 1.85, 0)))
        self.add(self.shade, self.ref, self.ef_p, self.ef_n, self.ec, self.ev, self.barrier, self.barrier_label, self.tags)
        self._barrier_on = None
        self._sync()
        self.add_updater(lambda m: m._sync())

    def band_edge(self, x: float, v: float | None = None) -> float:
        """Conduction-band energy (local units) at local position ``x`` for bias ``v``."""
        j = self.junction
        v = float(j.bias.get_value()) if v is None else v
        xp, xn = j.edges(v)
        b = max(j.vbi - v, 0.05 * j.vbi) * EV
        sp = xp / (xp + xn)
        if x <= -xp:
            return EC_N + b
        if x >= xn:
            return EC_N
        if x <= 0:
            return EC_N + b - b * sp * ((x + xp) / xp) ** 2
        return EC_N + b * (1 - sp) * ((xn - x) / xn) ** 2

    def _sync(self) -> None:
        j = self.junction
        v = float(j.bias.get_value())
        xp, xn = j.edges(v)
        b = max(j.vbi - v, 0.05 * j.vbi) * EV
        xs = np.unique(np.concatenate([np.linspace(-RW, RW, 70), [-xp, 0.0, xn]]))
        ec = [self.pt(x, self.band_edge(x, v)) for x in xs]
        self.ec.set_points_as_corners(ec)
        self.ev.set_points_as_corners([self.pt(x, self.band_edge(x, v) - EG) for x in xs])
        self.ef_p.set_points_as_corners([self.pt(-RW, -v * EV), self.pt(-xp, -v * EV)])
        self.ef_n.set_points_as_corners([self.pt(xn, 0.0), self.pt(RW, 0.0)])
        top, bottom = EC_N + b + 0.25, EC_N - EG - 0.1
        self.shade.set_points_as_corners([self.pt(-xp, bottom), self.pt(xn, bottom), self.pt(xn, top), self.pt(-xp, top),
                                          self.pt(-xp, bottom)])
        bx = RW * 0.72
        self.ref.set_points_as_corners([self.pt(0.0, EC_N + b), self.pt(bx + 0.2, EC_N + b)])
        show = b > 0.2
        if show:
            self.barrier.put_start_and_end_on(self.pt(bx, EC_N), self.pt(bx, EC_N + b))
            self.barrier_label.move_to(self.pt(bx + 0.55, EC_N + b / 2))
        if show != self._barrier_on:
            self.barrier.set_opacity(1.0 if show else 0.0)
            self.barrier_label.set_opacity(1.0 if show else 0.0)
            self._barrier_on = show

    @property
    def barrier_height(self) -> float:
        """Barrier height in volts, ``Vbi - V`` (clamped like the depletion model)."""
        j = self.junction
        return max(j.vbi - float(j.bias.get_value()), 0.05 * j.vbi)


class DiodeIVCurve(VGroup):
    """Intuition: a one-way valve. Almost nothing flows in reverse; past the knee the current takes off exponentially.

    The vertical scale is normalised so the curve reaches the top of the plot 50 mV below the largest forward voltage. The yellow dot is
    the operating point; ``bind`` it to the same tracker as a ``PNJunction`` and it climbs the curve as the moat narrows.
    """

    Y_TOP = 3.0

    def __init__(self, i_s: float = 1e-12, n: float = 1.0, v_range: tuple[float, float] = (-1.0, 0.8), *, width: float = 3.8,
                 height: float = 2.6):
        super().__init__()
        lo, hi = v_range
        if hi <= lo or lo >= 0 or hi <= 0:
            raise ValueError("v_range must be (low, high) with low < 0 < high")
        self.i_s, self.n, self.v_range = i_s, n, (lo, hi)
        self._i_ref = diode_current(hi - 0.05, i_s, n)  # normalise just below the top so the knee is visible
        self.axes = Axes(x_range=[lo, hi, 0.2], y_range=[-0.5, self.Y_TOP + 0.3, 1.0], x_length=width, y_length=height,
                         tips=False, axis_config={"color": PALETTE.muted, "stroke_width": 2, "include_ticks": False,
                                                 "include_numbers": False})
        self.curve = self.axes.plot(lambda v: self.display_current(v), x_range=[lo, hi, (hi - lo) / 160], color=PALETTE.blue,
                                    stroke_width=4)
        self.x_label = label("V", 22, PALETTE.muted).next_to(self.axes.x_axis.get_end(), DOWN, buff=0.15)
        self.y_label = label("I", 22, PALETTE.muted).next_to(self.axes.y_axis.get_end(), RIGHT, buff=0.15)
        self.tracker = ValueTracker(0.0)
        self.dot = Dot(radius=0.1, color=PALETTE.yellow)
        self.add(self.axes, self.curve, self.x_label, self.y_label, self.dot)
        self._sync()
        self.add_updater(lambda m: m._sync())

    def display_current(self, v: float) -> float:
        """Plot height for voltage ``v`` (current relative to the reference current, times the plot height, clipped at the top)."""
        return min(self.Y_TOP * diode_current(v, self.i_s, self.n) / self._i_ref, self.Y_TOP + 0.25)

    def current_at(self, v: float) -> float:
        return diode_current(v, self.i_s, self.n, )

    def _sync(self) -> None:
        v = min(max(float(self.tracker.get_value()), self.v_range[0]), self.v_range[1])
        self.dot.move_to(self.axes.c2p(v, self.display_current(v)))

    @property
    def data(self) -> dict:
        v = float(self.tracker.get_value())
        return {"v": v, "i": self.current_at(v), "i_s": self.i_s, "n": self.n}

    def bind(self, tracker: ValueTracker, param: str = "v") -> "DiodeIVCurve":
        if param != "v":
            raise ValueError("DiodeIVCurve can only bind 'v'")
        self.tracker = tracker
        self._sync()
        return self


SURF = 0.2  # silicon surface height in the MOSFET picture
HALF_L = 1.7  # half the channel length


def _smooth(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


@dataclass(frozen=True)
class MOSData:
    kind: str
    region: str
    vgs: float
    vds: float
    vth: float
    drain_current: float
    pinch: float | None  # fraction of the channel where it pinches off (None in cutoff)


class MOSFETCrossSection(Framed, VGroup):
    """Intuition: a plate that pulls carriers into a bridge under the oxide; the drain voltage tilts the bridge until it pinches off.

    Below ``vth`` there is no bridge (cutoff). Above it an inversion layer forms under the gate. Raising ``vds`` thins the
    layer toward the drain, and once ``vds`` passes ``vgs - vth`` the layer pinches off before reaching it (saturation).
    For ``kind="pmos"`` pass the voltage *magnitudes*; the picture swaps to a p+ source and drain over an N body with hole carriers.
    """

    def __init__(self, kind: str = "nmos", vgs: float = 0.0, vds: float = 0.0, vth: float = 1.0, *, k: float = 0.5):
        super().__init__()
        if kind not in ("nmos", "pmos"):
            raise ValueError("kind must be 'nmos' or 'pmos'")
        if vth <= 0:
            raise ValueError("vth must be positive (give the magnitude for a pmos)")
        self.kind, self.vth, self.k = kind, vth, k
        self.vgs, self.vds = ValueTracker(vgs), ValueTracker(vds)
        self._init_frame()
        n = kind == "nmos"
        body_c, sd_c, car_c = (PALETTE.orange, PALETTE.blue, PALETTE.blue) if n else (PALETTE.blue, PALETTE.orange, PALETTE.orange)
        self._ring = not n
        self._car_c = car_c
        body = Rectangle(width=6.4, height=1.6, stroke_color=body_c, stroke_width=2, fill_color=body_c, fill_opacity=0.10)
        body.move_to((0, SURF - 0.8, 0))
        src = Rectangle(width=1.5, height=0.8, stroke_color=sd_c, stroke_width=2, fill_color=sd_c, fill_opacity=0.35)
        drn = src.copy()
        src.move_to((-2.45, SURF - 0.4, 0))
        drn.move_to((2.45, SURF - 0.4, 0))
        oxide = Rectangle(width=2 * HALF_L, height=0.14, stroke_color=PALETTE.teal, stroke_width=1.5, fill_color=PALETTE.teal,
                          fill_opacity=0.5).move_to((0, SURF + 0.07, 0))
        gate = Rectangle(width=2 * HALF_L, height=0.4, stroke_color=PALETTE.muted, stroke_width=2, fill_color=PALETTE.muted,
                         fill_opacity=0.5).move_to((0, SURF + 0.34, 0))
        wires = VGroup(Line((-2.45, SURF, 0), (-2.45, 1.25, 0)), Line((0, SURF + 0.54, 0), (0, 1.25, 0)),
                       Line((2.45, SURF, 0), (2.45, 1.25, 0)), Line((0, SURF - 1.6, 0), (0, SURF - 1.75, 0)))
        wires.set_stroke(color=PALETTE.fg, width=2.5)
        tags = VGroup(label("S", 24, PALETTE.fg).move_to((-2.45, 1.5, 0)), label("G", 24, PALETTE.fg).move_to((0, 1.5, 0)),
                      label("D", 24, PALETTE.fg).move_to((2.45, 1.5, 0)), label("B", 24, PALETTE.fg).move_to((0, SURF - 1.95, 0)))
        doping = VGroup(label("n+" if n else "p+", 22, PALETTE.fg).move_to((-2.45, SURF - 0.4, 0)),
                        label("n+" if n else "p+", 22, PALETTE.fg).move_to((2.45, SURF - 0.4, 0)),
                        label("p-type body" if n else "n-type body", 20, PALETTE.muted).move_to((0, SURF - 1.3, 0)))
        self.depletion = VMobject(stroke_width=0, fill_color=PALETTE.teal, fill_opacity=0.15)
        self.inversion = VMobject(stroke_width=0, fill_color=car_c, fill_opacity=0.0)
        self.carriers = VGroup(*[(Dot(radius=0.06, color=car_c) if n else Dot(radius=0.06, color=car_c, fill_opacity=0.0)
                                  .set_stroke(color=car_c, width=2.5, opacity=1.0)) for _ in range(8)])
        self.pinch_dot = Dot(radius=0.09, color=PALETTE.yellow)
        self.pinch_label = label("pinch-off", 20, PALETTE.yellow)
        self.region_text = label("cutoff", 24, PALETTE.yellow)
        self._key, self._pinch_on = None, None
        self.add(body, src, drn, self.depletion, self.inversion, self.carriers, oxide, gate, wires, tags, doping, self.pinch_dot,
                 self.pinch_label, self.region_text)
        self._sync()
        self.add_updater(lambda m: m._sync())

    # -- numbers ------------------------------------------------------------------------------------------------------
    @property
    def region(self) -> str:
        return mos_region(float(self.vgs.get_value()), float(self.vds.get_value()), self.vth)

    def current(self) -> float:
        return drain_current(float(self.vgs.get_value()), float(self.vds.get_value()), self.vth, self.k)

    @property
    def i_ref(self) -> float:
        """Reference current (two volts of overdrive in saturation) used to scale arrows and readouts."""
        return 2.0 * self.k

    @property
    def data(self) -> MOSData:
        g, d = float(self.vgs.get_value()), float(self.vds.get_value())
        return MOSData(self.kind, mos_region(g, d, self.vth), g, d, self.vth, drain_current(g, d, self.vth, self.k),
                       pinch_point(g, d, self.vth))

    def channel_x(self, fraction: float) -> float:
        """Local x of a position along the channel (0 = source edge, 1 = drain edge)."""
        return -HALF_L + 2 * HALF_L * fraction

    # -- drawing --------------------------------------------------------------------------------------------------------
    def _sync(self) -> None:
        g, d = float(self.vgs.get_value()), float(self.vds.get_value())
        xs, depth = channel_profile(g, d, self.vth, n=21)
        top = [self.pt(self.channel_x(x), SURF) for x in xs]
        bottom = [self.pt(self.channel_x(x), SURF - 0.5 * dp) for x, dp in zip(xs[::-1], depth[::-1])]
        pts = top + bottom
        self.inversion.set_points_as_corners(pts + [pts[0]])
        strength = _smooth((g - self.vth) / 0.5)
        self.inversion.set_fill(opacity=0.55 * strength)
        dep_bottom = SURF - (0.55 + 0.25 * min(1.0, max(0.0, g / 3.0)))
        self.depletion.set_points_as_corners([self.pt(-HALF_L, SURF), self.pt(HALF_L, SURF), self.pt(HALF_L, dep_bottom),
                                              self.pt(-HALF_L, dep_bottom), self.pt(-HALF_L, SURF)])
        for i, c in enumerate(self.carriers.submobjects):
            f = (i + 0.5) / len(self.carriers.submobjects)
            dp = float(np.interp(f, xs, depth))
            c.move_to(self.pt(self.channel_x(f), SURF - 0.25 * dp - 0.02))
            on = 1.0 if dp > 0.03 and g > self.vth else 0.0
            if getattr(c, "_vis", None) != on:
                _set_vis(c, self._ring, on)
                c._vis = on
        p = pinch_point(g, d, self.vth)
        sat = p is not None and d >= (g - self.vth) - 1e-9 and d > 0
        if sat:
            self.pinch_dot.move_to(self.pt(self.channel_x(p), SURF - 0.06))
            self.pinch_label.move_to(self.pt(self.channel_x(p), SURF - 0.55))
        if sat != self._pinch_on:
            self.pinch_dot.set_opacity(1.0 if sat else 0.0)
            self.pinch_label.set_opacity(1.0 if sat else 0.0)
            self._pinch_on = sat
        key = mos_region(g, d, self.vth)
        if key != self._key:
            self.region_text.become(label(f"region: {key}", 24, PALETTE.yellow).move_to(self.pt(0, SURF - 2.35)))
            self._key = key

    # -- the Mobject contract --------------------------------------------------------------------------------------------
    def bind(self, tracker: ValueTracker, param: str) -> "MOSFETCrossSection":
        """Drive ``"vgs"`` or ``"vds"`` from ``tracker`` (volts; magnitudes for a pmos)."""
        if param not in ("vgs", "vds"):
            raise ValueError("param must be 'vgs' or 'vds'")
        setattr(self, param, tracker)
        self._sync()
        return self

    def animate_vgs(self, v_from: float, v_to: float, *, run_time: float = 3.0):
        """Raise the gate voltage: the inversion layer fades in as ``vgs`` crosses ``vth``."""
        self.vgs.set_value(v_from)
        return self.vgs.animate.set_value(v_to).build().set_run_time(run_time)

    def animate_vds(self, v_from: float, v_to: float, *, run_time: float = 3.0):
        """Raise the drain voltage: the layer tilts and thins toward the drain until it pinches off."""
        self.vds.set_value(v_from)
        return self.vds.animate.set_value(v_to).build().set_run_time(run_time)


class DrainCurrentVectors(VGroup):
    """Intuition: how hard the carriers are being pushed. Arrows run source to drain; longer arrows mean more drain current.

    Reads the device's trackers every frame, so the arrows are absent in cutoff, grow with ``vgs`` and stop growing once the
    device saturates (the current no longer depends on ``vds``).
    """

    N_ARROWS = 5

    def __init__(self, device: MOSFETCrossSection, scale: float = 1.0):
        super().__init__()
        if scale <= 0:
            raise ValueError("scale must be positive")
        self.device, self.scale_factor = device, scale
        self.arrows = [Arrow(LEFT, RIGHT, buff=0, color=PALETTE.yellow, stroke_width=4, max_tip_length_to_length_ratio=0.5)
                       for _ in range(self.N_ARROWS)]
        self.add(*self.arrows)
        self._visible = None
        self._sync()
        self.add_updater(lambda m: m._sync())

    def length(self) -> float:
        """Current arrow length in local units: zero in cutoff, 0.5 x scale at the reference current."""
        return self.scale_factor * 0.5 * min(1.0, self.device.current() / self.device.i_ref)

    def _sync(self) -> None:
        ln = self.length()
        show = ln >= 0.1
        if show:
            for i, a in enumerate(self.arrows):
                x = self.device.channel_x((i + 0.5) / self.N_ARROWS)
                a.put_start_and_end_on(self.device.pt(x - ln / 2, SURF - 0.22), self.device.pt(x + ln / 2, SURF - 0.22))
        if show != self._visible:
            for a in self.arrows:
                a.set_opacity(1.0 if show else 0.0)
            self._visible = show


def _plain_axes(x_max: float, y_max: float, width: float, height: float) -> Axes:
    return Axes(x_range=[0, x_max, x_max / 4], y_range=[0, y_max, y_max / 4], x_length=width, y_length=height, tips=False,
                axis_config={"color": PALETTE.muted, "stroke_width": 2, "include_ticks": False, "include_numbers": False})


class MOSFETCharacteristics(VGroup):
    """Intuition: a family of hills. Each gate voltage gets its own curve: it rises along the triode parabola, then flattens in saturation.

    The dashed line joins the points where each curve flattens (``vds = vgs - vth``). Call ``link(device)`` and the yellow
    dot follows the device's operating point from the cross-section.
    """

    def __init__(self, vth: float, k: float, vgs_list: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0), *, vds_max: float = 4.0,
                 width: float = 4.4, height: float = 3.0):
        super().__init__()
        if not 1 <= len(vgs_list) <= 8:
            raise ValueError("vgs_list needs between 1 and 8 values")
        if vth <= 0 or k <= 0 or vds_max <= 0:
            raise ValueError("vth, k and vds_max must be positive")
        self.vth, self.k, self.vds_max, self.vgs_list = vth, k, vds_max, tuple(sorted(vgs_list))
        top = drain_current(self.vgs_list[-1], vds_max, vth, k)
        self.y_max = (top if top > 0 else 0.5 * k) * 1.15
        self.axes = _plain_axes(vds_max, self.y_max, width, height)
        self.curves = VGroup()
        self.tags = VGroup()
        n = len(self.vgs_list)
        for i, vg in enumerate(self.vgs_list):
            col = colormap("blue_yellow", i, 0, max(1, n - 1))
            self.curves.add(self.axes.plot(lambda v, vg=vg: drain_current(vg, v, vth, k), x_range=[0, vds_max, 0.02], color=col,
                                           stroke_width=3.5))
            end = self.axes.c2p(vds_max, drain_current(vg, vds_max, vth, k))
            self.tags.add(label(f"{vg:g} V", 18, col).move_to(end + RIGHT * 0.38))
        self.edge = VGroup()
        cap = min(vds_max, self.vgs_list[-1] - vth)
        if cap > 0.05:
            curve = self.axes.plot(lambda v: 0.5 * k * v * v, x_range=[0, cap, 0.02], color=PALETTE.muted, stroke_width=2.5)
            self.edge.add(DashedVMobject(curve, num_dashes=14),
                          label("saturation edge", 18, PALETTE.muted).next_to(self.axes.c2p(cap, 0.5 * k * cap * cap), LEFT, buff=0.15))
        self.x_label = label("VDS", 22, PALETTE.muted).next_to(self.axes.x_axis.get_end(), DOWN, buff=0.15)
        self.y_label = label("ID", 22, PALETTE.muted).next_to(self.axes.y_axis.get_end(), RIGHT, buff=0.15)
        self.dot = Dot(radius=0.1, color=PALETTE.yellow).set_opacity(0.0)
        self.device = None
        self.add(self.axes, self.curves, self.tags, self.edge, self.x_label, self.y_label, self.dot)

    @property
    def data(self) -> dict:
        return {"vth": self.vth, "k": self.k, "vgs_list": self.vgs_list,
                "boundaries": [(vg - self.vth, 0.5 * self.k * (vg - self.vth) ** 2) for vg in self.vgs_list if vg > self.vth]}

    def link(self, device: MOSFETCrossSection) -> "MOSFETCharacteristics":
        """Make the yellow operating point follow ``device`` (its ``vgs`` and ``vds`` trackers)."""
        self.device = device
        self.dot.set_opacity(1.0)
        self.add_updater(lambda m: m._sync())
        self._sync()
        return self

    def _sync(self) -> None:
        if self.device is None:
            return
        g, d = float(self.device.vgs.get_value()), float(self.device.vds.get_value())
        i = drain_current(g, d, self.vth, self.k)
        self.dot.move_to(self.axes.c2p(min(max(d, 0.0), self.vds_max), min(i, self.y_max)))


class TransferCurve(VGroup):
    """Intuition: the switch's dimmer. Nothing flows until the gate crosses the threshold; after that the current grows as the square of the overdrive."""

    def __init__(self, vth: float, k: float, vgs_max: float = 4.0, *, width: float = 4.2, height: float = 3.0):
        super().__init__()
        if vth <= 0 or k <= 0 or vgs_max <= vth:
            raise ValueError("vth and k must be positive and vgs_max must exceed vth")
        self.vth, self.k, self.vgs_max = vth, k, vgs_max
        self.y_max = 0.5 * k * (vgs_max - vth) ** 2 * 1.15
        self.axes = _plain_axes(vgs_max, self.y_max, width, height)
        self.curve = self.axes.plot(lambda v: drain_current(v, 99.0, vth, k), x_range=[0, vgs_max, 0.02], color=PALETTE.blue,
                                    stroke_width=4)
        self.threshold = DashedLine(self.axes.c2p(vth, 0), self.axes.c2p(vth, self.y_max * 0.9), color=PALETTE.yellow,
                                    stroke_width=2.5)
        self.vth_label = label("Vth", 22, PALETTE.yellow).next_to(self.axes.c2p(vth, 0), DOWN, buff=0.15)
        self.x_label = label("VGS", 22, PALETTE.muted).next_to(self.axes.x_axis.get_end(), DOWN, buff=0.15)
        self.y_label = label("ID", 22, PALETTE.muted).next_to(self.axes.y_axis.get_end(), RIGHT, buff=0.15)
        self.dot = Dot(radius=0.1, color=PALETTE.yellow).set_opacity(0.0)
        self.device = None
        self.add(self.axes, self.curve, self.threshold, self.vth_label, self.x_label, self.y_label, self.dot)

    @property
    def data(self) -> dict:
        return {"vth": self.vth, "k": self.k, "i_at_max": drain_current(self.vgs_max, 99.0, self.vth, self.k)}

    def link(self, device: MOSFETCrossSection) -> "TransferCurve":
        self.device = device
        self.dot.set_opacity(1.0)
        self.add_updater(lambda m: m._sync())
        self._sync()
        return self

    def _sync(self) -> None:
        if self.device is None:
            return
        g, d = float(self.device.vgs.get_value()), float(self.device.vds.get_value())
        self.dot.move_to(self.axes.c2p(min(max(g, 0.0), self.vgs_max),
                                       min(drain_current(g, d, self.vth, self.k), self.y_max)))
