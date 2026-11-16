"""Pictures on the complex plane and the frequency axis: pole-zero map, Bode, Nyquist, root locus (README section 8.2)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, Arrow, Circle, Create, DashedLine, DashedVMobject, Dot, DoubleArrow, Line, Rectangle, Succession,
                   UpdateFromAlphaFunc, ValueTracker, VGroup, linear)

from ...ctx import label
from ..core.live import LiveValue
from ..core.palette import PALETTE
from ._math import (TF, ROOT_TOL, bode_asymptotes, bode_data, ceil_to, complex_list, nyquist_data, root_locus_data, tf_from_pz)
from ._parts import frame_for, hide, match_scale, plain_axes, pole_marker, polyline, reveal, runs_inside, zero_marker


# --------------------------------------------------------------------------------------------------------------------
# Pole-zero map
# --------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PZData:
    poles: tuple[complex, ...]
    zeros: tuple[complex, ...]
    domain: str
    stable: bool  # s-plane: every pole left of the imaginary axis. z-plane: every pole inside the unit circle


class PoleZeroPlot(VGroup):
    """Intuition: the map of where the system's natural motions live. Poles are crosses, zeros are circles; stay in the safe
    region (left half of the s-plane, inside the unit circle of the z-plane) and every motion dies out.

    Drag a pole across the line and the status word flips. ``domain`` is ``"s"`` or ``"z"``.
    """

    def __init__(self, poles=(), zeros=(), domain: str = "s", *, size: float = 3.6, extent: float | None = None):
        super().__init__()
        self._size = size
        if domain not in ("s", "z"):
            raise ValueError("domain must be 's' or 'z'")
        self.domain, self.poles, self.zeros = domain, complex_list(poles), complex_list(zeros)
        if not self.poles and not self.zeros:
            raise ValueError("give at least one pole or zero")
        reach = max([abs(r) for r in self.poles + self.zeros] + [1.0 if domain == "z" else 0.5])
        self.extent = extent if extent is not None else max(ceil_to(1.35 * reach, 0.5), 1.5 if domain == "z" else 1.0)
        e = self.extent
        self.axes = plain_axes([-e, e, e], [-e, e, e], size, size)
        self.frame = frame_for(self.axes, size, size)
        self.region = Rectangle(width=size / 2, height=size, stroke_width=0, fill_color=PALETTE.teal, fill_opacity=0.10)
        if domain == "s":
            self.region.move_to(self.axes.c2p(-e / 2, 0))
            self.boundary = Line(self.axes.c2p(0, -e), self.axes.c2p(0, e), color=PALETTE.yellow, stroke_width=3)
            tags = [label("σ", 22, PALETTE.muted).next_to(self.axes.c2p(e, 0), UP, buff=0.1),
                    label("jω", 22, PALETTE.muted).next_to(self.axes.c2p(0, e), RIGHT, buff=0.1)]
            self.zone = label("stable side", 18, PALETTE.teal).move_to(self.axes.c2p(-e / 2, -0.88 * e))
            if size < 2.5:  # an inset: keep only the picture and the verdict
                tags, self.zone = [], VGroup()
        else:
            unit = self.axes.c2p(1, 0)[0] - self.axes.c2p(0, 0)[0]
            self.region = Circle(radius=unit, stroke_width=0, fill_color=PALETTE.teal, fill_opacity=0.10).move_to(self.axes.c2p(0, 0))
            self.boundary = Circle(radius=unit, stroke_color=PALETTE.yellow, stroke_width=3, fill_opacity=0).move_to(self.axes.c2p(0, 0))
            tags = [label("Re", 22, PALETTE.muted).next_to(self.axes.c2p(e, 0), UP, buff=0.1),
                    label("Im", 22, PALETTE.muted).next_to(self.axes.c2p(0, e), RIGHT, buff=0.1)]
            self.zone = label("unit circle", 18, PALETTE.teal).move_to(self.axes.c2p(0, -0.88 * e))
            if size < 2.5:
                tags, self.zone = [], VGroup()
        self.pole_marks = [pole_marker() for _ in self.poles]
        self.zero_marks = [zero_marker() for _ in self.zeros]
        self.status = label("", 22, PALETTE.green)
        self.add(self.region, self.axes, self.frame, self.boundary, *tags, self.zone, *self.zero_marks, *self.pole_marks, self.status)
        self._sync()

    def _stable(self) -> bool:
        return all((p.real < -ROOT_TOL) if self.domain == "s" else (abs(p) < 1 - ROOT_TOL) for p in self.poles)

    def _sync(self) -> None:
        for z, m in zip(self.poles, self.pole_marks):
            m.move_to(self.axes.c2p(z.real, z.imag))
        for z, m in zip(self.zeros, self.zero_marks):
            m.move_to(self.axes.c2p(z.real, z.imag))
        ok = self._stable()
        word = ("stable" if ok else "unstable") if self.poles else "no poles"
        new = match_scale(label(word, 24, PALETTE.green if ok else PALETTE.red), self.frame, self._size).next_to(self.frame, DOWN, buff=0.2)
        self.status.become(new)

    @property
    def data(self) -> PZData:
        return PZData(tuple(self.poles), tuple(self.zeros), self.domain, self._stable())

    def set_pole(self, index: int, value: complex) -> "PoleZeroPlot":
        """Move pole ``index`` to ``value`` immediately; its conjugate partner (if any) follows."""
        old = self.poles[index]
        self.poles[index] = complex(value)
        if abs(old.imag) > ROOT_TOL:
            for j, p in enumerate(self.poles):
                if j != index and abs(p - old.conjugate()) < 1e-6:
                    self.poles[j] = complex(value).conjugate()
                    break
        self._sync()
        return self

    def animate_move_pole(self, index: int, to: complex, *, run_time: float = 2.0):
        """Slide pole ``index`` (and its conjugate partner) to ``to``; the status word flips as it crosses the boundary."""
        start, to = self.poles[index], complex(to)
        return UpdateFromAlphaFunc(self, lambda m, a: m.set_pole(index, start + (to - start) * a), run_time=run_time)


# --------------------------------------------------------------------------------------------------------------------
# Bode plot
# --------------------------------------------------------------------------------------------------------------------
class BodePlot(VGroup):
    """Intuition: the system's answer to a pure tone, at every pitch. Top: how loud it comes back; bottom: how late.

    Dashed teal lines are the straight-line approximation: each pole bends the slope down by 20 dB per decade at its
    corner. The yellow marks (``animate_margins``) show how far the loop is from instability: the gain margin on the top
    plot and the phase margin on the bottom plot.
    """

    def __init__(self, poles=(), zeros=(), gain: float = 1.0, w_range: tuple[float, float] | None = None, asymptotes: bool = True,
                 margins: bool = True, *, width: float = 5.6, height: float = 1.9):
        super().__init__()
        self.tf = tf_from_pz(poles, zeros, gain)
        self._poles, self._zeros, self._gain = complex_list(poles), complex_list(zeros), gain
        self.bode = bode_data(self.tf, w_range=w_range)
        b = self.bode
        lw = np.log10(b.w)
        self.x_lo, self.x_hi = math.floor(lw[0] + 1e-9), math.ceil(lw[-1] - 1e-9)
        m_lo = min(-20.0, ceil_to(float(b.mag_db.min()) - 10, 20) if b.mag_db.min() < -10 else -20.0)
        m_hi = max(20.0, ceil_to(float(b.mag_db.max()) + 10, 20))
        p_lo = min(-180.0, math.floor(float(b.phase_deg.min()) / 90.0) * 90.0)
        p_hi = max(0.0, ceil_to(float(b.phase_deg.max()), 90.0))
        self.mag_axes = plain_axes([self.x_lo, self.x_hi, 1], [m_lo, m_hi, 20], width, height)
        self.phase_axes = plain_axes([self.x_lo, self.x_hi, 1], [p_lo, p_hi, 90], width, height)
        self.phase_axes.next_to(self.mag_axes, DOWN, buff=0.6)
        self.mag_axes.set_opacity(0.0)  # the frame and dashed guides replace the axis lines (x = 0 is 1 rad/s, not an origin)
        self.phase_axes.set_opacity(0.0)
        self.mag_frame, self.phase_frame = frame_for(self.mag_axes, width, height), frame_for(self.phase_axes, width, height)
        self.magnitude_curve = polyline([self.mag_axes.c2p(x, y) for x, y in zip(lw, b.mag_db)], color=PALETTE.blue, width=4)
        self.phase_curve = polyline([self.phase_axes.c2p(x, y) for x, y in zip(lw, b.phase_deg)], color=PALETTE.blue, width=4)
        self.guides = VGroup(
            DashedLine(self.mag_axes.c2p(self.x_lo, 0), self.mag_axes.c2p(self.x_hi, 0), color=PALETTE.muted, stroke_width=2),
            DashedLine(self.phase_axes.c2p(self.x_lo, -180), self.phase_axes.c2p(self.x_hi, -180), color=PALETTE.muted, stroke_width=2))
        m_step = 20 if m_hi - m_lo <= 120 else 40
        self.y_labels = VGroup(
            *[label(f"{v}", 18, PALETTE.muted).next_to(self.mag_axes.c2p(self.x_lo, v), LEFT, buff=0.12)
              for v in range(int(ceil_to(m_lo, m_step)), int(m_hi) + 1, m_step)],
            *[label(f"{v}°", 18, PALETTE.muted).next_to(self.phase_axes.c2p(self.x_lo, v), LEFT, buff=0.12)
              for v in range(int(p_lo), int(p_hi) + 1, 90)])
        self.asymptote_lines = VGroup()
        if asymptotes:
            db = np.clip(bode_asymptotes(self._poles, self._zeros, gain, b.w), m_lo, m_hi)
            self.asymptote_lines.add(DashedVMobject(polyline([self.mag_axes.c2p(x, y) for x, y in zip(lw, db)], color=PALETTE.teal, width=3),
                                                    num_dashes=60))
        self.margin_marks = VGroup()
        if margins:
            self._build_margins()
        hide(self.margin_marks)
        decades = list(range(self.x_lo, self.x_hi + 1))
        step = 1 if len(decades) <= 6 else 2
        self.decade_labels = VGroup(*[label(f"{10.0 ** k:g}", 18, PALETTE.muted).next_to(self.phase_axes.c2p(k, p_lo), DOWN, buff=0.12)
                                      for k in decades[::step]])
        self.axis_names = VGroup(label("ω (rad/s)", 20, PALETTE.muted).next_to(self.decade_labels, DOWN, buff=0.1),
                                 label("gain (dB)", 20, PALETTE.muted).next_to(self.mag_frame, UP, buff=0.1),
                                 label("phase", 20, PALETTE.muted).next_to(self.phase_frame, UP, buff=0.1))
        self.sweep = ValueTracker(float(lw[0]))
        self.mag_dot, self.phase_dot = Dot(radius=0.09, color=PALETTE.yellow), Dot(radius=0.09, color=PALETTE.yellow)
        self.add(self.mag_axes, self.phase_axes, self.mag_frame, self.phase_frame, self.guides, self.y_labels, self.asymptote_lines, self.magnitude_curve,
                 self.phase_curve, self.margin_marks, self.decade_labels, self.axis_names, self.mag_dot, self.phase_dot)
        self._sync()
        self.add_updater(lambda m: m._sync())

    def _build_margins(self) -> None:
        b, ma, pa = self.bode, self.mag_axes, self.phase_axes
        if b.w_gc is not None and b.phase_margin_deg is not None:
            x, ph = math.log10(b.w_gc), b.phase_margin_deg - 180.0
            self.margin_marks.add(DashedLine(ma.c2p(x, ma.y_range[0]), ma.c2p(x, ma.y_range[1]), color=PALETTE.yellow, stroke_width=2.5),
                                  DashedLine(pa.c2p(x, pa.y_range[0]), pa.c2p(x, pa.y_range[1]), color=PALETTE.yellow, stroke_width=2.5),
                                  DoubleArrow(pa.c2p(x, -180), pa.c2p(x, ph), buff=0, color=PALETTE.yellow, stroke_width=4,
                                              max_tip_length_to_length_ratio=0.3),
                                  label(f"phase margin {b.phase_margin_deg:.0f}°", 20, PALETTE.yellow).next_to(self.phase_frame, RIGHT, buff=0.25))
        if b.w_pc is not None and b.gain_margin_db is not None:
            x, g = math.log10(b.w_pc), -b.gain_margin_db
            self.margin_marks.add(DashedLine(ma.c2p(x, ma.y_range[0]), ma.c2p(x, ma.y_range[1]), color=PALETTE.yellow, stroke_width=2.5),
                                  DoubleArrow(ma.c2p(x, g), ma.c2p(x, 0), buff=0, color=PALETTE.yellow, stroke_width=4,
                                              max_tip_length_to_length_ratio=0.3),
                                  label(f"gain margin {b.gain_margin_db:.0f} dB", 20, PALETTE.yellow).next_to(self.mag_frame, RIGHT, buff=0.25))

    def _sync(self) -> None:
        x = float(np.clip(self.sweep.get_value(), self.x_lo, self.x_hi))
        lw = np.log10(self.bode.w)
        self.mag_dot.move_to(self.mag_axes.c2p(x, float(np.clip(np.interp(x, lw, self.bode.mag_db), *self.mag_axes.y_range[:2]))))
        self.phase_dot.move_to(self.phase_axes.c2p(x, float(np.clip(np.interp(x, lw, self.bode.phase_deg), *self.phase_axes.y_range[:2]))))

    @property
    def data(self):
        return self.bode

    def bind(self, tracker: ValueTracker, param: str = "omega") -> "BodePlot":
        """Drive the sweep dots from ``tracker`` holding log10(ω). The only parameter is ``"omega"``."""
        if param != "omega":
            raise ValueError("BodePlot can only bind 'omega' (as log10 of rad/s)")
        self.sweep = tracker
        self._sync()
        return self

    def animate_sweep(self, *, run_time: float = 5.0):
        """A dot walks up in frequency along both curves, so the viewer reads gain and delay at the same pitch."""
        self.sweep.set_value(float(np.log10(self.bode.w[0])))
        return self.sweep.animate.set_value(float(np.log10(self.bode.w[-1]))).build().set_run_time(run_time).set_rate_func(linear)

    def animate_margins(self, *, run_time: float = 1.2):
        return reveal(self.margin_marks, run_time)


# --------------------------------------------------------------------------------------------------------------------
# Nyquist plot
# --------------------------------------------------------------------------------------------------------------------
class NyquistPlot(VGroup):
    """Intuition: a leash round a post. The post is the point -1; the curve is the open loop's answer as the pitch rises.

    If the curve winds round -1 the feedback loop reinforces itself and runs away. The count ``N = Z - P`` is read from the
    pole locations, so it stays right for integrators. The mirror image (dashed teal) is the response at negative frequency.
    """

    def __init__(self, poles=(), zeros=(), gain: float = 1.0, *, size: float = 4.2, w_range: tuple[float, float] | None = None):
        super().__init__()
        self.tf = tf_from_pz(poles, zeros, gain)
        self.nyq = nyquist_data(self.tf, w_range=w_range)
        z = self.nyq.points[np.isfinite(self.nyq.points)]
        wins = [2.0, 1.3 * float(np.percentile(np.abs(z), 60)) if len(z) else 2.0]
        bd = bode_data(self.tf, w_range=w_range)
        if bd.w_pc is not None:  # make room for where the curve crosses the real axis
            wins.append(1.6 * float(abs(self.tf(1j * bd.w_pc))))
        self.extent = min(6.0, ceil_to(max(wins), 0.5))
        e = self.extent
        self.axes = plain_axes([-e, e, e], [-e, e, e], size, size)
        self.frame = frame_for(self.axes, size, size)
        unit = self.axes.c2p(1, 0)[0] - self.axes.c2p(0, 0)[0]
        self.unit_circle = DashedVMobject(Circle(radius=unit, stroke_color=PALETTE.muted, stroke_width=2, fill_opacity=0).move_to(self.axes.c2p(0, 0)),
                                          num_dashes=36) if unit * 1.0 < size / 2 * 1.05 else VGroup()
        self.curve = VGroup(*[polyline([self.axes.c2p(p.real, p.imag) for p in run], color=PALETTE.blue, width=4)
                              for run in runs_inside(self.nyq.points, e, e)])
        self.mirror = VGroup(*[DashedVMobject(polyline([self.axes.c2p(p.real, -p.imag) for p in run], color=PALETTE.teal, width=3), num_dashes=40)
                               for run in runs_inside(self.nyq.points, e, e)])
        hide(self.mirror)
        self.critical = pole_marker(self.axes.c2p(-1, 0), color=PALETTE.red, size=0.13)
        self.critical_tag = label("-1", 24, PALETTE.red).next_to(self.critical, DOWN, buff=0.12)
        n = self.nyq.clockwise_encirclements
        self.verdict = label(f"encirclements N = Z - P = {self.nyq.closed_loop_rhp} - {self.nyq.open_loop_rhp} = {n}", 22, PALETTE.fg)
        self.verdict.next_to(self.frame, DOWN, buff=0.2)
        self.status = label("closed loop stable" if self.nyq.stable else "closed loop unstable", 24,
                            PALETTE.green if self.nyq.stable else PALETTE.red).next_to(self.verdict, DOWN, buff=0.12)
        self.axis_names = VGroup(label("Re", 20, PALETTE.muted).next_to(self.axes.c2p(e, 0), UP, buff=0.1),
                                 label("Im", 20, PALETTE.muted).next_to(self.axes.c2p(0, e), RIGHT, buff=0.1))
        self.add(self.axes, self.frame, self.unit_circle, self.mirror, self.curve, self.critical, self.critical_tag, self.axis_names, self.verdict,
                 self.status)

    @property
    def data(self):
        return self.nyq

    def animate_trace(self, *, run_time: float = 4.0):
        """The curve is drawn from low to high frequency; ``animate_mirror`` then reveals the negative-frequency half."""
        segs = list(self.curve)
        if not segs:
            raise ValueError("no part of the Nyquist curve lies inside the plot window")
        return Succession(*[Create(s, run_time=run_time / len(segs), rate_func=linear) for s in segs])

    def animate_mirror(self, *, run_time: float = 1.2):
        return reveal(self.mirror, run_time)


# --------------------------------------------------------------------------------------------------------------------
# Root locus
# --------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LocusInfo:
    k_max: float
    critical_gain: float | None
    open_loop_poles: tuple[complex, ...]
    zeros: tuple[complex, ...]
    branches: int


class RootLocus(VGroup):
    """Intuition: turn the gain knob and watch every closed-loop pole walk. Each pole starts on an open-loop pole (cross) and
    ends on a zero (circle) or runs off to infinity; the loop is stable only while all of them are left of the yellow line.

    ``num`` and ``den`` are the open-loop numerator and denominator, highest power first (``den`` has more terms).
    """

    def __init__(self, num, den, k_max: float = 10.0, *, size: float = 4.4):
        super().__init__()
        self._size = size
        self.num, self.den = np.atleast_1d(np.asarray(num, float)), np.atleast_1d(np.asarray(den, float))
        self.locus = root_locus_data(self.num, self.den, k_max=k_max)
        self.k_max = k_max
        self.k = ValueTracker(0.0)
        self.ol_poles = [complex(p) for p in np.roots(self.den)]
        self.ol_zeros = [complex(z) for z in np.roots(self.num)] if len(self.num) > 1 else []
        marks = np.array(self.ol_poles + self.ol_zeros, dtype=complex)
        spread = max(float(np.max(np.abs(marks))) if len(marks) else 1.0, 1.0)
        reach = float(np.percentile(np.abs(self.locus.roots), 95)) if self.locus.roots.size else spread
        e = max(1.0, ceil_to(1.2 * min(max(reach, spread), 3 * spread), 0.5))
        self.extent = e
        self.axes = plain_axes([-e, e, e], [-e, e, e], size, size)
        self.frame = frame_for(self.axes, size, size)
        self.unstable_zone = Rectangle(width=size / 2, height=size, stroke_width=0, fill_color=PALETTE.red, fill_opacity=0.08) \
            .move_to(self.axes.c2p(e / 2, 0))
        self.boundary = Line(self.axes.c2p(0, -e), self.axes.c2p(0, e), color=PALETTE.yellow, stroke_width=3)
        self.zone_tag = label("unstable side", 18, PALETTE.red).move_to(self.axes.c2p(e / 2, -0.88 * e))
        self.branches = VGroup(*[polyline([self.axes.c2p(p.real, p.imag) for p in run], color=PALETTE.teal, width=3.5)
                                 for col in range(self.locus.roots.shape[1]) for run in runs_inside(self.locus.roots[:, col], e, e)])
        self.pole_marks = VGroup(*[pole_marker(self.axes.c2p(p.real, p.imag)) for p in self.ol_poles])
        self.zero_marks = VGroup(*[zero_marker(self.axes.c2p(z.real, z.imag)) for z in self.ol_zeros])
        self.dots = VGroup(*[Dot(radius=0.1, color=PALETTE.blue) for _ in range(self.locus.roots.shape[1])])
        self.k_readout = LiveValue("K", self.k, decimals=1, color=PALETTE.yellow)
        self.status = label("", 22, PALETTE.green)
        self.add(self.unstable_zone, self.axes, self.frame, self.boundary, self.zone_tag, self.branches, self.zero_marks, self.pole_marks,
                 self.dots, self.k_readout, self.status)
        self.k_readout.next_to(self.frame, DOWN, buff=0.2).align_to(self.frame, LEFT)
        self._sync()
        self.add_updater(lambda m: m._sync())

    @classmethod
    def from_pz(cls, poles=(), zeros=(), gain: float = 1.0, k_max: float = 10.0, **kw) -> "RootLocus":
        tf = tf_from_pz(poles, zeros, gain)
        return cls(tf.num, tf.den, k_max, **kw)

    def poles_at(self, k: float) -> np.ndarray:
        """Closed-loop poles at gain ``k`` (interpolated along the pre-computed branches)."""
        kk, r = self.locus.k, self.locus.roots
        return np.array([np.interp(k, kk, r[:, j].real) + 1j * np.interp(k, kk, r[:, j].imag) for j in range(r.shape[1])])

    def _sync(self) -> None:
        k = float(np.clip(self.k.get_value(), 0.0, self.k_max))
        poles = self.poles_at(k)
        for d, p in zip(self.dots, poles):
            d.move_to(self.axes.c2p(float(np.clip(p.real, -self.extent, self.extent)), float(np.clip(p.imag, -self.extent, self.extent))))
        ok = bool(np.all(poles.real < -ROOT_TOL))
        word = match_scale(label("stable" if ok else "unstable", 24, PALETTE.green if ok else PALETTE.red), self.frame, self._size)
        self.status.become(word.next_to(self.frame, DOWN, buff=0.2).align_to(self.frame, RIGHT))

    @property
    def data(self) -> LocusInfo:
        return LocusInfo(self.k_max, self.locus.critical_gain, tuple(self.ol_poles), tuple(self.ol_zeros), self.locus.roots.shape[1])

    def bind(self, tracker: ValueTracker, param: str = "k") -> "RootLocus":
        if param != "k":
            raise ValueError("RootLocus can only bind 'k'")
        self.k = tracker
        self.k_readout._get = tracker.get_value
        self._sync()
        return self

    def animate_k(self, k_from: float = 0.0, k_to: float | None = None, *, run_time: float = 5.0):
        """Turn the gain knob from ``k_from`` to ``k_to`` (default: ``k_max``); the blue dots are the closed-loop poles."""
        self.k.set_value(k_from)
        return self.k.animate.set_value(self.k_max if k_to is None else k_to).build().set_run_time(run_time).set_rate_func(linear)
