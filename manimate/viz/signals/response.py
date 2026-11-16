"""Time-domain pictures (README section 8.2): the step/impulse response with its measurements, and the PID tuning demo."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, AnimationGroup, Create, DashedLine, DoubleArrow, Line, ValueTracker, VGroup, VMobject

from ...ctx import label
from ..core.live import Slider
from ..core.palette import PALETTE
from ._math import PLANTS, TF, ceil_to, pid_simulate, step_response, tf_from_coeffs
from ._parts import frame_for, hide, match_scale, plain_axes, polyline, reveal


class StepResponsePlot(VGroup):
    """Intuition: poke the system once and watch how it settles. Three numbers describe the whole shape: how fast it rises, how far it overshoots, how long it rings.

    The curve is drawn by ``animate_draw``; the yellow, teal and blue marks and the legend are revealed by ``animate_annotate``.
    With ``kind="impulse"`` (a sharp tap instead of a step) only the curve is shown.
    """

    def __init__(self, tf: TF, t_end: float | None = None, kind: str = "step", annotate: bool = True, *, width: float = 6.0, height: float = 3.2):
        super().__init__()
        self.tf = tf
        self.response = r = step_response(tf, None if t_end is None else np.linspace(0.0, t_end, 400), kind)
        t, y = r.t, np.clip(r.y, -1e3, 1e3)
        ref = r.final_value if (r.final_value and abs(r.final_value) > 1e-9) else max(1.0, float(np.max(np.abs(y))))
        lo = min(0.0, float(y.min()) * 1.1)
        hi = max(abs(ref) * 1.3, float(y.max()) * 1.1) if kind == "step" else max(float(y.max()) * 1.15, 0.1)
        self.axes = plain_axes([0, float(t[-1]), t[-1] / 4], [lo, hi, hi / 4], width, height)
        self.frame = frame_for(self.axes, width, height)
        self.curve = polyline([self.axes.c2p(a, b) for a, b in zip(t, np.clip(y, lo, hi))], color=PALETTE.blue, width=4)
        self.x_label = label("time (s)", 20, PALETTE.muted).next_to(self.frame, DOWN, buff=0.12)
        self.y_label = label("output", 20, PALETTE.muted).next_to(self.frame, UP, buff=0.1).align_to(self.frame, LEFT)
        self.annotations = VGroup()
        if annotate and kind == "step" and r.final_value is not None:
            self._annotate(r, ref, width)
        hide(self.annotations)
        name = "step response" if kind == "step" else "impulse response"
        title = label(name if r.stable else name + ": unstable, grows without bound", 22, PALETTE.fg if r.stable else PALETTE.red)
        title.next_to(self.frame, UP, buff=0.1).align_to(self.frame, RIGHT)
        self.add(self.axes, self.frame, self.curve, self.annotations, self.x_label, self.y_label, title)

    def _annotate(self, r, ref: float, width: float) -> None:
        ax, ann = self.axes, self.annotations
        x0, x1 = ax.c2p(0, 0)[0], ax.c2p(r.t[-1], 0)[0]
        ann.add(DashedLine(ax.c2p(0, ref), ax.c2p(r.t[-1], ref), color=PALETTE.muted, stroke_width=2))
        band = 0.02 * abs(ref)
        for s in (1, -1):
            ann.add(DashedLine(ax.c2p(0, ref + s * band), ax.c2p(r.t[-1], ref + s * band), color=PALETTE.teal, stroke_width=2))
        notes = [("final value " + f"{ref:.2f}", PALETTE.muted)]
        if r.rise_time is not None:
            i10, i90 = int(np.argmax(r.y >= 0.1 * ref)), int(np.argmax(r.y >= 0.9 * ref))
            base = ax.y_range[0]
            ann.add(DashedLine(ax.c2p(r.t[i10], base), ax.c2p(r.t[i10], 0.1 * ref), color=PALETTE.blue, stroke_width=2),
                    DashedLine(ax.c2p(r.t[i90], base), ax.c2p(r.t[i90], 0.9 * ref), color=PALETTE.blue, stroke_width=2),
                    DoubleArrow(ax.c2p(r.t[i10], base), ax.c2p(r.t[i90], base), buff=0, color=PALETTE.blue, stroke_width=4,
                                max_tip_length_to_length_ratio=0.3))
            notes.append((f"rise {r.rise_time:.2f} s", PALETTE.blue))
        if r.overshoot_pct is not None and r.overshoot_pct > 0.5:
            pk = ax.c2p(r.peak_time, float(np.max(r.y)))
            ann.add(DoubleArrow(ax.c2p(r.peak_time, ref), pk, buff=0, color=PALETTE.yellow, stroke_width=4, max_tip_length_to_length_ratio=0.35))
            notes.append((f"overshoot {r.overshoot_pct:.0f} %", PALETTE.yellow))
        if r.settling_time is not None:
            ann.add(DashedLine(ax.c2p(r.settling_time, 0), ax.c2p(r.settling_time, ref), color=PALETTE.teal, stroke_width=2.5))
            notes.append((f"settles {r.settling_time:.1f} s (2 % band)", PALETTE.teal))
        legend = VGroup(*[label(txt, 20, c) for txt, c in notes]).arrange(DOWN, buff=0.08, aligned_edge=LEFT)
        legend.next_to(self.frame, RIGHT, buff=0.25).align_to(self.frame, UP)
        ann.add(legend)

    @property
    def data(self):
        return self.response

    def animate_draw(self, *, run_time: float = 3.0):
        return Create(self.curve, run_time=run_time)

    def animate_annotate(self, *, run_time: float = 1.5):
        if not len(self.annotations):
            raise ValueError("nothing to annotate (impulse response, unstable system, or annotate=False)")
        return reveal(self.annotations, run_time)


@dataclass(frozen=True)
class PIDDemoData:
    kp: float
    ki: float
    kd: float
    stable: bool
    overshoot_pct: float | None
    settling_time: float | None
    steady_state_error: float


class PIDTuningDemo(VGroup):
    """Intuition: three dials on a thermostat. P pushes back in proportion to the error but leaves a gap; I keeps pushing until the gap closes; D brakes before it overshoots.

    Drive the sliders with ``animate_gains`` (or bind your own trackers) and the response is re-simulated every frame. The lower
    strip splits the control effort into its P, I and D parts; values beyond ``term_range`` are clipped at the frame.
    """

    def __init__(self, plant: TF | str = "second_order", kp: float = 1.0, ki: float = 0.0, kd: float = 0.0, setpoint: float = 1.0, *,
                 t_end: float = 12.0, term_range: float | None = None):
        super().__init__()
        self.plant = PLANTS[plant] if isinstance(plant, str) else plant
        self.setpoint, self.t_end = setpoint, t_end
        self.trackers = {"kp": ValueTracker(kp), "ki": ValueTracker(ki), "kd": ValueTracker(kd)}
        self._key, self._sim = None, None
        self.y_max = 2.0 * max(setpoint, 1e-3)
        self.u_max = term_range or 4.0 * max(1.0, setpoint)
        self.axes = plain_axes([0, t_end, t_end / 4], [-0.2 * self.y_max, self.y_max, self.y_max / 4], 6.0, 2.6)
        self.term_axes = plain_axes([0, t_end, t_end / 4], [-self.u_max, self.u_max, self.u_max / 2], 6.0, 1.5).next_to(self.axes, DOWN, buff=0.75)
        self.frames = VGroup(frame_for(self.axes, 6.0, 2.6), frame_for(self.term_axes, 6.0, 1.5))
        self.goal = DashedLine(self.axes.c2p(0, setpoint), self.axes.c2p(t_end, setpoint), color=PALETTE.muted, stroke_width=2.5)
        self.y_curve = VMobject(stroke_color=PALETTE.blue, stroke_width=4, fill_opacity=0)
        cols = (PALETTE.blue, PALETTE.teal, PALETTE.yellow)
        self.term_curves = [VMobject(stroke_color=c, stroke_width=3, fill_opacity=0) for c in cols]
        keys = VGroup(*[label(n, 20, c) for n, c in zip(("P", "I", "D"), cols)]).arrange(RIGHT, buff=0.3).next_to(self.frames[1], UP, buff=0.1).align_to(self.frames[1], RIGHT)
        self.tags = VGroup(label("goal (dashed) and output", 20, PALETTE.muted).next_to(self.frames[0], UP, buff=0.1).align_to(self.frames[0], LEFT),
                           label("control effort", 20, PALETTE.muted).next_to(self.frames[1], UP, buff=0.1).align_to(self.frames[1], LEFT), keys,
                           label("time (s)", 20, PALETTE.muted).next_to(self.frames[1], DOWN, buff=0.1))
        ranges = {"kp": (0, 10), "ki": (0, 10), "kd": (0, 5)}
        self.sliders = VGroup(*[Slider(self.trackers[k], ranges[k], k.upper().replace("K", "K"), length=2.4, decimals=1) for k in ("kp", "ki", "kd")])
        self.sliders.arrange(DOWN, buff=0.55).next_to(self.frames[0], RIGHT, buff=0.7).align_to(self.frames[0], UP)
        self.readout = label("", 22, PALETTE.fg)
        self.add(self.frames, self.axes, self.term_axes, self.goal, self.y_curve, *self.term_curves, self.tags, self.sliders, self.readout)
        self._sync()
        self.add_updater(lambda m: m._sync())

    def _gains(self) -> tuple[float, float, float]:
        return tuple(round(float(self.trackers[k].get_value()), 4) for k in ("kp", "ki", "kd"))  # type: ignore[return-value]

    def _simulate(self):
        key = self._gains()
        if key != self._key:
            self._sim, self._key = pid_simulate(*key, self.plant, self.setpoint, t_end=self.t_end, n=240), key
        return self._sim

    def _sync(self) -> None:
        d = self._simulate()
        step = 2
        for curve, ax, ys, lim in ((self.y_curve, self.axes, d.y, (-0.2 * self.y_max, self.y_max)),
                                   (self.term_curves[0], self.term_axes, d.p, (-self.u_max, self.u_max)),
                                   (self.term_curves[1], self.term_axes, d.i, (-self.u_max, self.u_max)),
                                   (self.term_curves[2], self.term_axes, d.d, (-self.u_max, self.u_max))):
            curve.set_points_as_corners([ax.c2p(t, float(np.clip(v, *lim))) for t, v in zip(d.t[::step], ys[::step])])
        os_ = "-" if d.overshoot_pct is None else f"{d.overshoot_pct:.0f} %"
        st = "does not settle" if d.settling_time is None else f"settles {d.settling_time:.1f} s"
        word = "stable" if d.stable else "UNSTABLE"
        text = label(f"{word}   overshoot {os_}   {st}   offset {d.steady_state_error:+.2f}", 20, PALETTE.green if d.stable else PALETTE.red)
        self.readout.become(match_scale(text, self.frames[0], 6.0).next_to(self.frames[1], DOWN, buff=0.55).align_to(self.frames[0], LEFT))

    @property
    def data(self) -> PIDDemoData:
        d, (kp, ki, kd) = self._simulate(), self._gains()
        return PIDDemoData(kp, ki, kd, d.stable, d.overshoot_pct, d.settling_time, d.steady_state_error)

    def bind(self, tracker: ValueTracker, param: str) -> "PIDTuningDemo":
        """Drive ``"kp"``, ``"ki"`` or ``"kd"`` from ``tracker``."""
        if param not in self.trackers:
            raise ValueError("param must be 'kp', 'ki' or 'kd'")
        self.trackers[param] = tracker
        self.sliders[("kp", "ki", "kd").index(param)].tracker = tracker
        self._sync()
        return self

    def animate_gains(self, kp: float | None = None, ki: float | None = None, kd: float | None = None, *, run_time: float = 3.0):
        """Turn the dials together; the response and effort curves morph continuously."""
        moves = [self.trackers[k].animate.set_value(v) for k, v in (("kp", kp), ("ki", ki), ("kd", kd)) if v is not None]
        if not moves:
            raise ValueError("give at least one of kp, ki, kd")
        return AnimationGroup(*moves, run_time=run_time)
