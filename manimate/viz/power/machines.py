"""Machines, drives and stability pictures (README section 9.4)."""
from __future__ import annotations

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, AnimationGroup, Arc, Arrow, Axes, Circle, Create, DashedLine, Dot, Line, Polygon, Rectangle,
                   Succession, Text, UpdateFromAlphaFunc, ValueTracker, VGroup, VMobject, linear)

from ... import theme
from ..core.diagram import Block, Wire
from ..core.flow import pulse_along
from ..core.palette import PALETTE
from . import _math as M
from ._widgets import Scope

PHASES = list(theme.PHASE_COLORS)  # a, b, c
D_COLOR, Q_COLOR, FIELD_COLOR = PALETTE.teal, PALETTE.purple, PALETTE.green  # never the phase colours


def _arrow(v, color, width=4.0, origin=(0.0, 0.0, 0.0)) -> Arrow:
    o = np.array(origin, dtype=float)
    end = o + np.array([v[0], v[1], 0.0])
    if np.linalg.norm(end - o) < 1e-3:
        end = o + np.array([1e-3, 0.0, 0.0])
    return Arrow(o, end, buff=0, color=color, stroke_width=width, max_tip_length_to_length_ratio=0.3, max_stroke_width_to_length_ratio=12)


class Stator3Phase(VGroup):
    """Intuition: a ring of slots; three coloured windings take turns magnetising it, which is what makes the field rotate."""

    def __init__(self, poles: int = 2, slots: int = 12, radius: float = 2.0):
        super().__init__()
        if slots % 6 != 0 or poles < 2:
            raise ValueError("slots must be a multiple of 6 (three phases, two sides) and poles >= 2")
        self.slots, self.poles = slots, poles
        self.add(Circle(radius=radius, color=PALETTE.muted, stroke_width=3), Circle(radius=radius * 0.72, color=PALETTE.muted, stroke_width=3))
        pattern = [(0, 1), (2, -1), (1, 1), (0, -1), (2, 1), (1, -1)]  # (phase, direction): a, -c, b, -a, c, -b
        self.coils = VGroup()
        for k in range(slots):
            ph, sign = pattern[k % 6]
            ang = 2 * np.pi * k / slots
            c = radius * 0.86 * np.array([np.cos(ang), np.sin(ang), 0.0])
            dot = Dot(c, radius=0.12, color=PHASES[ph]) if sign > 0 else Circle(radius=0.12, color=PHASES[ph], stroke_width=4).move_to(c)
            self.coils.add(dot)
        self.add(self.coils)


class RotatingMagneticField(VGroup):
    """Intuition: three pulsating arrows on fixed axes add up to one arrow of constant length that spins."""

    def __init__(self, freq: float = 50.0, poles: int = 2, show_phases: bool = True, scale: float = 1.4):
        super().__init__()
        self.freq, self.poles, self.scale_, self.show_phases = freq, poles, scale, show_phases
        self.axes_lines = VGroup(*(DashedLine(-2.0 * np.array([np.cos(a), np.sin(a), 0.0]) * 0.0, 2.0 * np.array([np.cos(a), np.sin(a), 0.0]), color=PHASES[k], stroke_width=1.5, stroke_opacity=0.5)
                                   for k, a in enumerate(2 * np.pi * np.arange(3) / 3)))
        self.phase_arrows = VGroup(*(_arrow((0, 0), PHASES[k], 3.0) for k in range(3)))
        self.resultant = _arrow((1, 0), FIELD_COLOR, 7.0)
        self.add(self.axes_lines, self.phase_arrows, self.resultant)
        self.t = 0.0
        self.set_time(0.0)

    def set_time(self, t: float) -> "RotatingMagneticField":
        vecs, res = M.rmf_vectors(t, self.freq)
        k = self.scale_
        for i in range(3):
            self.phase_arrows[i].become(_arrow(vecs[i] * k, PHASES[i], 3.0)).set_opacity(1.0 if self.show_phases else 0.0)
        self.resultant.become(_arrow(res * k / 1.5 * 1.3, FIELD_COLOR, 7.0))
        self.t = t
        return self

    def animate_rotate(self, turns: float = 1.0, run_time: float = 4.0):
        """Spin the resultant ``turns`` times (time is slowed down so you can watch it)."""
        return UpdateFromAlphaFunc(self, lambda m, a: m.set_time(a * turns / m.freq), run_time=run_time, rate_func=linear)


class RotorFrame(VGroup):
    """Intuition: the spinning part, with an arrow marking where its magnetic flux points (the d axis)."""

    def __init__(self, kind: str = "pm", radius: float = 1.0):
        super().__init__()
        if kind not in ("pm", "salient", "induction"):
            raise ValueError("kind must be 'pm', 'salient' or 'induction'")
        self.kind, self.radius, self.angle = kind, radius, 0.0
        body = VGroup(Circle(radius=radius, color=PALETTE.muted, stroke_width=3))
        if kind == "pm":
            body.add(Arc(radius=radius, start_angle=-np.pi / 2, angle=np.pi, color=PALETTE.red, stroke_width=10), Arc(radius=radius, start_angle=np.pi / 2, angle=np.pi, color=PALETTE.blue, stroke_width=10),
                     Text("N", font_size=24, color=PALETTE.red).move_to((radius * 0.5, 0, 0)), Text("S", font_size=24, color=PALETTE.blue).move_to((-radius * 0.5, 0, 0)))
        elif kind == "salient":
            for s in (1, -1):
                body.add(Rectangle(width=radius * 0.5, height=radius * 0.9, stroke_color=PALETTE.fg, stroke_width=3, fill_color=PALETTE.panel if hasattr(PALETTE, "panel") else PALETTE.bg, fill_opacity=1).move_to((s * radius * 0.72, 0, 0)))
        else:
            for k in range(10):
                a = 2 * np.pi * k / 10
                body.add(Dot(radius * 0.78 * np.array([np.cos(a), np.sin(a), 0.0]), radius=0.07, color=PALETTE.teal))
        self.body = body
        self.flux = _arrow((radius * 1.5, 0), PALETTE.yellow, 5.0)
        self.add(body, self.flux)

    def set_angle(self, theta: float) -> "RotorFrame":
        self.rotate(theta - self.angle, about_point=self.body[0].get_center())
        self.angle = theta
        return self


class DQFrame(VGroup):
    """Intuition: a camera riding on the rotor. Seen from there, the spinning currents hold still."""

    def __init__(self, theta_tracker: ValueTracker, show_abc: bool = True, show_alpha_beta: bool = True, radius: float = 1.8):
        super().__init__()
        self.theta_tracker, self.radius, self._angle = theta_tracker, radius, 0.0
        if show_abc:
            for k, a in enumerate(2 * np.pi * np.arange(3) / 3):
                u = np.array([np.cos(a), np.sin(a), 0.0])
                self.add(Line(0 * u, radius * u, color=PHASES[k], stroke_width=2, stroke_opacity=0.6), Text("abc"[k], font_size=18, color=PHASES[k]).move_to(radius * 1.12 * u))
        if show_alpha_beta:
            self.add(Line((-radius, 0, 0), (radius, 0, 0), color=PALETTE.muted, stroke_width=1.5), Line((0, -radius, 0), (0, radius, 0), color=PALETTE.muted, stroke_width=1.5),
                     Text("alpha", font_size=16, color=PALETTE.muted).move_to((radius * 1.18, -0.18, 0)), Text("beta", font_size=16, color=PALETTE.muted).move_to((0.3, radius * 1.1, 0)))
        self.dq = VGroup(_arrow((radius * 1.1, 0), D_COLOR, 5.0), _arrow((0, radius * 1.1), Q_COLOR, 5.0),
                         Text("d", font_size=22, color=D_COLOR).move_to((radius * 1.28, 0.18, 0)), Text("q", font_size=22, color=Q_COLOR).move_to((-0.2, radius * 1.28, 0)),
                         _arrow((radius * 0.8, 0), PALETTE.fg, 8.0))  # the flux vector, always on d
        self.add(self.dq)
        self._centre = np.zeros(3)
        self.add_updater(lambda m: m._refresh())
        self._refresh()

    def _refresh(self) -> None:
        th = self.theta_tracker.get_value()
        self.dq.rotate(th - self._angle, about_point=self.get_center_of_axes())
        self._angle = th

    def get_center_of_axes(self) -> np.ndarray:
        return self[0].get_start() if len(self) and isinstance(self[0], Line) else self.get_center()


class ClarkeParkTransform(VGroup):
    """Intuition: three wobbling waves become two (alpha, beta) and then two flat lines (d, q). Flat is easy to control."""

    def __init__(self, freq: float = 50.0, amp: float = 1.0, width: float = 4.6):
        super().__init__()
        t = np.linspace(0, 1 / freq, 240)
        th = 2 * np.pi * freq * t
        abc = np.stack([amp * np.cos(th - k * 2 * np.pi / 3) for k in range(3)])
        ab = M.clarke(abc)
        dq = M.park(ab, th)
        lim = (-1.3 * amp, 1.3 * amp)
        self.s_abc, self.s_ab, self.s_dq = Scope("a, b, c: three sinusoids", width, 1.1), Scope("alpha, beta: two sinusoids", width, 1.1), Scope("d, q: two constants", width, 1.1)
        for sc in (self.s_abc, self.s_ab, self.s_dq):
            sc.trace.set_stroke(opacity=0)
        self.traces = []
        for sc, series, cols in ((self.s_abc, abc, PHASES), (self.s_ab, ab, [PALETTE.green, PALETTE.fg]), (self.s_dq, dq, [D_COLOR, Q_COLOR])):
            sc.set_data(t, series[0], lim)
            for y, c in zip(series, cols):
                self.traces.append(VMobject(color=c, stroke_width=3).set_points_as_corners([sc.to_point(a, b) for a, b in zip(t, y)]))
        self.stack = VGroup(self.s_abc, self.s_ab, self.s_dq).arrange(DOWN, buff=0.3)
        # traces were built before arrange(): rebuild positions by re-mapping now
        self._remap(t, [abc, ab, dq], lim)
        self.add(self.stack, *self.traces)

    def _remap(self, t, series_list, lim) -> None:
        i = 0
        for sc, series in zip((self.s_abc, self.s_ab, self.s_dq), series_list):
            sc.set_data(t, series[0], lim)
            for y in series:
                self.traces[i].set_points_as_corners([sc.to_point(a, b) for a, b in zip(t, y)])
                i += 1
            sc.trace.set_stroke(opacity=0)

    def animate_steps(self, run_time: float = 6.0):
        groups = [AnimationGroup(*(Create(m) for m in self.traces[0:3])), AnimationGroup(*(Create(m) for m in self.traces[3:5])), AnimationGroup(*(Create(m) for m in self.traces[5:7]))]
        return Succession(*groups, run_time=run_time)


class TorqueDecomposition(VGroup):
    """Intuition: split the current into a part along the flux (i_d, makes no torque) and a part across it (i_q, makes all of it)."""

    def __init__(self, id: float = 0.0, iq: float = 1.0, size: float = 1.8):
        super().__init__()
        self.size = size
        self.axes_ = VGroup(Line((-size * 0.2, 0, 0), (size * 1.2, 0, 0), color=PALETTE.muted, stroke_width=2), Line((0, -size * 0.8, 0), (0, size * 1.2, 0), color=PALETTE.muted, stroke_width=2),
                            Text("d (flux)", font_size=18, color=D_COLOR).move_to((size * 1.1, -0.25, 0)), Text("q", font_size=18, color=Q_COLOR).move_to((-0.25, size * 1.15, 0)))
        self.vec = VGroup()
        self.readout = Text("", font_size=22, color=PALETTE.fg)
        self.add(self.axes_, self.vec, self.readout)
        self.set_currents(id, iq)

    def set_currents(self, id: float, iq: float) -> "TorqueDecomposition":
        s = self.size
        self.id_, self.iq_ = id, iq
        parts = VGroup(_arrow((id * s, iq * s), PALETTE.fg, 6.0), _arrow((id * s, 0), D_COLOR, 4.0), _arrow((0, iq * s), Q_COLOR, 4.0),
                       DashedLine((id * s, 0, 0), (id * s, iq * s, 0), color=PALETTE.muted, stroke_width=1.5), DashedLine((0, iq * s, 0), (id * s, iq * s, 0), color=PALETTE.muted, stroke_width=1.5))
        self.vec.become(parts)
        k = self.axes_[0].width / (1.4 * s)
        self.readout.become(Text(f"i_d = {id:.2f}  i_q = {iq:.2f}  torque ~ i_q = {M.foc_currents(id, iq, 0.0).torque:.2f}", font_size=17, color=PALETTE.fg).scale(k).next_to(self.axes_, DOWN, buff=0.4 * k))
        return self

    def animate_to(self, id: float, iq: float, run_time: float = 1.5):
        i0, q0 = self.id_, self.iq_
        return UpdateFromAlphaFunc(self, lambda m, a: m.set_currents(i0 + (id - i0) * a, q0 + (iq - q0) * a), run_time=run_time)


class FOCBlockDiagram(VGroup):
    """Intuition: field-oriented control as a loop: ask for speed, turn it into torque current, build voltages, spin the motor, measure, repeat."""

    def __init__(self):
        super().__init__()
        names = ["Speed PI", "Current PI (d,q)", "inv Park", "Modulator", "Inverter", "Motor"]
        self.blocks = [Block(n, w=1.7, h=0.8, size=17, color=PALETTE.blue if n != "Motor" else PALETTE.green) for n in names]
        for i, b in enumerate(self.blocks):
            b.move_to((-5.2 + 2.1 * i * 0.99, 0.9, 0))
        self.fb_park = Block("Clarke / Park", w=1.9, h=0.8, size=17, color=PALETTE.purple).move_to((-1.4, -1.2, 0))
        self.encoder = Block("Encoder", w=1.4, h=0.8, size=17, color=PALETTE.purple).move_to((3.4, -1.2, 0))
        self.chain_wires = [Wire(np.array([self.blocks[i].port("right"), self.blocks[i + 1].port("left")]), arrow=True, color=PALETTE.muted) for i in range(5)]
        fb_i = Wire.route(self.blocks[5].port("bottom") + np.array([-0.3, 0, 0]), self.fb_park.port("right"), via=[(self.blocks[5].port("bottom")[0] - 0.3, -1.2, 0)], arrow=True, color=PALETTE.purple)
        fb_to_pi = Wire.route(self.fb_park.port("left"), self.blocks[1].port("bottom"), via=[(self.blocks[1].port("bottom")[0], -1.2, 0)], arrow=True, color=PALETTE.purple)
        enc = Wire.route(self.blocks[5].port("bottom") + np.array([0.3, 0, 0]), self.encoder.port("top"), via=[(self.encoder.port("top")[0], self.blocks[5].port("bottom")[1] - 0.2, 0)], arrow=True, color=PALETTE.purple)
        self.feedback = [fb_i, fb_to_pi, enc]
        labels = VGroup(*(Text(t, font_size=14, color=PALETTE.muted).next_to(w, UP, buff=0.05) for w, t in zip(self.chain_wires, ["i_q*", "v_dq", "v_ab", "duty", "v_abc"])))
        self.add(*self.blocks, self.fb_park, self.encoder, *self.chain_wires, *self.feedback, labels)
        self.scale_to_fit_width(min(self.width, 12.0))

    def animate_signal(self, run_time: float = 4.0):
        """Send a pulse through the forward path, then back along the feedback path."""
        fwd = [pulse_along(w[0], PALETTE.yellow, run_time=run_time / 8) for w in self.chain_wires]
        back = [pulse_along(w[0], PALETTE.teal, run_time=run_time / 8) for w in self.feedback if len(w)]
        return Succession(*fwd, *back)


class DamperWindings(VGroup):
    """Intuition: copper bars in the pole faces. If the rotor drifts off synchronous speed they cut flux, current flows, and it brakes the drift."""

    def __init__(self, poles: int = 4, slip: float = 0.05, radius: float = 1.6):
        super().__init__()
        if poles < 2 or poles % 2:
            raise ValueError("poles must be an even number")
        self.poles, self.radius = poles, radius
        self.add(Circle(radius=radius * 0.45, color=PALETTE.muted, stroke_width=3))
        self.bars: list[Dot] = []
        for p in range(poles):
            a = 2 * np.pi * p / poles
            u = np.array([np.cos(a), np.sin(a), 0.0])
            shoe = Rectangle(width=radius * 0.8, height=radius * 0.22, stroke_color=PALETTE.fg, stroke_width=3).rotate(a + np.pi / 2).move_to(u * radius * 0.86)
            self.add(shoe, Line(u * radius * 0.45, u * radius * 0.75, color=PALETTE.muted, stroke_width=3))
            t = np.array([-u[1], u[0], 0.0])
            for j in (-1, 0, 1):
                bar = Dot(u * radius * 0.86 + t * j * radius * 0.25, radius=0.06, color=PALETTE.teal)
                self.bars.append(bar)
                self.add(bar)
        self.currents = VGroup()
        self.brake = VGroup()
        self.add(self.currents, self.brake)
        self.set_slip(slip)

    def set_slip(self, slip: float) -> "DamperWindings":
        self.slip = slip
        cur, torque = M.damper_torque(slip)
        k = self[0].width / (2 * self.radius * 0.45)
        self.currents.become(VGroup(*(Text("x" if (i // 3 + (1 if slip > 0 else 0)) % 2 == 0 else ".", font_size=int(10 + 26 * abs(float(cur))), color=PALETTE.yellow).scale(k).move_to(b)
                                    for i, b in enumerate(self.bars))) if abs(slip) > 1e-4 else VGroup())
        if abs(slip) > 1e-4:
            direction = -1 if slip > 0 else 1  # braking opposes the relative motion
            c = self[0].get_center()
            arc = Arc(radius=self.radius * 1.25 * k, start_angle=0.5, angle=direction * 1.4, color=PALETTE.red, stroke_width=3 + 5 * min(1.0, abs(float(torque))), arc_center=c)
            arc.add_tip(tip_length=0.2 * k)
            self.brake.become(VGroup(arc, Text("braking torque", font_size=16, color=PALETTE.red).scale(k).move_to(c + DOWN * self.radius * 1.5 * k)))
        else:
            self.brake.become(VGroup())
        return self

    def animate_slip(self, slip: float, run_time: float = 2.0):
        s0 = self.slip
        return UpdateFromAlphaFunc(self, lambda m, a: m.set_slip(s0 + (slip - s0) * a), run_time=run_time)


class RotorAngleOscillation(VGroup):
    """Intuition: a ball rolling about the bottom of a sinusoidal bowl: knock the rotor and it rocks around its equilibrium angle."""

    def __init__(self, H: float = 5.0, Pm: float = 0.8, Pmax: float = 1.8, D: float = 0.05, disturbance: float = 0.5, duration: float = 3.0, n: int = 240):
        super().__init__()
        self.t = np.linspace(0, duration, n)
        self.delta, _ = M.swing_simulate(H, Pm, Pmax, None, D, disturbance, self.t)
        self.Pm, self.Pmax = Pm, Pmax
        self.axes = Axes(x_range=[0, np.pi, np.pi / 4], y_range=[0, Pmax * 1.15, 0.5], x_length=3.6, y_length=2.4, tips=False,
                         axis_config={"color": PALETTE.muted, "include_numbers": False, "stroke_width": 2})
        curve = self.axes.plot(lambda d: M.power_angle_curve(Pmax, d), x_range=[0, np.pi, 0.02], color=PALETTE.blue)
        pm_line = DashedLine(self.axes.c2p(0, Pm), self.axes.c2p(np.pi, Pm), color=PALETTE.yellow, stroke_width=2)
        self.ball = Dot(radius=0.1, color=PALETTE.yellow)
        self.scope = Scope("rotor angle delta(t)", 3.6, 1.5, PALETTE.green).set_data(self.t, self.delta)
        self.scope.next_to(self.axes, RIGHT, buff=0.7)
        self.cursor = Dot(radius=0.07, color=PALETTE.yellow)
        self.labels = VGroup(Text("P = Pmax sin(delta)", font_size=16, color=PALETTE.muted).next_to(self.axes, UP, buff=0.1), Text("Pm", font_size=16, color=PALETTE.yellow).next_to(pm_line, LEFT, buff=0.05))
        self.add(self.axes, curve, pm_line, self.ball, self.scope, self.cursor, self.labels)
        self.set_index(0)

    def set_index(self, i: int) -> "RotorAngleOscillation":
        d = float(self.delta[i])
        self.ball.move_to(self.axes.c2p(d, M.power_angle_curve(self.Pmax, d)))
        self.cursor.move_to(self.scope.to_point(self.t[i], d))
        return self

    def animate_swing(self, run_time: float = 6.0):
        return UpdateFromAlphaFunc(self, lambda m, a: m.set_index(int(a * (len(m.t) - 1))), run_time=run_time, rate_func=linear)


class EqualAreaCriterion(VGroup):
    """Intuition: a fault builds up kinetic energy (area A_acc); once cleared the machine spends it again (area A_dec). Stable if the second area catches the first."""

    def __init__(self, Pm: float = 0.8, Pmax_pre: float = 1.8, Pmax_fault: float = 0.4, Pmax_post: float = 1.4, delta_clear: float = 0.9):
        super().__init__()
        self.ea = M.equal_area(Pm, Pmax_pre, Pmax_fault, Pmax_post, delta_clear)
        self.args = (Pm, Pmax_pre, Pmax_fault, Pmax_post)
        top = max(Pmax_pre, Pmax_post) * 1.15
        self.axes = Axes(x_range=[0, np.pi, np.pi / 4], y_range=[0, top, 0.5], x_length=5.0, y_length=3.0, tips=False,
                         axis_config={"color": PALETTE.muted, "include_numbers": False, "stroke_width": 2})
        sin = lambda P: self.axes.plot(lambda d: P * np.sin(d), x_range=[0, np.pi, 0.02], color=PALETTE.blue)
        pre, fault, post = sin(Pmax_pre), sin(Pmax_fault).set_color(PALETTE.red), sin(Pmax_post).set_color(PALETTE.teal)
        pm = DashedLine(self.axes.c2p(0, Pm), self.axes.c2p(np.pi, Pm), color=PALETTE.yellow, stroke_width=2)
        self.acc, self.dec = VMobject(), VMobject()
        self.readout = Text("", font_size=20, color=PALETTE.fg)
        self.curves = VGroup(pre, fault, post, pm)
        self.add(self.axes, self.curves, self.acc, self.dec, self.readout,
                 Text("pre-fault", font_size=14, color=PALETTE.blue).next_to(self.axes.c2p(np.pi / 2, Pmax_pre), UP, buff=0.05),
                 Text("fault", font_size=14, color=PALETTE.red).next_to(self.axes.c2p(np.pi / 2, Pmax_fault), DOWN, buff=0.05),
                 Text("post-fault", font_size=14, color=PALETTE.teal).next_to(self.axes.c2p(np.pi / 2 + 0.7, Pmax_post * np.sin(np.pi / 2 + 0.7)), RIGHT, buff=0.05))
        self.set_progress(0.0)

    def _area(self, d_from: float, d_to: float, upper, lower, color: str) -> VMobject:
        xs = np.linspace(d_from, max(d_to, d_from + 1e-6), 40)
        pts = [self.axes.c2p(x, upper(x)) for x in xs] + [self.axes.c2p(x, lower(x)) for x in xs[::-1]]
        return Polygon(*pts, stroke_width=0, fill_color=color, fill_opacity=0.55)

    def set_progress(self, a: float) -> "EqualAreaCriterion":
        """0..1 grows A_acc, 1..2 grows A_dec."""
        Pm, Ppre, Pf, Ppo = self.args
        e = self.ea
        a1, a2 = min(max(a, 0.0), 1.0), min(max(a - 1.0, 0.0), 1.0)
        self.acc.become(self._area(e.delta0, e.delta0 + (e.delta_clear - e.delta0) * a1, lambda x: Pm, lambda x: Pf * np.sin(x), PALETTE.red) if a1 > 0 else VMobject())
        self.dec.become(self._area(e.delta_clear, e.delta_clear + (e.delta_max - e.delta_clear) * a2, lambda x: Ppo * np.sin(x), lambda x: Pm, PALETTE.teal) if a2 > 0 else VMobject())
        shown = f"A_acc = {e.a_acc * min(a1, 1):.2f}   A_dec = {e.a_dec * a2:.2f}"
        verdict = ("  stable" if e.stable else "  UNSTABLE") if a >= 2.0 else ""
        k = self.axes.width / 5.0
        self.readout.become(Text(shown + verdict, font_size=20, color=PALETTE.fg if a < 2 else (PALETTE.green if e.stable else PALETTE.red)).scale(k).next_to(self.axes, DOWN, buff=0.3 * k))
        return self

    def animate_areas(self, run_time: float = 6.0):
        return UpdateFromAlphaFunc(self, lambda m, a: m.set_progress(2 * a), run_time=run_time, rate_func=linear)


class FaultCurrentTransient(VGroup):
    """Intuition: a short circuit starts huge and relaxes in three stages, riding on a decaying DC offset that tilts the wave."""

    def __init__(self, x_d2: float = 0.15, x_d1: float = 0.25, x_d: float = 1.2, dc_offset: bool = True, t_d2: float = 0.03, t_d1: float = 0.7, t_a: float = 0.15,
                 t_end: float = 0.5, width: float = 6.0, height: float = 2.6):
        super().__init__()
        t = np.linspace(0, t_end, 1500)
        self.data = M.fault_current(x_d2, x_d1, x_d, t_d2, t_d1, t_a, 0.0 if dc_offset else np.pi / 2, t)
        peak = float(max(np.abs(self.data.i).max(), self.data.envelope.max())) * 1.1
        self.scope = Scope("short-circuit current", width, height, PALETTE.fg).set_data(t, self.data.i, (-peak, peak))
        env_pts_hi = [self.scope.to_point(a, b) for a, b in zip(t, self.data.envelope)]
        env_pts_lo = [self.scope.to_point(a, -b) for a, b in zip(t, self.data.envelope)]
        self.env = VGroup(VMobject(color=PALETTE.yellow, stroke_width=2).set_points_as_corners(env_pts_hi), VMobject(color=PALETTE.yellow, stroke_width=2).set_points_as_corners(env_pts_lo))
        self.scope.trace.set_color(PALETTE.blue)
        b1, b2 = min(3 * t_d2, t_end * 0.35), min(t_end * 0.7, 0.6 * t_end + 0.0)
        self.regions = VGroup()
        for lo, hi, name, col in ((0.0, b1, "subtransient", PALETTE.red), (b1, b2, "transient", PALETTE.yellow), (b2, t_end, "steady state", PALETTE.green)):
            self.regions.add(Text(name, font_size=16, color=col).move_to(self.scope.to_point((lo + hi) / 2, peak * 0.95)))
            if lo > 0:
                self.regions.add(DashedLine(self.scope.to_point(lo, -peak), self.scope.to_point(lo, peak), color=PALETTE.muted, stroke_width=1))
        self.add(self.scope, self.env, self.regions)

    def animate_draw(self, run_time: float = 4.0):
        return AnimationGroup(Create(self.scope.trace), Create(self.env[0]), Create(self.env[1]), lag_ratio=0.0, run_time=run_time)
