"""DC-DC converters, PWM and inverters (README section 9.3)."""
from __future__ import annotations

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, AnimationGroup, Arrow, Axes, Create, Dot, Group, Line, Rectangle, Text, UpdateFromAlphaFunc,
                   ValueTracker, VGroup, VMobject, linear)

from ..core.numeric import rng
from ..core.palette import PALETTE
from . import _math as M
from ._widgets import Scope, capacitor, diode, inductor, resistor, source, switch, wire

ON_COLOR, OFF_COLOR = PALETTE.yellow, PALETTE.teal


class ConverterTopology(VGroup):
    """Intuition: a valve that chops energy into packets. Light shows where the current flows while the switch is on, then off."""

    def __init__(self, kind: str = "buck", vin: float = 12.0, duty: float = 0.5, L: float = 100e-6, C: float = 100e-6,
                 R: float = 5.0, fsw: float = 100e3):
        super().__init__()
        M.conversion_ratio(kind, duty)  # validates
        self.kind, self.vin, self._duty, self.L, self.C, self.R, self.fsw = kind, vin, duty, L, C, R, fsw
        self._tracker: ValueTracker | None = None
        self.on = True
        self.parts: dict[str, VGroup | Line] = {}
        self._paths: dict[str, list[str]] = {}
        self._build()
        self.readout = Text("", font_size=22, color=PALETTE.fg)
        self.state_label = Text("", font_size=22, color=PALETTE.yellow)
        self.add(*self.parts.values())
        self.add(self.state_label, self.readout)
        self._ref0 = self.parts["w_bot"].width
        self.sw_label = Text("S", font_size=18, color=PALETTE.fg)
        self.add(self.sw_label)
        self._apply_state(True)
        self.add_updater(lambda m: m._refresh())

    def _k(self) -> float:
        """How much the whole schematic has been scaled, so freshly made text can match it."""
        return self.parts["w_bot"].width / self._ref0

    # -- layout -------------------------------------------------------------
    def _build(self) -> None:
        P, k = self.parts, self.kind
        TL, BL, TR, BR = (-2.4, 1.0, 0), (-2.4, -1.0, 0), (2.2, 1.0, 0), (2.2, -1.0, 0)
        P["w_bot"] = wire(BL, BR)
        if k == "buck":
            P["vin"] = VGroup(wire((-2.4, 1, 0), (-2.4, 0.3, 0)), wire((-2.4, -0.3, 0), (-2.4, -1, 0)), source((-2.4, 0, 0), label="Vin"))
            P["w_in"] = wire((-2.4, 1, 0), (-1.8, 1, 0))
            self._sw_ends = ((-1.8, 1, 0), (-0.9, 1, 0))
            P["sw"] = switch(*self._sw_ends, True)
            P["w_sw"] = wire((-0.9, 1, 0), (-0.5, 1, 0))
            P["L"] = inductor((-0.5, 1, 0), (0.7, 1, 0))
            P["w_L"] = wire((0.7, 1, 0), (1.2, 1, 0))
            P["w_out"] = wire((1.2, 1, 0), (2.2, 1, 0))
            P["D"] = diode((-0.5, -1, 0), (-0.5, 1, 0))
            self._paths = {"on": ["vin", "w_in", "sw", "w_sw", "L", "w_L", "w_out", "C", "R", "w_bot"],
                           "off": ["D", "w_sw", "L", "w_L", "w_out", "C", "R", "w_bot"]}
            cap_x, res_x = 1.2, 2.2
        elif k == "boost":
            P["vin"] = VGroup(wire((-2.4, 1, 0), (-2.4, 0.3, 0)), wire((-2.4, -0.3, 0), (-2.4, -1, 0)), source((-2.4, 0, 0), label="Vin"))
            P["L"] = inductor((-2.4, 1, 0), (-0.7, 1, 0))
            P["w_sw"] = wire((-0.7, 1, 0), (-0.2, 1, 0))
            self._sw_ends = ((-0.2, 1, 0), (-0.2, -1, 0))
            P["sw"] = switch(*self._sw_ends, True)
            P["D"] = diode((-0.2, 1, 0), (1.3, 1, 0))
            P["w_out"] = wire((1.3, 1, 0), (2.2, 1, 0))
            self._paths = {"on": ["vin", "L", "w_sw", "sw", "w_bot"], "off": ["vin", "L", "w_sw", "D", "w_out", "C", "R", "w_bot"]}
            cap_x, res_x = 1.3, 2.2
        else:  # buck_boost
            P["vin"] = VGroup(wire((-2.4, 1, 0), (-2.4, 0.3, 0)), wire((-2.4, -0.3, 0), (-2.4, -1, 0)), source((-2.4, 0, 0), label="Vin"))
            P["w_in"] = wire((-2.4, 1, 0), (-1.8, 1, 0))
            self._sw_ends = ((-1.8, 1, 0), (-0.9, 1, 0))
            P["sw"] = switch(*self._sw_ends, True)
            P["w_sw"] = wire((-0.9, 1, 0), (-0.4, 1, 0))
            P["L"] = inductor((-0.4, 1, 0), (-0.4, -1, 0))
            P["D"] = diode((-0.4, 1, 0), (1.3, 1, 0))
            P["w_out"] = wire((1.3, 1, 0), (2.2, 1, 0))
            self._paths = {"on": ["vin", "w_in", "sw", "w_sw", "L", "w_bot"], "off": ["L", "w_sw", "D", "w_out", "C", "R", "w_bot"]}
            cap_x, res_x = 1.3, 2.2
        P["C"] = capacitor((cap_x, 1, 0), (cap_x, -1, 0))
        P["R"] = resistor((res_x, 1, 0), (res_x, -1, 0))

    # -- state --------------------------------------------------------------
    def bind(self, tracker: ValueTracker, name: str = "duty") -> "ConverterTopology":
        if name != "duty":
            raise ValueError("only 'duty' can be bound")
        self._tracker = tracker
        return self

    @property
    def duty(self) -> float:
        d = self._tracker.get_value() if self._tracker is not None else self._duty
        return float(min(max(d, 0.0), 0.95))

    def waveforms(self, periods: int = 3) -> M.ConverterWaveforms:
        return M.converter_waveforms(self.kind, self.vin, self.duty, self.L, self.C, self.R, self.fsw, periods)

    def _apply_state(self, on: bool) -> None:
        self.on = on
        new_sw = switch(*self._sw_ends, on)
        self.parts["sw"].become(new_sw)
        active = set(self._paths["on" if on else "off"])
        colour = ON_COLOR if on else OFF_COLOR
        for name, m in self.parts.items():
            m.set_color(colour if name in active else PALETTE.muted)
        k = self._k()
        self.state_label.become(Text("switch ON" if on else "switch OFF", font_size=22, color=colour).scale(k).next_to(self.parts["w_bot"], DOWN, buff=0.25 * k))
        a, b = (np.array(e, dtype=float) for e in self._sw_ends)
        self.sw_label.become(Text("S", font_size=18, color=PALETTE.fg).scale(k))
        self._place_switch_label()

    def _place_switch_label(self) -> None:
        """Put 'S' beside the switch, whatever the schematic's scale or position."""
        k = self._k()
        a, b = (np.array(e, dtype=float) for e in self._sw_ends)
        base = self.parts["w_bot"]
        origin = base.get_start() - np.array([-2.4, -1.0, 0]) * k  # schematic origin in scene coordinates
        mid = origin + (a + b) / 2 * k
        off = np.array([0, 0.45, 0]) if a[1] == b[1] else np.array([0.45, 0, 0])
        self.sw_label.move_to(mid + off * k)

    def _refresh(self) -> None:
        k = self._k()
        vout = self.vin * M.conversion_ratio(self.kind, self.duty)
        self.readout.become(Text(f"D = {self.duty:.2f}   Vout = {vout:.1f} V", font_size=22, color=PALETTE.fg).scale(k).next_to(self.state_label, DOWN, buff=0.15 * k))

    def animate_switching(self, periods: int = 3, run_time: float | None = None):
        """Flip the switch ``periods`` times; the active loop lights in the colour of the current state."""
        if periods < 1:
            raise ValueError("periods must be at least 1")

        def step(m, a):
            on = (a * periods) % 1.0 < m.duty
            if on != m.on:
                m._apply_state(on)

        return UpdateFromAlphaFunc(self, step, run_time=run_time or 1.0 * periods, rate_func=linear)


class InductorRamp(VGroup):
    """Intuition: the inductor current is a ramp up while the switch is on and a ramp down while it is off; the output barely ripples."""

    def __init__(self, converter: ConverterTopology, width: float = 3.6):
        super().__init__()
        self.converter, self._key, self._width0 = converter, None, width
        self.v_sw = Scope("switch node (V)", width, 0.9, PALETTE.blue)
        self.i_l = Scope("inductor current (A)", width, 1.3, PALETTE.yellow)
        self.v_out = Scope("output voltage (V)", width, 0.9, PALETTE.green)
        self.note_on = Text("", font_size=16, color=ON_COLOR)
        self.note_off = Text("", font_size=16, color=OFF_COLOR)
        self.add(VGroup(self.v_sw, self.i_l, self.v_out).arrange(DOWN, buff=0.35), self.note_on, self.note_off)
        self._refresh()
        self.add_updater(lambda m: m._refresh())

    _SLOPES = {"buck": ("+(Vin - Vout)/L", "-Vout/L"), "boost": ("+Vin/L", "+(Vin - Vout)/L"), "buck_boost": ("+Vin/L", "-Vout/L")}

    def _refresh(self) -> None:
        c = self.converter
        key = (round(c.duty, 3), c.kind)
        if key == self._key:
            return
        self._key = key
        w = c.waveforms(3)
        t = w.t * c.fsw
        self.v_sw.set_data(t, w.v_sw, (min(0.0, w.v_sw.min()), max(w.v_sw.max(), 1e-6) * 1.15))
        lo, hi = float(w.i_l.min()), float(w.i_l.max())
        pad = max(hi - lo, 1e-6) * 0.6
        self.i_l.set_data(t, w.i_l, (lo - pad, hi + pad))
        self.v_out.set_data(t, w.v_out, (float(w.v_out.min()) - 0.5 * abs(w.v_out.mean()), float(w.v_out.max()) + 0.5 * abs(w.v_out.mean())))
        on_txt, off_txt = self._SLOPES[c.kind]
        d = c.duty
        k = self.i_l.frame.width / self._width0
        fr = self.i_l.frame
        self.note_on.become(Text(f"on: di/dt = {on_txt}", font_size=15, color=ON_COLOR).scale(k).move_to(fr.get_corner(UP + LEFT) + np.array([0.55, -0.14, 0]) * 0.0).align_to(fr, LEFT).shift(RIGHT * 0.1 * k + DOWN * 0.14 * k).align_to(fr, UP).shift(DOWN * 0.06 * k))
        self.note_off.become(Text(f"off: di/dt = {off_txt}", font_size=15, color=OFF_COLOR).scale(k).align_to(fr, RIGHT).shift(LEFT * 0.1 * k).align_to(fr, DOWN).shift(UP * 0.06 * k))


class ConversionRatioCurve(VGroup):
    """Intuition: Vout/Vin against duty cycle: turn the duty knob and the operating point slides along the curve."""

    def __init__(self, kind: str = "buck", tracker: ValueTracker | None = None, width: float = 3.6, height: float = 2.4):
        super().__init__()
        self.kind, self._width0 = kind, width
        self.tracker = tracker if tracker is not None else ValueTracker(0.5)
        ymax = {"buck": 1.1, "boost": 6.0, "buck_boost": 0.0}[kind]
        ymin = -6.0 if kind == "buck_boost" else 0.0
        self.axes = Axes(x_range=[0, 1, 0.25], y_range=[ymin, ymax, (ymax - ymin) / 4], x_length=width, y_length=height,
                         axis_config={"color": PALETTE.muted, "include_numbers": False, "stroke_width": 2}, tips=False)
        curve = self.axes.plot(lambda d: float(np.clip(M.conversion_ratio(kind, min(d, 0.83)), ymin, ymax)), x_range=[0.0, 0.83, 0.01], color=PALETTE.blue)
        self.dot = Dot(radius=0.09, color=PALETTE.yellow)
        self.text = Text("", font_size=20, color=PALETTE.yellow)
        self.add(self.axes, curve, Text("duty D", font_size=16, color=PALETTE.muted).next_to(self.axes, DOWN, buff=0.1),
                 Text("Vout/Vin", font_size=16, color=PALETTE.muted).next_to(self.axes, UP, buff=0.1).align_to(self.axes, LEFT), self.dot, self.text)
        self._refresh()
        self.add_updater(lambda m: m._refresh())

    def _refresh(self) -> None:
        d = min(max(self.tracker.get_value(), 0.0), 0.83)
        r = float(np.clip(M.conversion_ratio(self.kind, d), -6.0, 6.0))
        self.dot.move_to(self.axes.c2p(d, r))
        k = self.axes.width / self._width0
        self.text.become(Text(f"D={d:.2f}  Vout/Vin={r:.2f}", font_size=18, color=PALETTE.yellow).scale(k).next_to(self.axes, RIGHT, buff=0.25 * k))


class PWMGenerator(VGroup):
    """Intuition: a slow wave and a fast ramp race; whenever the slow one is higher the output is on."""

    def __init__(self, reference: str = "sine", carrier: str = "triangle", ma: float = 0.8, ratio: int = 15,
                 width: float = 6.0, height: float = 2.6):
        super().__init__()
        if reference not in ("sine", "dc") or carrier != "triangle":
            raise ValueError("reference is 'sine' or 'dc'; carrier is 'triangle'")
        n = ratio * 48
        t = np.arange(n) / n
        ref = ma * np.sin(2 * np.pi * t) if reference == "sine" else np.full(n, ma)
        car = M.triangle(t, ratio)
        pulses = M.pwm_compare(ref, car)
        self.top = Scope("reference vs carrier", width, height * 0.6, PALETTE.blue)
        self.top.set_data(t, ref, (-1.1, 1.1))
        self.carrier = VMobject(color=PALETTE.muted, stroke_width=1.5).set_points_as_corners([self.top.to_point(a, b) for a, b in zip(t, car)])
        self.bottom = Scope("gate pulses", width, height * 0.3, PALETTE.yellow)
        self.bottom.set_data(t, pulses, (-0.2, 1.2))
        self.bottom.next_to(self.top, DOWN, buff=0.35)
        cross = np.where(np.diff(np.sign(ref - car)) != 0)[0]
        self.crossings = VGroup(*(Dot(self.top.to_point(t[i], ref[i]), radius=0.035, color=PALETTE.red) for i in cross[:: max(1, len(cross) // 40)]))
        self.add(self.top, self.carrier, self.crossings, self.bottom)
        self.data = (t, ref, car, pulses)

    def animate_build(self, run_time: float = 3.0):
        return AnimationGroup(Create(self.top.trace), Create(self.carrier), Create(self.bottom.trace), lag_ratio=0.5, run_time=run_time)


class HBridge(VGroup):
    """Intuition: four taps in an H. Open the diagonal pair and current pushes through the load one way; swap pairs and it reverses."""

    def __init__(self, modulation: str = "bipolar", ma: float = 0.8, load: str = "RL", ratio: int = 15):
        super().__init__()
        self.states = M.hbridge_states(modulation, ma, ratio)
        self.modulation, self.load = modulation, load
        pts = {"TL": (-1.2, 1.5, 0), "A": (-1.2, 0, 0), "BL": (-1.2, -1.5, 0), "TR": (1.2, 1.5, 0), "B": (1.2, 0, 0), "BR": (1.2, -1.5, 0)}
        self.pts = {k: np.array(v, dtype=float) for k, v in pts.items()}
        self.base = VGroup(wire(pts["TL"], pts["TR"]), wire(pts["BL"], pts["BR"]), wire(pts["TL"], pts["A"]), wire(pts["A"], pts["BL"]),
                           wire(pts["TR"], pts["B"]), wire(pts["B"], pts["BR"]), wire((-1.2, 0, 0), (-0.55, 0, 0)), wire((0.55, 0, 0), (1.2, 0, 0)))
        self.load_sym = resistor((-0.55, 0, 0), (0.55, 0, 0), color=PALETTE.fg)
        self.boxes: dict[str, Rectangle] = {}
        self.labels = VGroup()
        for name, (x, y) in {"Q1": (-1.2, 0.75), "Q2": (-1.2, -0.75), "Q3": (1.2, 0.75), "Q4": (1.2, -0.75)}.items():
            box = Rectangle(width=0.55, height=0.55, stroke_color=PALETTE.muted, stroke_width=3, fill_color=PALETTE.bg, fill_opacity=1).move_to((x, y, 0))
            self.boxes[name] = box
            self.labels.add(Text(name, font_size=16, color=PALETTE.fg).move_to(box))
        self.path = VMobject(color=PALETTE.yellow, stroke_width=6)
        self.arrow = Arrow((-0.4, 0.28, 0), (0.4, 0.28, 0), buff=0, color=PALETTE.teal, stroke_width=4, max_tip_length_to_length_ratio=0.4)
        self.readout = Text("", font_size=22, color=PALETTE.fg)
        self.add(self.base, self.load_sym, self.path, *self.boxes.values(), self.labels, self.arrow, self.readout)
        self.phase = 0.0
        self.set_phase(0.0)

    def set_phase(self, frac: float) -> "HBridge":
        s = self.states
        i = int(min(max(frac, 0.0), 0.999999) * len(s.t))
        on = {"Q1": s.q1[i], "Q2": s.q2[i], "Q3": s.q3[i], "Q4": s.q4[i]}
        for n, b in self.boxes.items():
            b.set_fill(PALETTE.yellow if on[n] else PALETTE.bg, opacity=1.0 if on[n] else 1.0).set_stroke(PALETTE.yellow if on[n] else PALETTE.muted)
        p = self.pts
        route = {(1, 0, 0, 1): ["TL", "A", "B", "BR"], (0, 1, 1, 0): ["TR", "B", "A", "BL"], (1, 0, 1, 0): ["TL", "A", "B", "TR"], (0, 1, 0, 1): ["BL", "A", "B", "BR"]}
        key = (on["Q1"], on["Q2"], on["Q3"], on["Q4"])
        pts = [p[n] for n in route.get(key, [])]
        self.path.set_points_as_corners(pts) if len(pts) > 1 else self.path.clear_points()
        direction = 1 if s.i_dir[i] > 0 else -1
        self.arrow.put_start_and_end_on(np.array([-0.4 * direction, 0.28, 0]), np.array([0.4 * direction, 0.28, 0]))
        v = s.v_out[i]
        k = self.base[0].width / 2.4
        self.readout.become(Text(f"Vout = {'+Vdc' if v > 0 else '-Vdc' if v < 0 else '0'}   load current {'right' if direction > 0 else 'left'}",
                                 font_size=22, color=PALETTE.fg).scale(k).next_to(self.base, DOWN, buff=0.3 * k))
        self.phase = frac
        return self

    def animate_cycle(self, run_time: float = 6.0):
        """Sweep one fundamental period: the lit pair changes with every carrier crossing."""
        return UpdateFromAlphaFunc(self, lambda m, a: m.set_phase(a), run_time=run_time, rate_func=linear)


class ThreePhaseInverter(VGroup):
    """Intuition: three reference waves 120 degrees apart each race one carrier, building three chopped pole voltages."""

    def __init__(self, vdc: float = 400.0, ma: float = 0.9, ratio: int = 21, width: float = 4.6):
        super().__init__()
        self.data = M.spwm_three_phase(vdc, ma, ratio, periods=1)
        d = self.data
        self.ref_scope = Scope("references vs carrier", width, 1.3, PALETTE.muted)
        self.ref_scope.set_data(d.t, d.refs[0], (-1.1, 1.1))
        cols = [PALETTE.red, PALETTE.green, PALETTE.blue]
        self.refs = VGroup(*(VMobject(color=cols[k], stroke_width=2).set_points_as_corners([self.ref_scope.to_point(a, b) for a, b in zip(d.t, d.refs[k])]) for k in range(3)))
        self.carrier = VMobject(color=PALETTE.muted, stroke_width=1).set_points_as_corners([self.ref_scope.to_point(a, b) for a, b in zip(d.t, d.carrier)])
        self.pole_scopes = [Scope(f"pole voltage {n}", width, 0.55, cols[k]).set_data(d.t, d.poles[k], (-vdc / 2 * 1.2, vdc / 2 * 1.2)) for k, n in enumerate("abc")]
        stack = VGroup(self.ref_scope, *self.pole_scopes).arrange(DOWN, buff=0.28)
        self.bridge = self._bridge()
        VGroup(self.bridge, stack).arrange(RIGHT, buff=0.6)
        self.add(self.bridge, stack, self.refs, self.carrier)
        self.cols = cols

    def _bridge(self) -> VGroup:
        g = VGroup(wire((-1.4, 1.5, 0), (1.4, 1.5, 0)), wire((-1.4, -1.5, 0), (1.4, -1.5, 0)), Text("DC bus", font_size=16, color=PALETTE.muted).move_to((-1.9, 0, 0)))
        for k, x in enumerate((-0.9, 0.0, 0.9)):
            g.add(wire((x, 1.5, 0), (x, 0.9, 0)), wire((x, -1.5, 0), (x, -0.9, 0)), wire((x, 0.4, 0), (x, -0.4, 0)))
            for y in (0.65, -0.65):
                g.add(Rectangle(width=0.4, height=0.5, stroke_color=PALETTE.muted, stroke_width=3).move_to((x, y, 0)))
            g.add(Text("abc"[k], font_size=18, color=PALETTE.fg).move_to((x, 0, 0)))
        return g

    def animate_build(self, run_time: float = 5.0):
        return AnimationGroup(Create(self.carrier), *(Create(r) for r in self.refs), *(Create(s.trace) for s in self.pole_scopes), lag_ratio=0.25, run_time=run_time)


def PWMToSine(inverter: ThreePhaseInverter, filter_hz: float = 500.0, f1: float = 50.0, run_time: float = 5.0):
    """Intuition: a window slides across the chopped pulses and averages them: out comes a clean sine."""
    d = inverter.data
    scope = inverter.pole_scopes[0]
    n = d.poles.shape[1]
    win = int(np.clip(round(n * f1 / filter_hz), 3, n // 2))
    kernel = np.ones(win) / win
    wrapped = np.concatenate([d.poles[0][-win:], d.poles[0], d.poles[0][:win]])
    filt = np.convolve(wrapped, kernel, mode="same")[win:-win]
    pts = [scope.to_point(a, b) for a, b in zip(d.t, filt)]
    out = VMobject(color=PALETTE.yellow, stroke_width=4)
    window = Rectangle(width=scope.w * win / n, height=scope.h, stroke_width=0, fill_color=PALETTE.yellow, fill_opacity=0.2)
    inverter.add(out, window)
    out.set_points_as_corners(pts[:2])
    window.move_to(scope.to_point(d.t[0], 0.0))

    def step(m, a):
        k = max(2, int(a * n))
        out.set_points_as_corners(pts[:k])
        window.move_to(np.array([pts[k - 1][0], scope.frame.get_center()[1], 0.0]))

    return UpdateFromAlphaFunc(inverter, step, run_time=run_time, rate_func=linear)
