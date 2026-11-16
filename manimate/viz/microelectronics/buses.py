"""Serial buses and waveforms (README section 7.3): SPI, I2C, UART, differential pairs, USB, CAN."""
from __future__ import annotations

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, AnimationGroup, Dot, FadeIn, Indicate, Line, Rectangle, Succession, Transform, VGroup, VMobject)

from ...ctx import label
from ..core.palette import PALETTE
from ._math import (ArbitrationResult, BusTimeline, CANFrame, DifferentialData, NRZIResult, can_arbitration, can_frame, differential,
                    i2c_frame, level_at, nrzi_encode, spi_frame, uart_frame)


class DigitalWaveform(VGroup):
    """Intuition: a light switch plotted against time. High is 1, low is 0, and every sharp edge is the line changing.

    Pass ``bits`` for a simple NRZ (level per bit) or NRZI (a 0 toggles the line) trace, or build one from step-function
    breakpoints with ``from_points`` (levels may be fractions, e.g. 0.5 for CAN's recessive state). Time is in bit periods;
    ``x_at(t)`` and ``y_at(level)`` map them to the screen so marks can be placed exactly.
    """

    def __init__(self, bits=None, name: str = "", bit_time: float = 1.0, style: str = "nrz", *, width: float | None = None,
                 height: float = 0.6, color: str = PALETTE.blue, show_bits: bool = True, points=None, duration: float | None = None,
                 grid: bool = True):
        super().__init__()
        if bits is not None:
            bits = list(bits)
            if not bits or any(b not in (0, 1) for b in bits):
                raise ValueError("bits must be a non-empty list of 0 and 1")
            if style not in ("nrz", "nrzi"):
                raise ValueError("style must be 'nrz' or 'nrzi'")
            level, pts = 1, []
            for i, b in enumerate(bits):
                if style == "nrzi":
                    if b == 0:
                        level ^= 1
                    lv = level
                else:
                    lv = b
                pts.append((i * bit_time, float(lv)))
            points, duration = pts, len(bits) * bit_time
        elif points is None or duration is None:
            raise ValueError("give either bits, or both points and duration")
        self.breaks, self.duration, self.bit_time, self.h = list(points), float(duration), bit_time, height
        self.w = width if width is not None else min(9.0, max(2.0, 0.9 * self.duration / bit_time))
        self.line_color = color
        corners = []
        for k, (t, lv) in enumerate(self.breaks):
            if corners:
                corners.append(np.array([self.x_at(t), corners[-1][1], 0.0]))
            corners.append(np.array([self.x_at(t), self.y_at(lv), 0.0]))
        corners.append(np.array([self.x_at(self.duration), corners[-1][1], 0.0]))
        self.trace = VMobject(stroke_color=color, stroke_width=4)
        self.trace.set_points_as_corners(corners)
        parts = [self.trace]
        if grid and bits is not None:
            for i in range(len(bits) + 1):
                x = self.x_at(i * bit_time)
                parts.append(Line((x, -0.1, 0), (x, height + 0.1, 0), color=PALETTE.muted, stroke_width=1, stroke_opacity=0.4))
        if show_bits and bits is not None and len(bits) <= 16:
            for i, b in enumerate(bits):
                parts.append(label(str(b), 16, PALETTE.muted).move_to((self.x_at((i + 0.5) * bit_time), height + 0.3, 0)))
        self.tag = label(name, 20, color).move_to((-0.35, height / 2, 0))
        self.tag.shift(LEFT * self.tag.width / 2)
        if name:
            parts.append(self.tag)
        self.add(*parts)

    @classmethod
    def from_points(cls, points, duration: float, name: str = "", **kw) -> "DigitalWaveform":
        return cls(None, name, points=points, duration=duration, **kw)

    def x_at(self, t: float) -> float:
        return self.w * t / self.duration

    def y_at(self, level: float) -> float:
        return self.h * level

    def level_at(self, t: float) -> float:
        return level_at(self.breaks, t)

    def mark(self, t: float, text: str = "", color: str = PALETTE.yellow) -> VGroup:
        """A dot on the trace at time ``t`` with an optional small caption above it (returned, not added)."""
        # positions are read from the current screen location of the trace's left end
        origin = self.trace.get_points()[0]
        p = origin + np.array([self.x_at(t), (self.y_at(self.level_at(t)) - self.y_at(self.breaks[0][1])), 0.0])
        out = VGroup(Dot(p, radius=0.07, color=color))
        if text:
            out.add(label(text, 16, color).next_to(p, UP, buff=0.12))
        return out


class ClockLine(DigitalWaveform):
    """Intuition: a metronome. Everything else on the bus moves to its beat; the sampled edge says when the receiver looks.

    ``cpol`` is the idle level. ``mark_sampling_edges`` puts a dot on every rising (or falling) edge.
    """

    def __init__(self, cycles: int = 8, cpol: int = 0, *, width: float | None = None, color: str = PALETTE.yellow):
        if not 1 <= cycles <= 32:
            raise ValueError("cycles must be between 1 and 32")
        if cpol not in (0, 1):
            raise ValueError("cpol must be 0 or 1")
        self.cycles, self.cpol = cycles, cpol
        pts = [(0.0, float(cpol))]
        for k in range(cycles):
            pts += [(k + 0.5, float(1 - cpol)), (k + 1.0, float(cpol))]
        super().__init__(None, "CLK", points=pts, duration=cycles + 0.5, width=width, color=color, show_bits=False, grid=False)

    def edges(self, edge: str = "rising") -> list[float]:
        """Times of the rising or falling edges."""
        if edge not in ("rising", "falling"):
            raise ValueError("edge must be 'rising' or 'falling'")
        want = 1.0 if edge == "rising" else 0.0
        return [t for (t, lv), (_, prev) in zip(self.breaks[1:], self.breaks[:-1]) if lv == want and prev != want]

    def mark_sampling_edges(self, edge: str = "rising") -> VGroup:
        """A dot on every ``edge`` edge: these are the instants at which a receiver samples its input."""
        return VGroup(*[self.mark(t) for t in self.edges(edge)])


class _BusDiagram(VGroup):
    """Shared layout for protocol pictures: stacked waveforms on one time axis, a labelled field row, and hidden sample marks."""

    ROW_GAP = 1.3

    def _build(self, timeline: BusTimeline, order: list[str], colors: dict[str, str], *, fields_under: str | None = None) -> None:
        self.timeline = timeline
        self.w = min(9.4, 0.9 * timeline.duration)
        self.waves: dict[str, DigitalWaveform] = {}
        parts = []
        for i, name in enumerate(order):
            wave = DigitalWaveform.from_points(timeline.signals[name], timeline.duration, name, width=self.w, color=colors.get(name, PALETTE.blue),
                                               show_bits=False, grid=False)
            wave.shift(DOWN * i * self.ROW_GAP)
            self.waves[name] = wave
            parts.append(wave)
        self.field_row = VGroup()
        if fields_under:
            y = -(len(order) - 1) * self.ROW_GAP - 0.55
            for k, (t0, t1, text) in enumerate(timeline.fields):
                x0, x1 = self.waves[order[0]].x_at(t0), self.waves[order[0]].x_at(t1)
                narrow = (x1 - x0) < 0.55
                yy = y - (0.32 if narrow and k % 2 else 0.0)
                self.field_row.add(Line((x0 + 0.03, y + 0.18, 0), (x1 - 0.03, y + 0.18, 0), color=PALETTE.muted, stroke_width=2),
                                   label(text, 14 if narrow else 16, PALETTE.fg).move_to(((x0 + x1) / 2, yy - 0.05, 0)))
        self.add(*parts, self.field_row)
        self.move_to((0, 0, 0))
        self.sample_marks: list[VGroup] = []
        for t, bit, sig in timeline.samples:
            m = self.waves[sig].mark(t, str(bit))
            m.set_opacity(0.0)
            self.sample_marks.append(m)
        self.add(*self.sample_marks)

    @property
    def data(self) -> BusTimeline:
        return self.timeline

    def animate_sampling(self, *, run_time: float = 4.0):
        """Reveal each sampling instant in time order: a dot on the line and the bit that the receiver reads."""
        per = run_time / max(1, len(self.sample_marks))
        order = sorted(range(len(self.sample_marks)), key=lambda i: self.timeline.samples[i][0])
        return Succession(*[self.sample_marks[i].animate.set_opacity(1.0).build().set_run_time(per) for i in order])


class SPIBus(_BusDiagram):
    """Intuition: a shared metronome (SCLK) and a data line. The mode (CPOL, CPHA) decides which clock edge the receiver looks at.

    CPOL is the clock's idle level; CPHA = 0 samples on the first (leading) edge, CPHA = 1 on the second (trailing) edge.
    Yellow dots mark the sampling instants and the shift register below fills one bit per sample.
    """

    COLORS = {"SCLK": PALETTE.yellow, "MOSI": PALETTE.blue, "MISO": PALETTE.teal, "CS": PALETTE.muted}

    def __init__(self, mosi, miso=None, cpol: int = 0, cpha: int = 0):
        super().__init__()
        data = [mosi] if isinstance(mosi, int) else list(mosi)
        if len(data) > 2:
            raise ValueError("the picture shows at most two bytes")
        self.cpol, self.cpha = cpol, cpha
        timeline = spi_frame(data, cpol, cpha, 8, miso)
        order = ["SCLK", "MOSI"] + (["MISO"] if "MISO" in timeline.signals else []) + ["CS"]
        self._build(timeline, order, self.COLORS)
        mode = (cpol << 1) | cpha
        self.mode_label = label(f"SPI mode {mode}: CPOL={cpol}, CPHA={cpha}", 22, PALETTE.fg).next_to(self, UP, buff=0.35)
        bits = timeline.bits
        self.register = VGroup(*[Rectangle(width=0.42, height=0.42, stroke_color=PALETTE.blue, stroke_width=2) for _ in bits]).arrange(RIGHT, buff=0.06)
        self.register.next_to(self, DOWN, buff=0.45)
        self.reg_bits = [label(str(b), 20, PALETTE.fg).move_to(r).set_opacity(0.0) for r, b in zip(self.register, bits)]
        self.reg_tag = label("MOSI shift register", 18, PALETTE.muted).next_to(self.register, LEFT, buff=0.25)
        self.add(self.mode_label, self.register, *self.reg_bits, self.reg_tag)

    def animate_sampling(self, *, run_time: float = 5.0):
        """Walk through the sampling edges, filling the shift register from the MOSI line."""
        mosi_idx = [i for i, s in enumerate(self.timeline.samples) if s[2] == "MOSI"]
        order = sorted(range(len(self.sample_marks)), key=lambda i: self.timeline.samples[i][0])
        per = run_time / max(1, len(mosi_idx))
        steps, group = [], []
        pos = {i: k for k, i in enumerate(mosi_idx)}
        for i in order:
            group.append(self.sample_marks[i].animate.set_opacity(1.0))
            if i in pos:
                group.append(self.reg_bits[pos[i]].animate.set_opacity(1.0))
                steps.append(AnimationGroup(*group, run_time=per))
                group = []
        if group:
            steps.append(AnimationGroup(*group, run_time=per))
        return Succession(*steps)


class I2CBus(_BusDiagram):
    """Intuition: a polite two-wire conversation. SDA may only change while SCL is low, except for START and STOP, which are deliberate violations.

    A frame is START, the 7-bit address, the R/W bit, an ACK pulled low by the slave, data bytes each followed by an
    ACK, and STOP. Fields are labelled under the SDA line; yellow dots show the bit read on each SCL pulse.
    """

    COLORS = {"SCL": PALETTE.yellow, "SDA": PALETTE.blue}

    def __init__(self, addr: int = 0x48, data=0xA5, rw: str = "w"):
        super().__init__()
        self._build(i2c_frame(addr, data, rw), ["SCL", "SDA"], self.COLORS, fields_under="SDA")
        self.addr, self.rw = addr, rw


class UARTFrame(_BusDiagram):
    """Intuition: Morse code with no shared clock. Both ends agree on the bit time, a falling edge (start bit) says "now", and each bit is read mid-bit.

    The line idles high. Data goes out least significant bit first, then an optional parity bit, then the stop bit(s) back at high.
    """

    def __init__(self, byte: int = 0x41, parity: str = "none", stop: int = 1):
        super().__init__()
        self.byte, self.parity = byte, parity
        self._build(uart_frame(byte, parity, stop), ["TX"], {"TX": PALETTE.blue}, fields_under="TX")


class DifferentialPair(VGroup):
    """Intuition: two wires that carry the same noise. The receiver subtracts them, so the noise cancels and the signal (the difference) survives.

    Four stacked traces: D+, D-, their difference, and their average (the common mode). ``animate_noise`` adds the same
    noise to both wires: D+, D- and the common mode get ragged, the difference stays clean.
    """

    PANEL = 1.0
    ROW_GAP = 1.45

    def __init__(self, bits=(1, 0, 1, 1, 0, 0, 1, 0), noise: float = 0.0, seed: int = 7, *, width: float = 8.0):
        super().__init__()
        bits = list(bits)
        if not 1 <= len(bits) <= 16:
            raise ValueError("give between 1 and 16 bits")
        self.noise_amp = noise if noise > 0 else 0.35
        self.clean: DifferentialData = differential(bits, 0.0, seed)
        self.noisy: DifferentialData = differential(bits, self.noise_amp, seed)
        self.bits, self.width_units = bits, width
        spec = [("D+", "dplus", PALETTE.blue, (-0.4, 2.4)), ("D-", "dminus", PALETTE.orange, (-0.4, 2.4)),
                ("D+ - D-", "diff", PALETTE.green, (-1.4, 1.4)), ("common mode", "common", PALETTE.purple, (-0.4, 2.4))]
        self.traces: list[VMobject] = []
        self._targets: list[VMobject] = []
        parts = []
        for i, (name, attr, col, rng) in enumerate(spec):
            y0 = -i * self.ROW_GAP
            self.traces.append(self._trace(getattr(self.clean, attr), rng, y0, col))
            self._targets.append(self._trace(getattr(self.noisy, attr), rng, y0, col))
            parts += [self.traces[-1], label(name, 20, col).move_to((-0.3 - 0.5, y0 + self.PANEL / 2, 0)).shift(LEFT * 0.4),
                      Line((0, y0, 0), (width, y0, 0), color=PALETTE.muted, stroke_width=1, stroke_opacity=0.4)]
        self.add(*parts)
        before = self.get_center().copy()
        self.move_to((0, 0, 0))
        shift = self.get_center() - before  # the noisy targets were built before the move: carry them along
        for tg in self._targets:
            tg.shift(shift)

    def _trace(self, values: np.ndarray, rng: tuple[float, float], y0: float, color: str) -> VMobject:
        n = len(values)
        xs = np.linspace(0, self.width_units, n)
        lo, hi = rng
        ys = y0 + (np.clip(values, lo, hi) - lo) / (hi - lo) * self.PANEL
        tr = VMobject(stroke_color=color, stroke_width=3.5)
        tr.set_points_as_corners([np.array([x, y, 0.0]) for x, y in zip(xs, ys)])
        return tr

    @property
    def data(self) -> dict:
        return {"clean": self.clean, "noisy": self.noisy, "noise": self.noise_amp,
                "residual": float(np.max(np.abs(self.noisy.diff - self.clean.diff)))}

    def animate_noise(self, *, run_time: float = 3.0):
        """Inject common-mode noise: every trace but the difference visibly changes."""
        return AnimationGroup(*[Transform(tr, tg) for tr, tg in zip(self.traces, self._targets)], run_time=run_time)


class USBLine(VGroup):
    """Intuition: a train that must keep changing. USB reads a 0 as a toggle and a 1 as "stay", so after six 1s in a row a 0 is stuffed in to force an edge.

    Top row: the bits sent (red boxes are stuffed zeros). Below: the D+ and D- lines (always opposite) after NRZI encoding.
    """

    def __init__(self, bits=(1, 1, 1, 1, 1, 1, 1, 0, 1), stuffing: bool = True):
        super().__init__()
        bits = list(bits)
        if not 1 <= len(bits) <= 20:
            raise ValueError("give between 1 and 20 bits")
        self.result: NRZIResult = nrzi_encode(bits, stuffing)
        n = len(self.result.data)
        w = min(9.0, 0.62 * n)
        pts_p = [(float(i), float(lv)) for i, lv in enumerate(self.result.levels)]
        pts_m = [(float(i), 1.0 - lv) for i, lv in pts_p]
        self.d_plus = DigitalWaveform.from_points(pts_p, float(n), "D+", width=w, color=PALETTE.blue, show_bits=False, grid=True)
        self.d_minus = DigitalWaveform.from_points(pts_m, float(n), "D-", width=w, color=PALETTE.orange, show_bits=False, grid=False)
        self.d_minus.shift(DOWN * 1.3)
        cw = w / n
        self.bit_boxes = VGroup()
        for i, b in enumerate(self.result.data):
            stuffed = i in self.result.stuffed
            col = PALETTE.red if stuffed else PALETTE.muted
            box = Rectangle(width=cw - 0.04, height=0.45, stroke_color=col, stroke_width=2, fill_color=col, fill_opacity=0.35 if stuffed else 0.1)
            box.move_to(((i + 0.5) * cw, 1.15, 0))
            self.bit_boxes.add(VGroup(box, label(str(b), 20, PALETTE.fg).move_to(box)))
        self.caption = label("bits sent (red = stuffed 0)" if self.result.stuffed else "bits sent", 18, PALETTE.muted).next_to(self.bit_boxes, UP, buff=0.15)
        self.add(self.bit_boxes, self.caption, self.d_plus, self.d_minus)
        self.move_to((0, 0, 0))

    @property
    def data(self) -> NRZIResult:
        return self.result

    def animate_stuffing(self, *, run_time: float = 2.0):
        """Flash each stuffed bit in turn so the extra zeros are easy to spot."""
        if not self.result.stuffed:
            raise ValueError("no bits were stuffed in this stream")
        per = run_time / len(self.result.stuffed)
        return Succession(*[Indicate(self.bit_boxes[i], color=PALETTE.red, scale_factor=1.3, run_time=per) for i in self.result.stuffed])


def _baseline(wave: DigitalWaveform) -> float:
    """Screen y of a waveform's level 0 (read from the trace so it stays right after the group is moved)."""
    return float(wave.trace.get_points()[0][1] - wave.y_at(wave.breaks[0][1]))


def _left(wave: DigitalWaveform) -> float:
    return float(wave.trace.get_points()[0][0])


class CANBus(_BusDiagram):
    """Intuition: two wires pushed apart to say 0 (dominant) and relaxed together to say 1 (recessive); a dominant bit always wins on the shared bus.

    Shows CAN_H and CAN_L for a data frame. Stuffed bits (an opposite bit inserted after five equal ones) are marked with a
    red S. By default the first 30 bits are shown (SOF, identifier, control bits, the start of the data field).
    """

    def __init__(self, frame_bits=None, *, window: tuple[int, int] | None = None):
        super().__init__()
        if frame_bits is None:
            frame_bits = can_frame(0x123, [0xDE, 0xAD])
        if isinstance(frame_bits, CANFrame):
            self.frame = frame_bits
            bits, fields, stuffed = list(frame_bits.bits), list(frame_bits.fields), set(frame_bits.stuffed)
        else:
            self.frame = None
            bits, fields, stuffed = list(frame_bits), [], set()
        if not bits or any(b not in (0, 1) for b in bits):
            raise ValueError("frame_bits must be a CANFrame or a non-empty list of 0 and 1")
        a, b = window if window is not None else (0, min(len(bits), 30))
        if not (0 <= a < b <= len(bits)):
            raise ValueError("window must satisfy 0 <= start < end <= number of bits")
        self.window = (a, b)
        n = b - a
        h = [(float(i), 1.0 if bits[a + i] == 0 else 0.5) for i in range(n)]
        l = [(float(i), 0.0 if bits[a + i] == 0 else 0.5) for i in range(n)]
        shown = [(max(f0, a) - a, min(f1, b) - a, name) for f0, f1, name in fields if f1 > a and f0 < b]
        timeline = BusTimeline({"CAN_H": h, "CAN_L": l}, float(n), [(float(x0), float(x1), nm) for x0, x1, nm in shown], [], bits[a:b])
        self._build(timeline, ["CAN_H", "CAN_L"], {"CAN_H": PALETTE.orange, "CAN_L": PALETTE.blue}, fields_under="CAN_L")
        self.stuff_marks = VGroup()
        wave = self.waves["CAN_L"]
        for i in range(n):
            if (a + i) in stuffed:
                self.stuff_marks.add(label("S", 18, PALETTE.red).move_to((_left(wave) + wave.x_at(i + 0.5), _baseline(wave) - 0.3, 0)))
        self.add(self.stuff_marks)
        top = _baseline(self.waves["CAN_H"]) + wave.h + 0.2
        bottom = _baseline(wave) - 0.15
        self.cursor = Rectangle(width=wave.w / n, height=top - bottom, stroke_color=PALETTE.yellow, stroke_width=3, fill_opacity=0.0)
        self.cursor.move_to((_left(wave) + wave.w / n / 2, (top + bottom) / 2, 0)).set_stroke(opacity=0.0)
        self.add(self.cursor)
        self._n = n

    def animate_scan(self, *, run_time: float = 5.0):
        """Sweep a cursor along the frame from the first bit to the last."""
        wave = self.waves["CAN_L"]
        end_x = _left(wave) + wave.w * (self._n - 0.5) / self._n
        start_x = _left(wave) + wave.w * 0.5 / self._n
        self.cursor.move_to((start_x, self.cursor.get_center()[1], 0))
        return self.cursor.animate(rate_func=lambda t: t).set_stroke(opacity=1.0).move_to((end_x, self.cursor.get_center()[1], 0)).build().set_run_time(run_time)

    def animate_stuffing(self, *, run_time: float = 2.0):
        if len(self.stuff_marks) == 0:
            raise ValueError("no stuffed bits in the shown window")
        per = run_time / len(self.stuff_marks)
        return Succession(*[Indicate(m, color=PALETTE.red, scale_factor=1.6, run_time=per) for m in self.stuff_marks])


class CANArbitration(VGroup):
    """Intuition: everyone speaks at once, listening as they talk; a node that says 1 and hears 0 knows someone more urgent is talking and goes quiet.

    ``nodes`` are identifiers (or ``(name, id)`` pairs). The bottom row is the bus, the AND of all senders. The lowest
    identifier never hears anything unexpected, so it wins without losing a single bit.
    """

    ROW = 1.25

    def __init__(self, nodes):
        super().__init__()
        items = []
        for k, n in enumerate(nodes):
            name, ident = (n if isinstance(n, tuple) else (f"Node {chr(65 + k)}", n))
            items.append((name, int(ident)))
        self.names = [n for n, _ in items]
        self.result: ArbitrationResult = can_arbitration([i for _, i in items])
        w = 8.6
        colors = [PALETTE.blue, PALETTE.teal, PALETTE.purple, PALETTE.orange]
        self.rows = []
        parts = []
        for r, (name, ident) in enumerate(items):
            wave = DigitalWaveform(list(self.result.bits[r]), f"{name} 0x{ident:03X}", width=w, color=colors[r % 4], show_bits=False)
            wave.shift(DOWN * r * self.ROW)
            self.rows.append(wave)
            parts.append(wave)
        self.bus_row = DigitalWaveform(list(self.result.bus), "bus", width=w, color=PALETTE.yellow, show_bits=False)
        self.bus_row.shift(DOWN * (len(items) * self.ROW + 0.1))
        parts.append(self.bus_row)
        self.add(*parts)
        self.move_to((0, 0, 0))
        left = _left(self.rows[0])
        self._col_w = w / 11
        top = _baseline(self.rows[0]) + self.rows[0].h + 0.25
        bottom = _baseline(self.bus_row) - 0.15
        self.cursor = Rectangle(width=self._col_w, height=top - bottom, stroke_color=PALETTE.yellow, stroke_width=3, fill_opacity=0.0)
        self.cursor.move_to((left + self._col_w / 2, (top + bottom) / 2, 0)).set_stroke(opacity=0.0)
        self.lost_tags = []
        for r, k in enumerate(self.result.lost_at):
            if k is None:
                self.lost_tags.append(None)
                continue
            t = label("lost", 18, PALETTE.red).move_to((left + self._col_w * (k + 0.5), _baseline(self.rows[r]) + self.rows[r].h + 0.28, 0))
            t.set_opacity(0.0)
            self.lost_tags.append(t)
        win = self.result.winner
        self.winner_tag = label(f"{self.names[win]} wins arbitration", 22, PALETTE.green).next_to(self.bus_row, DOWN, buff=0.3)
        self.winner_tag.set_opacity(0.0)
        self.add(self.cursor, *[t for t in self.lost_tags if t is not None], self.winner_tag)
        self._left = left

    @property
    def data(self) -> ArbitrationResult:
        return self.result

    def animate_arbitration(self, *, run_time: float = 6.0):
        """Scan the 11 identifier bits left to right; each loser dims and is tagged "lost" at the bit where it dropped out."""
        per = run_time / 12
        steps = []
        y = self.cursor.get_center()[1]
        for k in range(11):
            parts = [self.cursor.animate.move_to((self._left + self._col_w * (k + 0.5), y, 0)).set_stroke(opacity=1.0)]
            for r, lost in enumerate(self.result.lost_at):
                if lost == k:
                    parts += [self.rows[r].trace.animate.set_stroke(opacity=0.25), self.lost_tags[r].animate.set_opacity(1.0)]
            steps.append(AnimationGroup(*parts, run_time=per))
        steps.append(AnimationGroup(self.cursor.animate.set_stroke(opacity=0.0), self.winner_tag.animate.set_opacity(1.0), run_time=per))
        return Succession(*steps)
