"""Block-diagram pictures for control (README section 8.2): transfer-function blocks, summing junctions, feedback loops, PID, state space."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, Circle, Dot, Line, Mobject, Rectangle, Succession, ValueTracker, VGroup, VMobject, linear

from ...ctx import label
from ..core.diagram import PORTS, Block, Wire
from ..core.flow import pulse_along
from ..core.labels import MathLabel
from ..core.live import ScrollingScope
from ..core.palette import PALETTE
from ._math import TF, poly_str, pid_terms, state_space, tf_from_coeffs, tf_from_pz
from .plots import PoleZeroPlot

MAX_NODES = 12


def _fraction(num: str, den: str, size: int = 24) -> VGroup:
    n, d = label(num, size, PALETTE.fg), label(den, size, PALETTE.fg)
    bar = Line(LEFT, RIGHT, color=PALETTE.fg, stroke_width=2.5).set_width(max(n.width, d.width) + 0.2)
    return VGroup(n, bar, d).arrange(DOWN, buff=0.08)


@dataclass(frozen=True)
class TFBlockData:
    tf: TF | None
    poles: tuple[complex, ...]
    zeros: tuple[complex, ...]
    stable: bool | None


class TransferFunctionBlock(Block):
    """Intuition: a box that answers a question: "if I feed you a signal, what comes out?" Its fraction is the whole answer.

    Give ``numerator`` and ``denominator`` (highest power first), or ``poles``/``zeros``/``gain``, or ``latex`` for a custom
    label. ``show_poles_zeros`` hangs a small pole-zero map under the box.
    """

    def __init__(self, numerator=None, denominator=None, *, latex: str | None = None, poles=None, zeros=None, gain: float = 1.0,
                 show_poles_zeros: bool = False, w: float = 2.6, h: float = 1.3, color: str = PALETTE.blue):
        if numerator is not None and denominator is not None:
            tf = tf_from_coeffs(numerator, denominator)
        elif poles is not None or zeros is not None:
            tf = tf_from_pz(poles or (), zeros or (), gain)
        elif latex is not None:
            tf = None
        else:
            raise ValueError("give numerator and denominator, or poles and zeros, or latex")
        super().__init__("H", w=w, h=h, color=color)
        self.remove(self.text)
        self.tf = tf
        self.text = MathLabel(latex, size=26) if latex is not None else _fraction(poly_str(tf.num), poly_str(tf.den))
        self.text.scale_to_fit_width(min(self.text.width, 0.9 * w)).move_to(self.box)
        if self.text.height > 0.85 * h:
            self.text.scale_to_fit_height(0.85 * h).move_to(self.box)
        self.add(self.text)
        self.inset = None
        if show_poles_zeros:
            if tf is None:
                raise ValueError("show_poles_zeros needs a numerator and denominator or poles and zeros, not latex alone")
            self.inset = PoleZeroPlot(list(tf.poles), list(tf.zeros), size=1.7).next_to(self.box, DOWN, buff=0.25)
            self.add(self.inset)

    @property
    def data(self) -> TFBlockData:
        if self.tf is None:
            return TFBlockData(None, (), (), None)
        return TFBlockData(self.tf, tuple(complex(p) for p in self.tf.poles), tuple(complex(z) for z in self.tf.zeros), self.tf.stable)


class SummingJunction(VGroup):
    """Intuition: the place where signals are added or subtracted. A minus sign means "take this one away"."""

    def __init__(self, signs: Sequence[str] = ("+", "-"), radius: float = 0.25, *, color: str = PALETTE.blue):
        super().__init__()
        if not 1 <= len(signs) <= 3 or any(s not in ("+", "-") for s in signs):
            raise ValueError("signs must be one to three of '+' or '-'")
        self.signs = tuple(signs)
        self.circle = Circle(radius=radius, stroke_color=color, stroke_width=3, fill_opacity=0)
        r = radius * 0.62
        self.cross = VGroup(Line((-r, -r, 0), (r, r, 0), color=color, stroke_width=2.5), Line((-r, r, 0), (r, -r, 0), color=color, stroke_width=2.5))
        self.add(self.circle, self.cross)
        spots = {"left": (-0.2, 0.2), "bottom": (0.2, -0.2), "top": (0.2, 0.2)}
        self.input_ports = ("left", "bottom", "top")[:len(signs)]
        for side, sign in zip(self.input_ports, signs):
            dx, dy = spots[side]
            tag = label(sign if sign == "+" else "−", 22, PALETTE.fg).move_to(self.port(side) + np.array([dx * 1.4, dy * 1.4, 0.0]))
            self.add(tag)

    def move_to(self, point_or_mobject, *args, **kwargs):
        """Place the circle's centre (not the centre of the whole group): the sign tags sit off to one side and must not shift it."""
        target = point_or_mobject.get_center() if isinstance(point_or_mobject, Mobject) else np.array(point_or_mobject, dtype=float)
        return self.shift(target - self.circle.get_center())

    def port(self, name: str) -> np.ndarray:
        if name not in PORTS:
            raise ValueError(f"unknown port '{name}'; choose one of {PORTS}")
        d = {"left": (-1, 0, 0), "right": (1, 0, 0), "top": (0, 1, 0), "bottom": (0, -1, 0)}[name]
        return self.circle.get_edge_center(np.array(d, dtype=float))


def _right_of(m: Mobject, y: float = 0.0) -> np.ndarray:
    """A point just right of a text label, at wire height ``y`` (a label's own centre can sit a little off the wire)."""
    return np.array([m.get_right()[0] + 0.05, y, 0.0])


def _left_of(m: Mobject, y: float = 0.0) -> np.ndarray:
    return np.array([m.get_left()[0] - 0.05, y, 0.0])


def _as_block(item, *, w: float = 1.7, h: float = 0.8) -> Mobject:
    if isinstance(item, str):
        return Block(item, w=w, h=h)
    if not hasattr(item, "port"):
        raise ValueError("blocks must be text or a Mobject with a port() method (such as Block)")
    return item


@dataclass(frozen=True)
class LoopData:
    forward: tuple[str, ...]
    feedback: tuple[str, ...]
    negative: bool


class FeedbackLoop(VGroup):
    """Intuition: a loop that watches its own result. The output is measured, sent back, and subtracted from the goal, so the
    controller only ever sees the error.

    ``forward`` and ``feedback`` are block labels (or Blocks). With an empty ``feedback`` the loop has unity feedback (a plain wire).
    """

    def __init__(self, forward: Sequence = ("Controller", "Plant"), feedback: Sequence = (), reference: str = "r", output: str = "y", *,
                 negative: bool = True, gap: float = 0.8):
        super().__init__()
        if not 1 <= len(forward) <= 4 or len(feedback) > 3:
            raise ValueError("a loop needs 1 to 4 forward blocks and at most 3 feedback blocks")
        self.negative, self._names = negative, (tuple(str(f) for f in forward), tuple(str(f) for f in feedback))
        self.forward_blocks = [_as_block(f) for f in forward]
        self.feedback_blocks = [_as_block(f) for f in feedback]
        self.ref = label(reference, 30, PALETTE.fg)
        self.sum = SummingJunction(("+", "-" if negative else "+"))
        self.out = label(output, 30, PALETTE.fg)
        x = 0.0
        self.ref.move_to((x, 0, 0))
        x += 0.6 + gap
        self.sum.move_to((x, 0, 0))
        x += 0.25 + gap
        for b in self.forward_blocks:
            x += b.width / 2
            b.move_to((x, 0, 0))
            x += b.width / 2 + gap
        take_x = x - gap / 2
        self.out.move_to((x + 0.1, 0, 0))
        fb_y = -1.6
        cur = take_x
        for b in self.feedback_blocks:
            cur -= gap / 2 + b.width / 2 if b is self.feedback_blocks[0] else gap + b.width / 2
            b.move_to((cur, fb_y, 0))
            cur -= b.width / 2
        sx, wires = self.sum.circle.get_center()[0], VGroup()
        fwd = [_right_of(self.ref), self.sum.port("left")]
        wires.add(Wire(fwd, arrow=True))
        prev = self.sum.port("right")
        for b in self.forward_blocks:
            wires.add(Wire([prev, b.port("left")], arrow=True))
            prev = b.port("right")
        tap = np.array([take_x, 0, 0])
        wires.add(Wire([prev, _left_of(self.out)], arrow=True))
        path = [tap, np.array([take_x, fb_y, 0])]
        prev_pt = path[-1]
        wires.add(Wire([tap, prev_pt]))
        for b in self.feedback_blocks:
            wires.add(Wire([prev_pt, b.port("right")]))
            prev_pt = b.port("left")
            path += [b.port("right"), b.port("left")]
        back = [prev_pt, np.array([sx, fb_y, 0]), self.sum.port("bottom")]
        wires.add(Wire(back, arrow=True, color=PALETTE.teal))
        self.tap = Dot(tap, radius=0.07, color=PALETTE.muted)
        self.forward_path = VMobject().set_points_as_corners([self.ref.get_center(), self.out.get_center()])
        self.loop_path = VMobject().set_points_as_corners([self.sum.circle.get_center(), tap, *path[1:], np.array([sx, fb_y, 0]), self.sum.circle.get_center()])
        self.wires = wires
        self.add(wires, self.tap, self.ref, self.sum, self.out, *self.forward_blocks, *self.feedback_blocks)

    @property
    def data(self) -> LoopData:
        return LoopData(self._names[0], self._names[1], self.negative)

    def animate_signal(self, *, pulses: int = 2, run_time: float = 4.0):
        """A yellow pulse crosses from goal to result, then a teal pulse rides the feedback path back to the junction."""
        if pulses < 1:
            raise ValueError("pulses must be at least 1")
        each = run_time / (2 * pulses)
        anims = []
        for _ in range(pulses):
            anims += [pulse_along(self.forward_path, PALETTE.yellow, each), pulse_along(self.loop_path, PALETTE.teal, each)]
        return Succession(*anims)


@dataclass(frozen=True)
class PIDStructure:
    kp: float
    ki: float
    kd: float
    peak_p: float
    peak_i: float
    peak_d: float


def _default_error(t: float) -> float:
    return math.exp(-0.4 * t) * math.cos(1.6 * t) if t >= 0 else 0.0


class PIDController(VGroup):
    """Intuition: three advisers look at the same error. P reacts to how big it is now, I to how long it has lingered, D to how fast it is changing.

    The three small scopes on the right show each adviser's contribution to the control signal ``u`` for the error signal ``error_fn``.
    """

    def __init__(self, kp: float = 1.0, ki: float = 0.5, kd: float = 0.2, error_fn: Callable[[float], float] | None = None, *,
                 t_end: float = 12.0, window: float = 4.0):
        super().__init__()
        if min(kp, ki, kd) < 0:
            raise ValueError("kp, ki and kd must not be negative")
        self.kp, self.ki, self.kd, self.t_end, self.window = kp, ki, kd, t_end, window
        self.clock = ValueTracker(0.0)
        t, p, i, d = pid_terms(error_fn or _default_error, kp, ki, kd, t_end)
        self.terms = (p, i, d)
        amp = max(1e-6, *(float(np.max(np.abs(a))) for a in self.terms))
        ys, names = (1.1, 0.0, -1.1), (f"P  {kp:g}", f"I  {ki:g} / s", f"D  {kd:g} s")
        cols = (PALETTE.blue, PALETTE.teal, PALETTE.yellow)
        self.blocks = [Block(n, w=1.6, h=0.7, color=c).move_to((-1.2, y, 0)) for n, y, c in zip(names, ys, cols)]
        self.sum = SummingJunction(("+", "+", "+")).move_to((1.0, 0, 0))
        e_tag, u_tag = label("e", 30, PALETTE.fg).move_to((-3.7, 0, 0)), label("u", 30, PALETTE.fg).move_to((2.4, 0, 0))
        split = -2.6
        wires = VGroup(Wire([_right_of(e_tag), (split, 0, 0)]), Wire([(split, ys[0], 0), (split, ys[2], 0)]))
        for b, y in zip(self.blocks, ys):
            wires.add(Wire([(split, y, 0), b.port("left")], arrow=True))
        wires.add(Wire([self.blocks[0].port("right"), (1.0, ys[0], 0), self.sum.port("top")], arrow=True),
                  Wire([self.blocks[1].port("right"), self.sum.port("left")], arrow=True),
                  Wire([self.blocks[2].port("right"), (1.0, ys[2], 0), self.sum.port("bottom")], arrow=True),
                  Wire([self.sum.port("right"), _left_of(u_tag)], arrow=True))
        self.scopes = VGroup()
        for arr, y, c, tag in zip(self.terms, ys, cols, "PID"):
            fn = lambda x, arr=arr: float(np.interp(x, t, arr, left=0.0, right=arr[-1]))
            sc = ScrollingScope(fn, tracker=self.clock, window=window, width=2.8, height=0.8, amplitude=amp, color=c)
            box = Rectangle(width=2.8, height=0.8, stroke_color=PALETTE.muted, stroke_width=1.5, fill_opacity=0).move_to(sc)
            self.scopes.add(VGroup(box, sc, label(tag, 24, c).next_to(box, LEFT, buff=0.15)))
        self.scopes.arrange(DOWN, buff=0.3).move_to((5.6, 0, 0))
        self.scope_title = label("contribution to u", 20, PALETTE.muted).next_to(self.scopes, UP, buff=0.2)
        self.add(wires, e_tag, u_tag, *self.blocks, self.sum, self.scopes, self.scope_title)

    @property
    def data(self) -> PIDStructure:
        return PIDStructure(self.kp, self.ki, self.kd, *(float(np.max(np.abs(a))) for a in self.terms))

    def animate_response(self, *, run_time: float = 6.0):
        """Play the error signal through time: the three traces scroll past, each showing its adviser's share of ``u``."""
        self.clock.set_value(0.0)
        return self.clock.animate.set_value(self.t_end).build().set_run_time(run_time).set_rate_func(linear)


@dataclass(frozen=True)
class StateSpaceData:
    states: int
    inputs: int
    outputs: int
    eigenvalues: tuple[complex, ...]
    stable: bool


class StateSpaceDiagram(VGroup):
    """Intuition: a bucket brigade. The integrators hold the state; A feeds it back on itself, B pours the input in, C reads it out, D lets input leak straight through."""

    def __init__(self, A, B, C, D=None):
        super().__init__()
        self.ss = state_space(A, B, C, D)
        s = self.ss
        w = 1.0
        blk = lambda name, x, y, shape: (Block(name, w=w, h=0.7).move_to((x, y, 0)),
                                         label(shape, 18, PALETTE.muted).move_to((x, y - 0.55, 0)))
        (self.B, tb), (self.I, ti) = blk("B", -3.3, 0, f"{s.n}×{s.inputs}"), blk("∫", -0.6, 0, f"{s.n} states")
        (self.C, tc), (self.A, ta) = blk("C", 1.9, 0, f"{s.outputs}×{s.n}"), blk("A", -0.6, -1.4, f"{s.n}×{s.n}")
        self.D, td = blk("D", -0.6, 1.4, f"{s.outputs}×{s.inputs}")
        self.A.set_color(PALETTE.teal)
        self.D.set_color(PALETTE.teal)
        s1, s2 = SummingJunction(("+", "+")).move_to((-1.9, 0, 0)), SummingJunction(("+", "+")).move_to((3.3, 0, 0))
        self.sum_in, self.sum_out = s1, s2
        u, y = label("u", 30, PALETTE.fg).move_to((-4.7, 0, 0)), label("y", 30, PALETTE.fg).move_to((4.7, 0, 0))
        tap_u, tap_x = (-4.0, 0, 0), (0.75, 0, 0)
        wires = VGroup(
            Wire([_right_of(u), self.B.port("left")], arrow=True), Wire([self.B.port("right"), s1.port("left")], arrow=True),
            Wire([s1.port("right"), self.I.port("left")], arrow=True), Wire([self.I.port("right"), self.C.port("left")], arrow=True),
            Wire([self.C.port("right"), s2.port("left")], arrow=True), Wire([s2.port("right"), _left_of(y)], arrow=True),
            Wire([tap_x, (0.75, -1.4, 0), self.A.port("right")]), Wire([self.A.port("left"), (-1.9, -1.4, 0), s1.port("bottom")], arrow=True),
            Wire([tap_u, (-4.0, 1.4, 0), self.D.port("left")]), Wire([self.D.port("right"), (3.3, 1.4, 0), s2.port("top")], arrow=True))
        taps = VGroup(*[Dot(p, radius=0.07, color=PALETTE.muted) for p in (tap_u, tap_x)])
        self.add(wires, taps, u, y, self.B, self.I, self.C, self.A, self.D, tb, ti, tc, ta, td, s1, s2)

    @property
    def data(self) -> StateSpaceData:
        s = self.ss
        return StateSpaceData(s.n, s.inputs, s.outputs, tuple(complex(e) for e in s.eigenvalues), s.stable)


@dataclass(frozen=True)
class BlockDiagramData:
    layers: dict
    feedback_edges: tuple[tuple[str, str], ...]


class BlockDiagram(VGroup):
    """Intuition: a map of who talks to whom. Signals flow left to right; any edge that points back is drawn as a loop underneath.

    ``nodes`` maps a name to a label, or to ``{"label": ..., "kind": "block" | "sum" | "io", "signs": ("+", "-")}``.
    ``edges`` are ``(src, dst)`` or ``(src, dst, text)``.
    """

    def __init__(self, nodes: dict, edges: Sequence[tuple], *, layout: str = "layered", dx: float = 2.6, dy: float = 1.5):
        super().__init__()
        if layout != "layered":
            raise ValueError("only layout='layered' is supported")
        if not 1 <= len(nodes) <= MAX_NODES:
            raise ValueError(f"a block diagram needs 1 to {MAX_NODES} nodes")
        specs = {n: ({"label": s} if isinstance(s, str) else dict(s)) for n, s in nodes.items()}
        norm = []
        for e in edges:
            if len(e) not in (2, 3) or e[0] not in specs or e[1] not in specs:
                raise ValueError(f"edge {e!r} must be (src, dst) or (src, dst, text) between known node names")
            norm.append((e[0], e[1], e[2] if len(e) == 3 else ""))
        succ = {n: [d for s, d, _ in norm if s == n] for n in specs}
        has_in = {d for _, d, _ in norm}
        state, back = {}, set()

        def dfs(u):
            state[u] = 1
            for v in succ[u]:
                if state.get(v) == 1:
                    back.add((u, v))
                elif v not in state:
                    dfs(v)
            state[u] = 2

        for n in [n for n in specs if n not in has_in] + list(specs):
            if n not in state:
                dfs(n)
        layer = {n: 0 for n in specs}
        for _ in range(len(specs)):
            for s, d, _ in norm:
                if (s, d) not in back:
                    layer[d] = max(layer[d], layer[s] + 1)
        by_layer: dict[int, list[str]] = {}
        for n in specs:
            by_layer.setdefault(layer[n], []).append(n)
        self.nodes: dict[str, Mobject] = {}
        for ly, names in by_layer.items():
            for i, n in enumerate(names):
                sp, kind = specs[n], specs[n].get("kind", "block")
                mob = SummingJunction(tuple(sp.get("signs", ("+", "-")))) if kind == "sum" else \
                    label(sp.get("label", n), 28, PALETTE.fg) if kind == "io" else Block(sp.get("label", n), w=1.7, h=0.8)
                self.nodes[n] = mob.move_to((ly * dx, ((len(names) - 1) / 2 - i) * dy, 0))
        low = min(m.get_bottom()[1] for m in self.nodes.values()) - 0.55
        used, lanes, self.wires = {n: 0 for n in specs}, 0, VGroup()
        for s, d, text in norm:
            a, b = self._out(s), None
            if (s, d) in back:
                end = self.nodes[d]
                pb = end.port("bottom") if hasattr(end, "port") else end.get_bottom()
                y = low - 0.35 * lanes
                lanes += 1
                pts = [self._edge(s, DOWN), np.array([self._edge(s, DOWN)[0], y, 0]), np.array([pb[0], y, 0]), pb]
            else:
                dst = self.nodes[d]
                side = ("left", "top", "bottom")[min(used[d], 2)] if isinstance(dst, SummingJunction) else "left"
                used[d] += 1
                b = dst.port(side) if hasattr(dst, "port") else dst.get_left()
                if side == "left":
                    mx = 0.5 * (a[0] + b[0])
                    pts = [a, b] if abs(a[1] - b[1]) < 1e-6 else [a, np.array([mx, a[1], 0]), np.array([mx, b[1], 0]), b]
                else:
                    pts = [a, np.array([b[0], a[1], 0]), b]
            self.wires.add(Wire(pts, arrow=True))
            if text:
                self.wires.add(label(text, 20, PALETTE.muted).next_to(pts[0] * 0.5 + pts[1] * 0.5, UP, buff=0.08))
        self._data = BlockDiagramData(layer, tuple(sorted(back)))
        self.clock = ValueTracker(0.0)
        self.add(self.wires, *self.nodes.values())

    def _edge(self, name: str, direction) -> np.ndarray:
        m = self.nodes[name]
        if hasattr(m, "port"):
            return m.port({tuple(RIGHT): "right", tuple(DOWN): "bottom"}[tuple(direction)])
        return m.get_edge_center(direction)

    def _out(self, name: str) -> np.ndarray:
        return self._edge(name, RIGHT)

    @property
    def data(self) -> BlockDiagramData:
        return self._data

    def tag_scope(self, node: str, signal_fn: Callable[[float], float], *, tracker: ValueTracker | None = None, window: float = 4.0,
                  color: str = PALETTE.green) -> ScrollingScope:
        """Hang a small live scope above ``node`` showing ``signal_fn(t)``; it follows ``tracker`` (default: this diagram's clock)."""
        if node not in self.nodes:
            raise ValueError(f"unknown node '{node}'")
        scope = ScrollingScope(signal_fn, tracker=tracker or self.clock, window=window, width=1.7, height=0.6, color=color)
        scope.next_to(self.nodes[node], UP, buff=0.4)
        self.add(scope)
        return scope

    def animate_time(self, duration: float = 8.0, *, run_time: float = 6.0):
        self.clock.set_value(0.0)
        return self.clock.animate.set_value(duration).build().set_run_time(run_time).set_rate_func(linear)
