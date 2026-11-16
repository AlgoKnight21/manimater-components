"""Grid components and one-line diagrams (README section 9.2)."""
from __future__ import annotations

from collections import deque

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, AnimationGroup, Arrow, Circle, DashedLine, Dot, Flash, Indicate, Line, Rectangle, Rotate, Succession,
                   Text, Triangle, VGroup, VMobject)

from ..core.flow import pulse_along
from ..core.palette import PALETTE
from ._widgets import capacitor, inductor, resistor, wire

HOT = PALETTE.yellow


class Bus(VGroup):
    """Intuition: a thick copper bar that everything in a substation hangs off."""

    def __init__(self, name: str, kv: float | None = None, length: float = 2.0):
        super().__init__()
        self.name, self.kv, self.length = name, kv, length
        self.bar = Line((-length / 2, 0, 0), (length / 2, 0, 0), color=PALETTE.blue, stroke_width=10)
        txt = name if kv is None else f"{name}  {kv:g} kV"
        self.label = Text(txt, font_size=18, color=PALETTE.fg).next_to(self.bar, UP, buff=0.12)
        self.add(self.bar, self.label)

    def port(self, frac: float = 0.5) -> np.ndarray:
        """A point on the bar: 0 is the left end, 1 the right end."""
        return self.bar.point_from_proportion(min(max(frac, 0.0), 1.0))


class Breaker(VGroup):
    """Intuition: a box that can cut a fault current: solid when closed, hollow with a gap when open."""

    def __init__(self, closed: bool = True, size: float = 0.34):
        super().__init__()
        self.size, self.closed = size, closed
        self.box = Rectangle(width=size, height=size, stroke_width=3)
        self.add(self.box)
        self._paint()

    def _paint(self) -> None:
        if self.closed:
            self.box.set_stroke(PALETTE.fg).set_fill(PALETTE.fg, opacity=1.0)
        else:
            self.box.set_stroke(PALETTE.red).set_fill(PALETTE.bg, opacity=1.0)

    def set_closed(self, closed: bool) -> "Breaker":
        self.closed = closed
        self._paint()
        return self

    def animate_trip(self, run_time: float = 0.5):
        """Fling open: flash, then go hollow and red."""
        self.closed = False
        return AnimationGroup(Flash(self.box, color=PALETTE.red, flash_radius=self.size * 1.2), self.box.animate.set_stroke(PALETTE.red).set_fill(PALETTE.bg, opacity=1.0), run_time=run_time)


class TransmissionLineSymbol(VGroup):
    """Intuition: a long wire is not a wire: it has resistance, inductance and a little capacitance to the ground."""

    def __init__(self, length_km: float = 100.0, kv: float = 220.0, model: str = "lumped", width: float = 3.4):
        super().__init__()
        if model not in ("lumped", "pi"):
            raise ValueError("model must be 'lumped' or 'pi'")
        self.model = model
        label = Text(f"{length_km:g} km, {kv:g} kV", font_size=18, color=PALETTE.muted)
        if model == "lumped":
            self.add(wire((-width / 2, 0, 0), (width / 2, 0, 0), PALETTE.fg, 4), label.next_to((0, 0, 0), UP, buff=0.15))
        else:
            self.add(wire((-width / 2, 0, 0), (-width / 2 + 0.3, 0, 0), PALETTE.fg), resistor((-width / 2 + 0.3, 0, 0), (-0.1, 0, 0), PALETTE.fg, n=4),
                     inductor((-0.1, 0, 0), (width / 2 - 0.3, 0, 0), PALETTE.fg, loops=3), wire((width / 2 - 0.3, 0, 0), (width / 2, 0, 0), PALETTE.fg))
            self.add(Text("R", font_size=16, color=PALETTE.fg).move_to((-width / 4 - 0.1, 0.3, 0)), Text("X", font_size=16, color=PALETTE.fg).move_to((width / 4 - 0.1, 0.3, 0)))
            for x in (-width / 2 + 0.3, width / 2 - 0.3):
                self.add(capacitor((x, 0, 0), (x, -0.9, 0), PALETTE.teal), wire((x - 0.2, -0.9, 0), (x + 0.2, -0.9, 0), PALETTE.muted, 3))
                self.add(Text("B/2", font_size=14, color=PALETTE.teal).move_to((x + 0.35, -0.4, 0)))
            self.add(label.next_to((0, 0, 0), UP, buff=0.55))


class TransformerSymbol(VGroup):
    """Intuition: two coils sharing one iron loop. Alternating current in one makes alternating flux that drives the other."""

    def __init__(self, ratio: float = 2.0, connection: str = "Dyn11"):
        super().__init__()
        self.ratio, self.connection = ratio, connection
        self.coil_p = inductor((-0.55, 0.7, 0), (-0.55, -0.7, 0), PALETTE.blue, loops=4)
        self.coil_s = inductor((0.55, 0.7, 0), (0.55, -0.7, 0), PALETTE.teal, loops=4)
        self.core = Rectangle(width=1.1, height=1.45, stroke_color=PALETTE.muted, stroke_width=2).move_to((0, 0, 0))
        self.label = Text(f"{ratio:g}:1  {connection}", font_size=18, color=PALETTE.muted).next_to(self.core, DOWN, buff=0.35)
        self.add(self.core, self.coil_p, self.coil_s, self.label)

    def animate_flux(self, run_time: float = 2.0):
        """Flux races round the iron loop one way, then the other (it alternates)."""
        loop = self.core.copy()
        return Succession(pulse_along(loop, PALETTE.yellow, run_time=run_time / 2, width=0.15), pulse_along(loop.copy().reverse_points(), PALETTE.yellow, run_time=run_time / 2, width=0.15))


class SynchronousGeneratorSymbol(VGroup):
    """Intuition: a circle marked G, with a magnet spinning inside it at exactly grid speed."""

    def __init__(self, rating_mva: float = 100.0, poles: int = 2, radius: float = 0.6):
        super().__init__()
        self.poles = poles
        self.shell = Circle(radius=radius, color=PALETTE.fg, stroke_width=4)
        self.rotor = VGroup(Line((-radius * 0.6, 0, 0), (radius * 0.6, 0, 0), color=PALETTE.yellow, stroke_width=6),
                            Dot((radius * 0.6, 0, 0), radius=0.07, color=PALETTE.red), Dot((-radius * 0.6, 0, 0), radius=0.07, color=PALETTE.blue))
        self.label = Text(f"{rating_mva:g} MVA", font_size=16, color=PALETTE.muted).next_to(self.shell, DOWN, buff=0.1)
        self.add(self.shell, self.rotor, Text("G", font_size=18, color=PALETTE.fg).move_to((0, radius * 0.0, 0)).set_opacity(0.0), self.label)

    def animate_spin(self, turns: float = 2.0, run_time: float = 3.0):
        return Rotate(self.rotor, angle=2 * np.pi * turns, about_point=self.shell.get_center(), run_time=run_time, rate_func=lambda t: t)


class Substation(VGroup):
    """Intuition: how feeders hang off the bus bars: one bar (cheap), two bars (flexible), a ring (any one breaker can be taken out)."""

    def __init__(self, scheme: str = "double", feeders: int = 3, spacing: float = 1.5):
        super().__init__()
        if scheme not in ("single", "double", "ring"):
            raise ValueError("scheme must be 'single', 'double' or 'ring'")
        if not 1 <= feeders <= 6:
            raise ValueError("feeders must be between 1 and 6")
        self.scheme, self.feeders = scheme, feeders
        self.elements: dict[str, VGroup | Line | Breaker] = {}
        w = spacing * feeders
        if scheme == "ring":
            self._ring(spacing)
        else:
            self._bars(scheme, feeders, spacing, w)
        self.add(*self.elements.values())

    def _bars(self, scheme: str, n: int, sp: float, w: float) -> None:
        E = self.elements
        E["bus_a"] = Line((-w / 2, 1.0, 0), (w / 2, 1.0, 0), color=PALETTE.blue, stroke_width=9)
        if scheme == "double":
            E["bus_b"] = Line((-w / 2, 0.45, 0), (w / 2, 0.45, 0), color=PALETTE.purple, stroke_width=9)
        for i in range(n):
            x = -w / 2 + sp * (i + 0.5)
            E[f"fdr{i}"] = Line((x, 1.0, 0), (x, -1.4, 0), color=PALETTE.muted, stroke_width=3)
            E[f"brk{i}"] = Breaker(True).move_to((x, -0.4, 0))
            E[f"dis_a{i}"] = Dot((x, 1.0, 0), radius=0.08, color=PALETTE.fg)
            if scheme == "double":
                E[f"dis_b{i}"] = Dot((x, 0.45, 0), radius=0.08, color=PALETTE.bg).set_stroke(PALETTE.fg, 2)
            E[f"load{i}"] = Triangle(color=PALETTE.muted, stroke_width=3).scale(0.13).rotate(np.pi).move_to((x, -1.55, 0))
        E["label_a"] = Text("bus A", font_size=16, color=PALETTE.blue).next_to(E["bus_a"], LEFT, buff=0.1)
        if scheme == "double":
            E["label_b"] = Text("bus B", font_size=16, color=PALETTE.purple).next_to(E["bus_b"], LEFT, buff=0.1)
            E["coupler"] = Breaker(True).move_to((w / 2 + 0.35, 0.72, 0))

    def _ring(self, sp: float) -> None:
        E = self.elements
        a, b, c, d = (-sp, sp * 0.7, 0), (sp, sp * 0.7, 0), (sp, -sp * 0.7, 0), (-sp, -sp * 0.7, 0)
        corners = [a, b, c, d]
        for k, (p, q) in enumerate(zip(corners, corners[1:] + corners[:1])):
            E[f"ring{k}"] = Line(p, q, color=PALETTE.blue, stroke_width=9)
            E[f"brk{k}"] = Breaker(True).move_to((np.array(p) + np.array(q)) / 2)
        for i in range(min(self.feeders, 4)):
            p = np.array(corners[i], dtype=float)
            out = p / np.linalg.norm(p)
            E[f"fdr{i}"] = Line(p, p + out * 0.9, color=PALETTE.muted, stroke_width=3)
            E[f"load{i}"] = Triangle(color=PALETTE.muted, stroke_width=3).scale(0.13).move_to(p + out * 1.05).rotate(np.arctan2(out[1], out[0]) - np.pi / 2 + np.pi)

    def animate_switching(self, path: list[str], run_time: float | None = None):
        """Light the named elements one after another: that is the route the current now takes."""
        missing = [p for p in path if p not in self.elements]
        if missing:
            raise ValueError(f"unknown elements {missing}; choose from {sorted(self.elements)}")
        for name, el in self.elements.items():
            if name.startswith(("label", "load")):
                continue
            (el.box if isinstance(el, Breaker) else el).set_color(PALETTE.muted if not name.startswith(("bus", "ring")) else el.get_color())
        anims = [(el.box if isinstance(el, Breaker) else el).animate.set_color(HOT) for el in (self.elements[p] for p in path)]
        return AnimationGroup(*anims, lag_ratio=0.35, run_time=run_time or 0.6 * max(len(path), 1))


class PowerFlowArrows(VGroup):
    """Intuition: two arrows riding a line: yellow for real power P (the useful part), teal for reactive Q (swapped back and forth)."""

    def __init__(self, start, end, P: float, Q: float = 0.0, per_unit: float = 0.5):
        super().__init__()
        s, e = np.array(start, dtype=float), np.array(end, dtype=float)
        u = (e - s) / np.linalg.norm(e - s)
        n = np.array([-u[1], u[0], 0.0])
        mid = (s + e) / 2
        for val, col, off in ((P, HOT, 0.14), (Q, PALETTE.teal, -0.14)):
            if abs(val) < 1e-9:
                continue
            L = float(np.clip(0.25 + abs(val) * per_unit, 0.3, np.linalg.norm(e - s) * 0.8))
            d = u * np.sign(val)
            self.add(Arrow(mid + n * off - d * L / 2, mid + n * off + d * L / 2, buff=0, color=col, stroke_width=5 if col == HOT else 3, max_tip_length_to_length_ratio=0.35))


class PowerGrid(VGroup):
    """Intuition: the whole network on one page: sources, lines, loads, and arrows showing which way and how much power moves."""

    def __init__(self, nodes: dict[str, dict], edges: list[tuple[str, str]], flows: dict[tuple[str, str], tuple[float, float]] | None = None, scale: float = 1.0):
        super().__init__()
        self.nodes, self.edges, self.flows = nodes, [tuple(e) for e in edges], flows or {}
        for a, b in self.edges:
            if a not in nodes or b not in nodes:
                raise ValueError(f"edge ({a}, {b}) uses an unknown node")
        self.pos = {k: np.array([*v["pos"][:2], 0.0], dtype=float) * scale for k, v in nodes.items()}
        self.lines: dict[tuple[str, str], Line] = {e: Line(self.pos[e[0]], self.pos[e[1]], color=PALETTE.fg, stroke_width=4) for e in self.edges}
        self.symbols: dict[str, VGroup] = {}
        self.breakers: dict[tuple[str, str], tuple[Breaker, Breaker]] = {}
        for k, v in nodes.items():
            kind = v.get("type", "bus")
            p = self.pos[k]
            if kind == "gen":
                sym = VGroup(Circle(radius=0.32, color=PALETTE.green, stroke_width=4, fill_color=PALETTE.bg, fill_opacity=1), Text("G", font_size=22, color=PALETTE.green)).move_to(p)
            elif kind == "load":
                sym = VGroup(Triangle(color=PALETTE.red, stroke_width=4, fill_color=PALETTE.bg, fill_opacity=1).scale(0.25).rotate(np.pi)).move_to(p)
            else:
                sym = VGroup(Line(p + LEFT * 0.35, p + RIGHT * 0.35, color=PALETTE.blue, stroke_width=9)).move_to(p)
            self.symbols[k] = sym
            lab = Text(k, font_size=16, color=PALETTE.muted).next_to(sym, UP if kind != "load" else DOWN, buff=0.12)
            sym.add(lab)
        for e, ln in self.lines.items():
            ba, bb = Breaker(True, 0.2).move_to(ln.point_from_proportion(0.2)), Breaker(True, 0.2).move_to(ln.point_from_proportion(0.8))
            self.breakers[e] = (ba, bb)
        self.arrows = VGroup(*(PowerFlowArrows(self.pos[e[0]], self.pos[e[1]], *self.flows[e]) for e in self.edges if e in self.flows))
        self.add(*self.lines.values(), *(b for pair in self.breakers.values() for b in pair), *self.symbols.values(), self.arrows)

    @classmethod
    def radial(cls, n_loads: int = 3, flows: bool = True) -> "PowerGrid":
        nodes = {"G1": {"pos": (-4.5, 0), "type": "gen"}, "Bus1": {"pos": (-2.0, 0)}}
        edges = [("G1", "Bus1")]
        fl = {("G1", "Bus1"): (float(n_loads), 0.5)}
        for i in range(n_loads):
            y = (i - (n_loads - 1) / 2) * 1.6
            nodes[f"L{i + 1}"] = {"pos": (2.0, y), "type": "load"}
            edges.append(("Bus1", f"L{i + 1}"))
            fl[("Bus1", f"L{i + 1}")] = (1.0, 0.3)
        return cls(nodes, edges, fl if flows else None)

    @classmethod
    def ring(cls, n: int = 4, radius: float = 2.2) -> "PowerGrid":
        nodes = {"G1": {"pos": (0, radius * 1.0 + 0.0), "type": "gen"}}
        for i in range(n):
            a = np.pi / 2 - 2 * np.pi * (i + 1) / (n + 1)
            nodes[f"B{i + 1}"] = {"pos": (radius * np.cos(a) * 1.4, radius * np.sin(a))}
        names = ["G1"] + [f"B{i + 1}" for i in range(n)]
        edges = list(zip(names, names[1:] + names[:1]))
        return cls(nodes, edges, {e: (0.8, 0.2) for e in edges})

    def _neighbours(self) -> dict[str, list[str]]:
        adj: dict[str, list[str]] = {k: [] for k in self.nodes}
        for a, b in self.edges:
            adj[a].append(b)
            adj[b].append(a)
        return adj

    def _route(self, src: str, dst: str) -> list[str]:
        adj, prev, q = self._neighbours(), {src: None}, deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in prev:
                    prev[v] = u
                    q.append(v)
        out, cur = [], dst
        while cur is not None and cur in prev:
            out.append(cur)
            cur = prev[cur]
        return out[::-1]

    def animate_fault(self, at: str | tuple[str, str], run_time: float = 2.0):
        """Flash the fault and send a surge of current into it from every generator."""
        if isinstance(at, tuple):
            at = tuple(at)
            point, target = (self.pos[at[0]] + self.pos[at[1]]) / 2, at[0]
        else:
            if at not in self.nodes:
                raise ValueError(f"unknown node {at}")
            point, target = self.pos[at], at
        surges = []
        for g, v in self.nodes.items():
            if v.get("type") == "gen":
                route = self._route(g, target)
                if len(route) > 1:
                    path = VMobject().set_points_as_corners([self.pos[n] for n in route] + ([point] if isinstance(at, tuple) else []))
                    surges.append(pulse_along(path, PALETTE.red, run_time=run_time * 0.6, width=0.18))
        return Succession(Flash(point, color=PALETTE.red, flash_radius=0.6, run_time=run_time * 0.3), AnimationGroup(*surges))

    def animate_trip(self, breakers: list[tuple[str, str]], run_time: float = 1.0):
        """Open both breakers on each named line: the faulted line goes dark and the rest keeps running."""
        anims = []
        for e in breakers:
            e = tuple(e)
            if e not in self.breakers:
                raise ValueError(f"no line {e}")
            anims += [b.animate_trip() for b in self.breakers[e]] + [self.lines[e].animate.set_color(PALETTE.muted).set_stroke(opacity=0.4)]
        return AnimationGroup(*anims, run_time=run_time)
