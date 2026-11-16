"""Caches, buses, RAM and DMA (README section 7.3)."""
from __future__ import annotations

import math

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, AnimationGroup, Arrow, FadeIn, FadeOut, Indicate, LaggedStart, Line, MoveAlongPath,
                   Rectangle, ShowPassingFlash, Succession, Transform, VGroup, VMobject)

from ...ctx import label
from ..core.glow import GlowDot
from ..core.palette import PALETTE
from ._frame import Framed
from ._math import CacheModel, DMASchedule, dma_schedule

BIT_W = 0.3  # width of one address bit in the address bar


class CacheArray(Framed, VGroup):
    """Intuition: a small set of shelves. The index picks the shelf, the tag says which book is on it, and the offset picks a word inside the book.

    The cache is stepped for real by ``CacheModel`` each time ``animate_access`` is called, so the pictures always match the
    numbers in ``data``. Hits flash green; misses flash red, then the chosen way is overwritten (the victim is the LRU or
    oldest way). The small grey text per set names the next victim.
    """

    CELL_W, CELL_H = 1.7, 0.5

    def __init__(self, sets: int = 4, ways: int = 2, block: int = 4, policy: str = "lru", *, addr_bits: int = 8):
        super().__init__()
        self._init_frame()
        self.model = CacheModel(sets, ways, block, policy)
        m = self.model
        self.addr_bits = addr_bits
        self.tag_bits = addr_bits - m.index_bits - m.offset_bits
        if self.tag_bits < 1:
            raise ValueError("addr_bits is too small for this many sets and this block size")
        if not (1 <= sets <= 8 and 1 <= ways <= 4):
            raise ValueError("the picture supports 1 to 8 sets and 1 to 4 ways")
        # address bar
        self.field_boxes, self.field_texts = [], []
        spec = [("tag", self.tag_bits, PALETTE.teal), ("index", m.index_bits, PALETTE.blue), ("offset", m.offset_bits, PALETTE.purple)]
        x = 0.0
        bar = VGroup()
        for name, bits, col in spec:
            w = max(bits, 1) * BIT_W + 0.2
            rect = Rectangle(width=w, height=0.5, stroke_color=col, stroke_width=2, fill_color=col, fill_opacity=0.25).move_to((x + w / 2, 0, 0))
            txt = label("0" * bits if bits else "-", 22, PALETTE.fg).move_to(rect)
            cap = label(name, 18, col).next_to(rect, UP, buff=0.08)
            self.field_boxes.append(rect)
            self.field_texts.append(txt)
            bar.add(rect, txt, cap)
            x += w + 0.08
        bar.move_to((0, 0, 0))
        # shelves
        self.rects, self.texts = {}, {}
        self.order_texts = []
        grid = VGroup()
        top = -0.9
        for s in range(sets):
            y = top - s * (self.CELL_H + 0.1)
            grid.add(label(f"set {s}", 20, PALETTE.muted).move_to((-ways * self.CELL_W / 2 - 0.55, y, 0)))
            for w_ in range(ways):
                cx = (w_ - (ways - 1) / 2) * (self.CELL_W + 0.08)
                r = Rectangle(width=self.CELL_W, height=self.CELL_H, stroke_color=PALETTE.muted, stroke_width=2, fill_opacity=0.0)
                r.move_to((cx, y, 0))
                t = label("-", 20, PALETTE.muted).move_to(r)
                self.rects[(s, w_)], self.texts[(s, w_)] = r, t
                grid.add(r, t)
            ot = label(f"victim: way {m.order[s][0]}", 16, PALETTE.muted).move_to((ways * self.CELL_W / 2 + 0.95, y, 0))
            self.order_texts.append(ot)
            grid.add(ot)
        self.status = label("accesses: 0", 22, PALETTE.fg).move_to((0, top - sets * (self.CELL_H + 0.1) - 0.1, 0))
        self.add(bar, grid, self.status)
        self.move_to((0, 0, 0))

    @property
    def data(self) -> dict:
        h = self.model.history
        hits = sum(1 for a in h if a.hit)
        return {"accesses": list(h), "hits": hits, "misses": len(h) - hits, "hit_rate": hits / len(h) if h else 0.0,
                "tag_bits": self.tag_bits, "index_bits": self.model.index_bits, "offset_bits": self.model.offset_bits}

    def animate_access(self, address: int, *, run_time: float = 2.4):
        """Look ``address`` up: split it into tag, index and offset, search the set, then hit or replace a way."""
        if not 0 <= address < 2 ** self.addr_bits:
            raise ValueError(f"address must fit in {self.addr_bits} bits")
        acc = self.model.access(address)
        m = self.model
        vals = [format(acc.tag, f"0{self.tag_bits}b"), format(acc.index, f"0{m.index_bits}b") if m.index_bits else "-",
                format(acc.offset, f"0{m.offset_bits}b") if m.offset_bits else "-"]
        t = run_time / 4
        split = AnimationGroup(*[Transform(txt, label(v, 22, PALETTE.fg).scale(self.unit).move_to(txt)) for txt, v in zip(self.field_texts, vals)],
                               run_time=t)
        row = [self.rects[(acc.index, w)] for w in range(m.ways)]
        ring = Rectangle(width=m.ways * (self.CELL_W + 0.08) + 0.15, height=self.CELL_H + 0.2, stroke_color=PALETTE.yellow,
                         stroke_width=4).move_to(VGroup(*row).get_center())
        look = Succession(FadeIn(ring, run_time=t * 0.5), FadeOut(ring, run_time=t * 0.5))
        cell, ctext = self.rects[(acc.index, acc.way)], self.texts[(acc.index, acc.way)]
        if acc.hit:
            outcome = Indicate(cell, color=PALETTE.green, scale_factor=1.12, run_time=t)
        else:
            new_text = label(f"tag {acc.tag}", 20, PALETTE.fg).scale(self.unit).move_to(cell)
            outcome = Succession(Indicate(cell, color=PALETTE.red, scale_factor=1.12, run_time=t * 0.5),
                                 AnimationGroup(Transform(ctext, new_text), cell.animate.set_fill(PALETTE.teal, opacity=0.3)
                                                .set_stroke(PALETTE.teal), run_time=t * 0.5))
        n = len(m.history)
        hits = sum(1 for a in m.history if a.hit)
        verdict = "HIT" if acc.hit else "MISS"
        st = label(f"access {n}: 0x{address:02X} {verdict} | hit rate {hits}/{n}", 22,
                   PALETTE.green if acc.hit else PALETTE.red).scale(self.unit).move_to(self.status)
        victim = label(f"victim: way {acc.order[0]}", 16, PALETTE.muted).scale(self.unit).move_to(self.order_texts[acc.index])
        tail = AnimationGroup(Transform(self.status, st), Transform(self.order_texts[acc.index], victim), run_time=t)
        return Succession(split, look, outcome, tail)


class CacheHierarchy(Framed, VGroup):
    """Intuition: a stack of ever-bigger, ever-slower pantries. Check the nearest one first; each miss sends you one level further for a longer wait.

    Each lookup supplies which levels hit (``hits``). The token walks down until a level hits (red = miss, green = hit), then
    returns, and the running average access time is updated. The cost of a lookup is the sum of the latencies of every level
    visited, so an L2 hit costs ``L1 + L2``.
    """

    def __init__(self, levels: tuple[str, ...] = ("L1", "L2", "L3"), ram: bool = True,
                 latencies: tuple[float, ...] = (1, 10, 40, 200)):
        super().__init__()
        self._init_frame()
        levels = tuple(levels)
        if not 1 <= len(levels) <= 4:
            raise ValueError("give between 1 and 4 cache levels")
        if len(latencies) != len(levels) + (1 if ram else 0):
            raise ValueError("latencies needs one entry per level, plus one for RAM when ram=True")
        if any(x <= 0 for x in latencies):
            raise ValueError("latencies must be positive")
        self.levels, self.ram, self.latencies = levels, ram, tuple(float(x) for x in latencies)
        names = ["CPU"] + list(levels) + (["RAM"] if ram else [])
        colors = [PALETTE.fg, PALETTE.blue, PALETTE.teal, PALETTE.purple, PALETTE.green, PALETTE.orange]
        self.blocks = []
        n = len(names)
        for i, name in enumerate(names):
            w = 2.0 + 0.7 * i
            y = ((n - 1) / 2 - i) * 1.0
            col = PALETTE.fg if i == 0 else (PALETTE.orange if (ram and i == n - 1) else colors[min(i, len(colors) - 1)])
            rect = Rectangle(width=w, height=0.7, stroke_color=col, stroke_width=3, fill_color=col, fill_opacity=0.15)
            grp = VGroup(rect, label(name, 24, PALETTE.fg).move_to(rect)).move_to((0, y, 0))
            self.blocks.append(grp)
        self.latency_tags = VGroup(*[label(f"{x:g} cycles", 18, PALETTE.muted).next_to(self.blocks[i + 1], RIGHT, buff=0.3)
                                     for i, x in enumerate(self.latencies)])
        self.links = VGroup(*[Arrow(self.blocks[i][0].get_bottom(), self.blocks[i + 1][0].get_top(), buff=0.03, color=PALETTE.muted,
                                    stroke_width=3, max_tip_length_to_length_ratio=0.3) for i in range(n - 1)])
        self.readout = label("average access time: -", 24, PALETTE.yellow).move_to((0, -(n - 1) / 2 - 0.9, 0))
        self.token = GlowDot(self.blocks[0].get_center(), radius=0.12)
        self.add(self.links, *self.blocks, self.latency_tags, self.readout, self.token)
        self.move_to((0, 0, 0))
        self._home = self.token.get_center().copy()
        self.lookups: list[tuple[int, float]] = []  # (level index that hit, cost in cycles)

    @property
    def amat_measured(self) -> float:
        """Average cost over the lookups played so far (0 before the first)."""
        return float(np.mean([c for _, c in self.lookups])) if self.lookups else 0.0

    @property
    def data(self) -> dict:
        return {"lookups": list(self.lookups), "amat": self.amat_measured, "latencies": self.latencies}

    def animate_lookup(self, address: int | None = None, hits: tuple[bool, ...] = (False, False, True), *, run_time: float = 3.0):
        """Play one memory access. ``hits[i]`` says whether level ``i`` holds the data; if none does it goes to RAM (needs ``ram=True``)."""
        hits = tuple(hits)
        if len(hits) != len(self.levels):
            raise ValueError(f"hits needs {len(self.levels)} entries, one per cache level")
        hit_level = next((i for i, h in enumerate(hits) if h), None)
        if hit_level is None:
            if not self.ram:
                raise ValueError("no level hits and there is no RAM to fall back on")
            hit_level = len(self.levels)
        cost = sum(self.latencies[: hit_level + 1])
        self.lookups.append((hit_level, cost))
        steps = hit_level + 2
        t = run_time / steps
        anims = []
        for j in range(hit_level + 1):
            blk = self.blocks[j + 1]
            ok = j == hit_level
            anims.append(AnimationGroup(self.token.animate.move_to(blk[0].get_center()),
                                        Indicate(blk[0], color=PALETTE.green if ok else PALETTE.red, scale_factor=1.08), run_time=t))
        anims.append(AnimationGroup(self.token.animate.move_to(self._home),
                                    Transform(self.readout, label(f"average access time: {self.amat_measured:.1f} cycles", 24,
                                                                   PALETTE.yellow).scale(self.unit).move_to(self.readout)), run_time=t))
        return Succession(*anims)


class _Bus(Framed, VGroup):
    """Parallel wires. A value is sent by flashing a pulse down every wire that carries a 1."""

    KIND, COLOR = "BUS", PALETTE.blue

    def __init__(self, width: int = 8, name: str | None = None, *, length: float = 4.0):
        super().__init__()
        self._init_frame()
        if not 1 <= width <= 16:
            raise ValueError("width must be between 1 and 16 bits")
        self.width_bits = width
        self.bus_name = name or f"{self.KIND}[{width - 1}:0]"
        gap = 0.2
        self.wires = VGroup(*[Line((-length / 2, (width - 1) / 2 * gap - i * gap, 0), (length / 2, (width - 1) / 2 * gap - i * gap, 0),
                                   color=self.COLOR, stroke_width=3, stroke_opacity=0.5) for i in range(width)])
        self.tag = label(self.bus_name, 22, self.COLOR).next_to(self.wires, LEFT, buff=0.3)
        self.value_label = label("--", 24, PALETTE.yellow).next_to(self.wires, RIGHT, buff=0.3)
        self.add(self.wires, self.tag, self.value_label)
        self.value = None

    def bits(self, value: int) -> list[int]:
        """The wires' levels, most significant bit on the top wire."""
        if not 0 <= value < (1 << self.width_bits):
            raise ValueError(f"value must fit in {self.width_bits} bits")
        return [(value >> (self.width_bits - 1 - i)) & 1 for i in range(self.width_bits)]

    @property
    def data(self) -> dict:
        return {"width": self.width_bits, "value": self.value}

    def animate_transfer(self, value: int, *, run_time: float = 1.5):
        """Drive ``value`` onto the bus: a bright pulse runs along every wire holding a 1, and the readout updates."""
        levels = self.bits(value)
        self.value = value
        digits = math.ceil(self.width_bits / 4)
        pulses = [ShowPassingFlash(w.copy().set_stroke(PALETTE.yellow, 7, opacity=1.0), time_width=0.5, run_time=run_time)
                  for w, b in zip(self.wires, levels) if b]
        new = label(f"0x{value:0{digits}X}", 24, PALETTE.yellow).scale(self.unit).move_to(self.value_label)
        parts = pulses + [Transform(self.value_label, new, run_time=run_time * 0.4)]
        return AnimationGroup(*parts, lag_ratio=0.0)


class AddressBus(_Bus):
    """Intuition: the CPU's pointing finger. One wire per address bit; a 1 means "high" on that wire, and together they name one memory location."""

    KIND, COLOR = "ADDR", PALETTE.teal


class DataBus(_Bus):
    """Intuition: a conveyor belt for values. One wire per data bit, carrying a word to or from memory."""

    KIND, COLOR = "DATA", PALETTE.blue


class RAMArray(VGroup):
    """Intuition: a grid of mailboxes. The row decoder picks a row, the column mux picks a box in it, and where they cross is the cell you read.

    ``highlight=(row, col)`` marks a cell at the start; ``animate_select`` plays the two-step lookup.
    """

    CELL = 0.5

    def __init__(self, rows: int = 4, cols: int = 4, highlight: tuple[int, int] | None = None):
        super().__init__()
        if not (1 <= rows <= 8 and 1 <= cols <= 8):
            raise ValueError("rows and cols must each be between 1 and 8")
        self.rows, self.cols = rows, cols
        c = self.CELL
        self.cells = {}
        grid = VGroup()
        for r in range(rows):
            for k in range(cols):
                sq = Rectangle(width=c, height=c, stroke_color=PALETTE.muted, stroke_width=2, fill_color=PALETTE.panel, fill_opacity=0.4)
                sq.move_to(((k - (cols - 1) / 2) * c, ((rows - 1) / 2 - r) * c, 0))
                self.cells[(r, k)] = sq
                grid.add(sq)
        row_tags = VGroup(*[label(f"r{r}", 18, PALETTE.muted).next_to(self.cells[(r, 0)], LEFT, buff=0.15) for r in range(rows)])
        col_tags = VGroup(*[label(f"c{k}", 18, PALETTE.muted).next_to(self.cells[(0, k)], UP, buff=0.12) for k in range(cols)])
        self.decoder = label("row decoder", 20, PALETTE.blue).next_to(row_tags, LEFT, buff=0.3)
        self.mux = label("column mux", 20, PALETTE.purple).next_to(col_tags, UP, buff=0.15)
        self.add(grid, row_tags, col_tags, self.decoder, self.mux)
        self.selected = None
        self.marker = Rectangle(width=c, height=c, stroke_color=PALETTE.yellow, stroke_width=5, fill_color=PALETTE.yellow, fill_opacity=0.35)
        self.marker.set_opacity(0.0)
        self.add(self.marker)
        self.move_to((0, 0, 0))
        if highlight is not None:
            self.select(*highlight)

    def _check(self, r: int, k: int) -> None:
        if not (0 <= r < self.rows and 0 <= k < self.cols):
            raise ValueError(f"cell ({r}, {k}) is outside a {self.rows} x {self.cols} array")

    def select(self, r: int, k: int) -> "RAMArray":
        """Mark a cell immediately (no animation)."""
        self._check(r, k)
        self.selected = (r, k)
        self.marker.move_to(self.cells[(r, k)].get_center()).set_stroke(opacity=1.0).set_fill(opacity=0.35)
        return self

    @property
    def data(self) -> dict:
        return {"rows": self.rows, "cols": self.cols, "selected": self.selected}

    def animate_select(self, r: int, k: int, *, run_time: float = 2.0):
        """Select a cell in two steps: the row decoder lights a row, then the column mux picks the box in it."""
        self._check(r, k)
        self.selected = (r, k)
        c = self.CELL
        t = run_time / 3
        row_bar = Rectangle(width=self.cols * c, height=c, stroke_width=0, fill_color=PALETTE.blue, fill_opacity=0.35)
        row_bar.move_to(VGroup(*[self.cells[(r, j)] for j in range(self.cols)]).get_center())
        col_bar = Rectangle(width=c, height=self.rows * c, stroke_width=0, fill_color=PALETTE.purple, fill_opacity=0.35)
        col_bar.move_to(VGroup(*[self.cells[(i, k)] for i in range(self.rows)]).get_center())
        self.marker.move_to(self.cells[(r, k)].get_center())
        return Succession(FadeIn(row_bar, run_time=t), FadeIn(col_bar, run_time=t),
                          AnimationGroup(FadeOut(row_bar), FadeOut(col_bar),
                                         self.marker.animate.set_stroke(opacity=1.0).set_fill(opacity=0.35), run_time=t))


class DMAController(VGroup):
    """Intuition: a courier who moves boxes between the loading dock and the warehouse so the manager (CPU) can keep working.

    Words travel peripheral -> bus -> memory under DMA control. The strip underneath shows who owns the bus each cycle
    (blue = CPU, yellow = DMA) for the chosen mode: ``burst`` hogs the bus until done, ``steal`` takes only the cycles the
    CPU leaves free (``cpu_busy`` is the fraction of cycles the CPU wants).
    """

    def __init__(self, words: int = 8, cpu_busy: float = 0.5, mode: str = "steal"):
        super().__init__()
        if not 1 <= words <= 8:
            raise ValueError("words must be between 1 and 8 for the picture")
        self.words, self.mode = words, mode
        self.schedule: DMASchedule = dma_schedule(words, cpu_busy, mode)

        def box(name: str, x: float, y: float, color: str, w: float = 2.2, h: float = 0.9) -> VGroup:
            r = Rectangle(width=w, height=h, stroke_color=color, stroke_width=3, fill_color=color, fill_opacity=0.15)
            return VGroup(r, label(name, 22, PALETTE.fg).move_to(r)).move_to((x, y, 0))

        self.cpu = box("CPU", -4.2, 1.7, PALETTE.blue)
        self.dma = box("DMA", 0.0, 1.7, PALETTE.yellow)
        self.mem = box("Memory", 4.2, 1.7, PALETTE.green, 3.0, 1.0)
        self.periph = box("Peripheral", 4.2, -1.1, PALETTE.orange, 3.0, 1.0)
        self.bus = Line((-5.6, 0.2, 0), (5.6, 0.2, 0), color=PALETTE.muted, stroke_width=6)
        bus_tag = label("system bus", 18, PALETTE.muted).move_to((-4.6, -0.2, 0))
        drops = VGroup(*[Line((b.get_center()[0], b.get_bottom()[1] if b is not self.periph else b.get_top()[1], 0), (b.get_center()[0], 0.2, 0),
                              color=PALETTE.muted, stroke_width=3) for b in (self.cpu, self.dma, self.mem, self.periph)])
        sw = 0.2
        self.slots = VGroup(*[Rectangle(width=sw, height=0.3, stroke_color=PALETTE.green, stroke_width=1.5, fill_opacity=0.0)
                              .move_to((self.mem.get_center()[0] + (i - (words - 1) / 2) * (sw + 0.06), self.mem.get_center()[1] - 0.15, 0))
                              for i in range(words)])
        self.tokens = VGroup(*[Rectangle(width=sw, height=0.3, stroke_width=0, fill_color=PALETTE.orange, fill_opacity=0.9)
                               .move_to((self.periph.get_center()[0] + (i - (words - 1) / 2) * (sw + 0.06), self.periph.get_center()[1] - 0.15, 0))
                               for i in range(words)])
        n = len(self.schedule.owners)
        cw = min(0.4, 8.0 / n)
        colors = {"CPU": PALETTE.blue, "DMA": PALETTE.yellow, "idle": PALETTE.muted}
        self.strip = VGroup(*[Rectangle(width=cw, height=0.4, stroke_width=1, stroke_color=PALETTE.bg, fill_color=colors[o], fill_opacity=0.7)
                              .move_to(((i - (n - 1) / 2) * cw, -2.3, 0)) for i, o in enumerate(self.schedule.owners)])
        self.strip_tag = label(f"bus owner per cycle ({mode})", 18, PALETTE.muted).next_to(self.strip, UP, buff=0.12)
        self.legend = VGroup(label("CPU", 18, PALETTE.blue), label("DMA", 18, PALETTE.yellow)).arrange(RIGHT, buff=0.4).next_to(self.strip, DOWN, buff=0.12)
        self.playhead = Line((0, -2.6, 0), (0, -2.0, 0), color=PALETTE.fg, stroke_width=4)
        self.playhead.move_to((self.strip[0].get_left()[0], -2.3, 0))
        self.add(self.bus, bus_tag, drops, self.cpu, self.dma, self.mem, self.periph, self.slots, self.tokens, self.strip, self.strip_tag,
                 self.legend, self.playhead)
        self.move_to((0, 0, 0))
        self._cw = cw

    @property
    def data(self) -> DMASchedule:
        return self.schedule

    def animate_transfer(self, *, run_time: float = 5.0):
        """Move every word from the peripheral to memory while the playhead sweeps the bus-ownership strip."""
        bus_y = self.bus.get_center()[1]
        moves = []
        for tok, slot in zip(self.tokens, self.slots):
            path = VMobject()
            path.set_points_as_corners([tok.get_center(), np.array([tok.get_center()[0], bus_y, 0]),
                                        np.array([slot.get_center()[0], bus_y, 0]), slot.get_center()])
            moves.append(MoveAlongPath(tok, path))
        end = np.array([self.strip[-1].get_right()[0], self.strip[0].get_center()[1], 0])
        return AnimationGroup(LaggedStart(*moves, lag_ratio=0.5),
                              Indicate(self.dma[0], color=PALETTE.yellow, scale_factor=1.05, run_time=run_time * 0.5),
                              self.playhead.animate(rate_func=lambda t: t).move_to(end),
                              run_time=run_time)
