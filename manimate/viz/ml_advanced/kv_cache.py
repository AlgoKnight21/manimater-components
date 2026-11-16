"""The KV cache as a row of slots, and the saved work as a timeline (README section 6.2)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, AnimationGroup, Indicate, Rectangle, RoundedRectangle, Square, Succession, Text, Transform, UpdateFromAlphaFunc, VGroup)

from ..core.palette import PALETTE
from ._math import KVStep, MAX_TOKENS, kv_cache_steps

SLOT_W, SLOT_H = 0.95, 1.35


@dataclass(frozen=True)
class KVCacheBufferData:
    n_slots: int
    window: int | None
    contents: tuple[int | None, ...]   # token index held by each slot, or None when empty


class KVCacheBuffer(VGroup):
    """Intuition: a notebook of what every earlier word offered. Each slot keeps one word's key (top half) and value (bottom
    half), so the model reads them back instead of recomputing them. With a window the notebook is a ring: when it is full,
    the oldest page is overwritten.
    """

    def __init__(self, n_tokens: int, window: int | None = None):
        super().__init__()
        if not 1 <= n_tokens <= MAX_TOKENS:
            raise ValueError(f"n_tokens must be between 1 and {MAX_TOKENS}")
        if window is not None and not 1 <= window <= n_tokens:
            raise ValueError(f"window must be between 1 and {n_tokens}")
        self.n_tokens, self.window = n_tokens, window
        self.n_slots = window or n_tokens
        self.steps = kv_cache_steps(n_tokens, window)
        self._contents: list[int | None] = [None] * self.n_slots
        self.slots: list[VGroup] = []
        for s in range(self.n_slots):
            box = RoundedRectangle(corner_radius=0.08, width=SLOT_W, height=SLOT_H, stroke_color=PALETTE.muted, stroke_width=3, fill_color=PALETTE.panel, fill_opacity=1.0)
            k = Rectangle(width=SLOT_W - 0.16, height=SLOT_H / 2 - 0.12, stroke_width=0, fill_color=PALETTE.teal, fill_opacity=0.0).move_to(box).shift(UP * SLOT_H / 4)
            v = Rectangle(width=SLOT_W - 0.16, height=SLOT_H / 2 - 0.12, stroke_width=0, fill_color=PALETTE.blue, fill_opacity=0.0).move_to(box).shift(DOWN * SLOT_H / 4)
            kt = Text("K", font_size=18, color=PALETTE.bg).move_to(k).set_opacity(0.0)
            vt = Text("V", font_size=18, color=PALETTE.bg).move_to(v).set_opacity(0.0)
            slot = VGroup(box, k, v, kt, vt)
            slot.move_to((s * (SLOT_W + 0.12), 0, 0))
            self.slots.append(slot)
            self.add(slot)
        self.labels = [self._label(s, None) for s in range(self.n_slots)]
        self.add(*self.labels)
        self.center()
        for s in range(self.n_slots):
            self.labels[s].move_to(self.slots[s][0].get_bottom() + DOWN * 0.3)
        if self.width > 11.0:
            self.scale_to_fit_width(11.0)

    def _label(self, s: int, tok: int | None) -> Text:
        t = Text(f"t{tok}" if tok is not None else "t0", font_size=20, color=PALETTE.muted)
        t.move_to(self.slots[s][0].get_bottom() + DOWN * 0.3)
        if tok is None:
            t.set_opacity(0.0)
        return t

    def slot_of(self, step: int) -> int:
        """Which slot token ``step`` is written to (a ring when there is a window)."""
        if not 0 <= step < self.n_tokens:
            raise ValueError(f"step {step} is outside 0..{self.n_tokens - 1}")
        return step % self.window if self.window else step

    def _fill(self, s: int, filled: bool):
        _, k, v, kt, vt = self.slots[s]
        return [k.animate.set_fill(opacity=1.0 if filled else 0.0), v.animate.set_fill(opacity=1.0 if filled else 0.0),
                kt.animate.set_opacity(1.0 if filled else 0.0), vt.animate.set_opacity(1.0 if filled else 0.0)]

    def set_step(self, step: int) -> "KVCacheBuffer":
        """Jump to the state after token ``step`` was processed, with no animation."""
        self.slot_of(step)
        contents: list[int | None] = [None] * self.n_slots
        for t in self.steps[step].attended:
            contents[self.slot_of(t)] = t
        for s, tok in enumerate(contents):
            _, k, v, kt, vt = self.slots[s]
            k.set_fill(opacity=1.0 if tok is not None else 0.0)
            v.set_fill(opacity=1.0 if tok is not None else 0.0)
            for t in (kt, vt):
                t.set_opacity(1.0 if tok is not None else 0.0)
            new = self._label(s, tok)
            self.labels[s].become(new)
        self._contents = contents
        return self

    def animate_step(self, step: int, *, run_time: float = 1.0):
        """Process token ``step``: if the window is full flash the oldest slot out, then write the new K and V."""
        s = self.slot_of(step)
        info: KVStep = self.steps[step]
        anims = []
        if info.evicted is not None:
            anims.append(Indicate(self.slots[s][0], color=PALETTE.red, scale_factor=1.12))
        anims += self._fill(s, True)
        anims.append(Transform(self.labels[s], self._label(s, step).set_opacity(1.0)))
        self._contents[s] = step
        return AnimationGroup(*anims, run_time=run_time, lag_ratio=0.0)

    @property
    def data(self) -> KVCacheBufferData:
        return KVCacheBufferData(self.n_slots, self.window, tuple(self._contents))


@dataclass(frozen=True)
class DecodeTimelineData:
    steps: tuple[KVStep, ...]
    total_no_cache: int
    total_with_cache: int


class DecodeTimeline(VGroup):
    """Intuition: two ways to write a sentence one word at a time. Without a cache the model re-reads and re-derives
    everything it already knew at every step, so the pile of work grows like a staircase; with a cache each step adds one
    small block. The gap between the two shapes is the saving.
    """

    def __init__(self, n_tokens: int, window: int | None = None):
        super().__init__()
        if not 1 <= n_tokens <= MAX_TOKENS:
            raise ValueError(f"n_tokens must be between 1 and {MAX_TOKENS}")
        self.n_tokens = n_tokens
        self.steps = kv_cache_steps(n_tokens, window)
        self.total_no_cache = sum(s.work_without_cache for s in self.steps)
        self.total_with_cache = sum(s.work_with_cache for s in self.steps)
        q = 0.2
        dx = min(0.62, 10.0 / n_tokens)
        self.columns: list[VGroup] = []
        top = VGroup()
        for t, st in enumerate(self.steps):
            col = VGroup(*(Square(side_length=q, stroke_width=0.5, stroke_color=PALETTE.bg, fill_color=PALETTE.orange, fill_opacity=1.0) for _ in range(st.work_without_cache)))
            col.arrange(UP, buff=0.02)
            top.add(col)
        for t, col in enumerate(top):
            col.move_to((t * dx, 0, 0))
        base_y = 1.0
        for col in top:
            col.shift(UP * (base_y - col.get_bottom()[1]))
        bottom = VGroup(*(Square(side_length=q, stroke_width=0.5, stroke_color=PALETTE.bg, fill_color=PALETTE.green, fill_opacity=1.0).move_to((t * dx, -1.4, 0)) for t in range(n_tokens)))
        self.top, self.bottom = top, bottom
        self.columns = [VGroup(top[t], bottom[t]) for t in range(n_tokens)]
        body = VGroup(top, bottom)
        self.add(body)
        self.center()
        x_left = self.get_left()[0]
        self.add(Text("without a cache", font_size=20, color=PALETTE.orange).move_to((x_left, top.get_top()[1] + 0.35, 0), aligned_edge=LEFT),
                 Text("with a cache", font_size=20, color=PALETTE.green).move_to((x_left, bottom.get_top()[1] + 0.45, 0), aligned_edge=LEFT))
        self.totals = Text(f"K,V rows computed:  {self.total_no_cache}  vs  {self.total_with_cache}", font_size=24, color=PALETTE.fg).next_to(bottom, DOWN, buff=0.5)
        self.add(self.totals)

    def animate_play(self, *, run_time: float = 4.0):
        """Decode one token at a time: both piles grow column by column."""
        per = run_time / self.n_tokens
        steps = []
        for col in self.columns:
            parts = []
            for mob in col.family_members_with_points():
                parts.append(UpdateFromAlphaFunc(mob, lambda m, a: m.set_fill(opacity=a).set_stroke(opacity=a)))
            steps.append(AnimationGroup(*parts, run_time=per))
        return Succession(*steps)

    @property
    def data(self) -> DecodeTimelineData:
        return DecodeTimelineData(tuple(self.steps), self.total_no_cache, self.total_with_cache)
