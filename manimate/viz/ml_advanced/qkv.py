"""X through three lenses: the Q, K and V projections drawn as matrices (README section 6.2)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, AnimationGroup, Group, Indicate, Succession, SurroundingRectangle, Text, ValueTracker,
                   VGroup)

from ...ctx import STAGE_H, STAGE_W
from ..core.grids import HeatmapGrid
from ..core.palette import PALETTE
from ._math import init_weights, qkv_project

MAX_QKV_TOKENS = 6
MAX_QKV_D_MODEL = 6
MAX_QKV_D_HEAD = 4
CELL = 0.24


@dataclass(frozen=True)
class QKVData:
    X: np.ndarray
    Wq: np.ndarray
    Wk: np.ndarray
    Wv: np.ndarray
    Q: np.ndarray
    K: np.ndarray
    V: np.ndarray


def _grid(values: np.ndarray) -> HeatmapGrid:
    hi = float(np.max(np.abs(values))) or 1.0
    return HeatmapGrid(values, cmap="coolwarm", vmin=-hi, vmax=hi, cell=CELL, show_values=False)


class QKVProjection(Group):
    """Intuition: one token, three roles. The same row of X is turned into a question (Q), a label others can match
    against (K) and a payload to hand over (V) by three different matrices: a matrix product is a change of viewpoint.

    It is a ``Group`` because its grids are ``HeatmapGrid`` objects (which are ``Group`` too).
    """

    def __init__(self, X, d_head: int, seed: int = 7, weights=None):
        super().__init__()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D: one row per token")
        n, d = X.shape
        if n > MAX_QKV_TOKENS or d > MAX_QKV_D_MODEL:
            raise ValueError(f"QKVProjection draws at most {MAX_QKV_TOKENS} tokens x {MAX_QKV_D_MODEL} features, got {n} x {d}")
        if not 1 <= d_head <= MAX_QKV_D_HEAD:
            raise ValueError(f"d_head must be between 1 and {MAX_QKV_D_HEAD}")
        if weights is None:
            weights = tuple(init_weights((d, d_head), seed=seed + k) for k in range(3))
        Wq, Wk, Wv = (np.asarray(w, dtype=float) for w in weights)
        Q, K, V = qkv_project(X, Wq, Wk, Wv)
        self._data = QKVData(X, Wq, Wk, Wv, Q, K, V)
        self.n_tokens = n
        self.x_grid = _grid(X)
        self.w_grids = {k: _grid(w) for k, w in (("q", Wq), ("k", Wk), ("v", Wv))}
        self.out_grids = {k: _grid(a) for k, a in (("q", Q), ("k", K), ("v", V))}
        self.parts = Group(self.x_grid)
        self._rows: list[VGroup] = []
        row_h = max(d, n) * CELL
        gap = 0.55
        total = 3 * row_h + 2 * gap
        names = {"q": "Q", "k": "K", "v": "V"}
        self.x_grid.move_to((-3.6, 0, 0))
        self.parts.add(Text("X (tokens)", font_size=20, color=PALETTE.muted).next_to(self.x_grid, UP, buff=0.3))
        for r, key in enumerate("qkv"):
            y = total / 2 - row_h / 2 - r * (row_h + gap)
            w, o = self.w_grids[key], self.out_grids[key]
            w.move_to((-0.4, y, 0))
            o.move_to((2.8, y, 0))
            times = Text("×", font_size=30, color=PALETTE.muted).move_to(((self.x_grid.get_right()[0] + w.get_left()[0]) / 2, y, 0))
            eq = Text("=", font_size=30, color=PALETTE.muted).move_to(((w.get_right()[0] + o.get_left()[0]) / 2, y, 0))
            wl = Text(f"W_{names[key]}", font_size=20, color=PALETTE.muted).next_to(w, UP, buff=0.12)
            ol = Text(names[key], font_size=26, color=PALETTE.yellow if key == "q" else PALETTE.teal if key == "k" else PALETTE.blue).next_to(o, RIGHT, buff=0.25)
            self.parts.add(w, o, times, eq, wl, ol)
        self.add(self.parts)
        self.move_to((0, 0, 0))
        if self.height > STAGE_H - 0.1 or self.width > STAGE_W - 0.1:  # the biggest case would clip the stage
            self.scale(min((STAGE_H - 0.1) / self.height, (STAGE_W - 0.1) / self.width))
        self.cursor = VGroup(*(SurroundingRectangle(self._row_group(g, 0), color=PALETTE.yellow, buff=0.04, stroke_width=3)
                               for g in (self.x_grid, *self.out_grids.values())))
        self.cursor.set_stroke(opacity=0.0)
        self.add(self.cursor)

    def _row_group(self, grid: HeatmapGrid, i: int) -> VGroup:
        return VGroup(*grid.cells[i])

    def _check(self, i: int) -> int:
        if not 0 <= i < self.n_tokens:
            raise ValueError(f"token index {i} is outside 0..{self.n_tokens - 1}")
        return i

    def set_token(self, i: int) -> "QKVProjection":
        """Move the yellow cursor to row ``i`` of X and of Q, K and V right now."""
        self._check(i)
        for rect, grid in zip(self.cursor, (self.x_grid, *self.out_grids.values())):
            rect.become(SurroundingRectangle(self._row_group(grid, i), color=PALETTE.yellow, buff=0.04, stroke_width=3))
        return self

    def animate_project(self, *, run_time: float = 3.0):
        """Sweep every token row of X through the three matrices and light the matching rows of Q, K and V."""
        per = run_time / self.n_tokens
        steps = [AnimationGroup(Indicate(self._row_group(self.x_grid, i), color=PALETTE.yellow),
                                *(Indicate(self._row_group(g, i), color=PALETTE.yellow) for g in self.out_grids.values()),
                                run_time=per) for i in range(self.n_tokens)]
        return Succession(*steps)

    def bind(self, tracker: ValueTracker, param: str = "token") -> "QKVProjection":
        """Keep the cursor on the token index held by the tracker."""
        if param != "token":
            raise ValueError("QKVProjection can only bind 'token'")
        last = {"i": -1}

        def follow(m):
            i = int(round(min(max(tracker.get_value(), 0), m.n_tokens - 1)))
            if i != last["i"]:
                last["i"] = i
                m.set_token(i)

        self.add_updater(follow)
        return self

    @property
    def data(self) -> QKVData:
        return self._data
