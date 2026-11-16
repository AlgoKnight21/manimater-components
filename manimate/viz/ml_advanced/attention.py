"""Attention pictures: tokens, the score grid and the arcs view (README section 6.2)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from manim import (DOWN, UP, AnimationGroup, ArcBetweenPoints, Create, FadeIn, FadeOut, Group, Indicate, RoundedRectangle,
                   Succession, SurroundingRectangle, Text, Transform, VGroup, Wait)

from ..core.grids import MAX_CELL_GRID, SHOW_VALUES_UP_TO, HeatmapGrid
from ..core.palette import PALETTE, colormap, relative_luminance
from ._math import attention_scores, attention_weights

MAX_TOKENS = 12


def _check_tokens(tokens) -> list[str]:
    tokens = [str(t) for t in tokens]
    if not 1 <= len(tokens) <= MAX_TOKENS:
        raise ValueError(f"give between 1 and {MAX_TOKENS} tokens")
    return tokens


class TokenStrip(VGroup):
    """Intuition: a sentence cut into pieces: each token is a box, optionally with a tiny bar chart (its embedding) below."""

    def __init__(self, tokens, embeddings=None, highlight: int | None = None, *, font_size: int = 24):
        super().__init__()
        self.tokens = _check_tokens(tokens)
        emb = None if embeddings is None else np.asarray(embeddings, dtype=float)
        if emb is not None and (emb.ndim != 2 or emb.shape[0] != len(self.tokens)):
            raise ValueError("embeddings must be a 2-D array with one row per token")
        self.boxes: list[RoundedRectangle] = []
        self.labels: list[Text] = []
        self.bars: list[VGroup] = []
        x = 0.0
        for k, tok in enumerate(self.tokens):
            text = Text(tok, font_size=font_size, color=PALETTE.fg)
            w = max(0.9, text.width + 0.3)
            box = RoundedRectangle(width=w, height=0.6, corner_radius=0.1, color=PALETTE.blue, stroke_width=3)
            box.move_to((x + w / 2, 0, 0))
            text.move_to(box)
            x += w + 0.15
            self.boxes.append(box)
            self.labels.append(text)
            self.add(box, text)
            if emb is not None:
                self.bars.append(self._bars(emb[k], box))
                self.add(self.bars[-1])
        self.center()
        if highlight is not None:
            self.boxes[highlight].set_color(PALETTE.yellow)

    @staticmethod
    def _bars(row: np.ndarray, box) -> VGroup:
        peak = max(float(np.abs(row).max()), 1e-9)
        w = 0.5 / max(len(row), 1)
        bars = VGroup()
        for i, v in enumerate(row):
            h = max(0.04, 0.5 * abs(v) / peak)
            b = RoundedRectangle(width=w * 0.8, height=h, corner_radius=0.0, stroke_width=0, fill_opacity=1.0,
                                 fill_color=PALETTE.teal if v >= 0 else PALETTE.red)
            bars.add(b.move_to((box.get_center()[0] - 0.25 + w * (i + 0.5), box.get_bottom()[1] - 0.2 - 0.25 + h / 2, 0)))
        return bars

    def animate_highlight(self, i: int, run_time: float = 0.6):
        if not 0 <= i < len(self.boxes):
            raise ValueError("token index out of range")
        return AnimationGroup(Indicate(self.boxes[i], color=PALETTE.yellow), self.boxes[i].animate.set_color(PALETTE.yellow),
                              run_time=run_time)


@dataclass(frozen=True)
class AttentionData:
    raw: np.ndarray       # Q K^T
    scaled: np.ndarray    # divided by sqrt(d_k) when scale=True
    mask: np.ndarray      # True where the future is hidden
    weights: np.ndarray   # softmax of the masked, scaled scores


class AttentionHeatmap(Group):
    """Intuition: a who-looks-at-whom table. Each row is one word's budget of attention, shared out among the words it can see.

    It is a ``Group`` because its grid is a ``Group``. Stages, in order: raw scores, scaled, masked, softmax.
    """

    def __init__(self, Q, K, tokens=None, *, causal: bool = False, scale: bool = True, cmap: str = "viridis", cell: float = 0.55):
        super().__init__()
        raw = attention_scores(Q, K, causal=False, scale=False)
        if max(raw.shape) > MAX_CELL_GRID:
            raise ValueError(f"attention heatmaps support at most {MAX_CELL_GRID} tokens")
        scaled = attention_scores(Q, K, causal=False, scale=scale)
        mask = np.triu(np.ones(raw.shape, dtype=bool), k=1) if causal else np.zeros(raw.shape, dtype=bool)
        self.data = AttentionData(raw, scaled, mask, attention_weights(Q, K, causal=causal, scale=scale))
        self.cmap, self.causal, self.stage = cmap, causal, "raw"
        self._lo = float(min(raw.min(), scaled.min()))
        self._hi = float(max(raw.max(), scaled.max()))
        labels = _check_tokens(tokens) if tokens is not None else None
        self.grid = HeatmapGrid(raw, cmap=cmap, vmin=self._lo, vmax=self._hi, cell=cell, row_labels=labels, col_labels=labels,
                                show_values=False)
        self.add(self.grid)
        self.numbers: list[list[Text]] | None = None
        if max(raw.shape) <= SHOW_VALUES_UP_TO:
            state = self._state("raw")
            self.numbers = [[self._number(state[i][j][1], state[i][j][0], i, j) for j in range(raw.shape[1])] for i in range(raw.shape[0])]
            self.add(*(t for row in self.numbers for t in row))
        self._highlight: list = []

    def _state(self, stage: str):
        d = self.data
        out = []
        for i in range(d.raw.shape[0]):
            row = []
            for j in range(d.raw.shape[1]):
                if stage == "softmax":
                    row.append((colormap(self.cmap, d.weights[i, j], 0.0, 1.0), f"{d.weights[i, j]:.2f}"))
                elif stage == "masked" and d.mask[i, j]:
                    row.append((PALETTE.bg, "-inf"))
                else:
                    v = d.raw[i, j] if stage == "raw" else d.scaled[i, j]
                    row.append((colormap(self.cmap, v, self._lo, self._hi), f"{v:.1f}"))
            out.append(row)
        return out

    def _number(self, text: str, fill: str, i: int, j: int) -> Text:
        dark = relative_luminance(fill) > 0.4
        return Text(text, font_size=max(12, int(self.grid.cell * 32)), color=PALETTE.bg if dark else PALETTE.fg).move_to(self.grid.cells[i][j])

    def _go(self, stage: str, run_time: float):
        state = self._state(stage)
        anims = []
        for i, row in enumerate(state):
            for j, (fill, text) in enumerate(row):
                anims.append(self.grid.cells[i][j].animate.set_fill(fill))
                if self.numbers is not None:
                    anims.append(Transform(self.numbers[i][j], self._number(text, fill, i, j)))
        self.stage = stage
        return AnimationGroup(*anims, run_time=run_time)

    def animate_scale(self, run_time: float = 1.0):
        return self._go("scaled", run_time)

    def animate_mask(self, run_time: float = 1.0):
        if not self.causal:
            return Wait(run_time)  # nothing is hidden, so nothing to show
        return self._go("masked", run_time)

    def animate_softmax(self, run_time: float = 1.2):
        return self._go("softmax", run_time)

    def animate_dot_product(self, i: int, j: int, run_time: float = 1.2):
        """Light query row ``i`` and key column ``j``, show the multiply-and-sum, pulse the cell."""
        n_q, n_k = self.data.raw.shape
        if not (0 <= i < n_q and 0 <= j < n_k):
            raise ValueError("cell index out of range")
        row = SurroundingRectangle(VGroup(*self.grid.cells[i]), color=PALETTE.yellow, buff=0.04)
        col = SurroundingRectangle(VGroup(*(r[j] for r in self.grid.cells)), color=PALETTE.blue, buff=0.04)
        note = Text(f"q{i} . k{j} = {self.data.raw[i, j]:.2f}", font_size=24, color=PALETTE.yellow).next_to(self.grid, DOWN, buff=0.35)
        self._highlight = [row, col, note]
        return Succession(FadeIn(row), FadeIn(col), FadeIn(note), Indicate(self.grid.cells[i][j], color=PALETTE.yellow), run_time=run_time)

    def animate_clear(self, run_time: float = 0.4):
        items, self._highlight = self._highlight, []
        return FadeOut(*items, run_time=run_time) if items else Wait(run_time)


class AttentionArcs(VGroup):
    """Intuition: one word reaching out to the others: a thicker, brighter arc means it listens harder."""

    def __init__(self, tokens, weights, query_index: int, top_k: int | None = None):
        super().__init__()
        self.strip = TokenStrip(tokens)
        w = np.asarray(weights, dtype=float)
        if w.ndim != 1 or len(w) != len(self.strip.tokens) or np.any(w < 0):
            raise ValueError("weights must be one non-negative number per token")
        if not 0 <= query_index < len(w):
            raise ValueError("query_index out of range")
        self.query_index, self.weights = query_index, w
        order = [j for j in np.argsort(-w) if j != query_index]
        self.shown = sorted(order[:top_k] if top_k else order)
        peak = max(float(w[self.shown].max()) if self.shown else 1.0, 1e-9)
        self.arcs: list[ArcBetweenPoints] = []
        src = self.strip.boxes[query_index].get_top()
        for j in self.shown:
            dst = self.strip.boxes[j].get_top()
            a = ArcBetweenPoints(src, dst, angle=-1.6 if dst[0] > src[0] else 1.6, color=PALETTE.yellow,
                                 stroke_width=1.5 + 9 * w[j] / peak, stroke_opacity=0.25 + 0.75 * w[j] / peak)
            self.arcs.append(a)
        self.strip.boxes[query_index].set_color(PALETTE.yellow)
        self.add(self.strip, *self.arcs)

    def animate_in(self, run_time: float = 1.2):
        return AnimationGroup(*(Create(a) for a in self.arcs), lag_ratio=0.2, run_time=run_time)
