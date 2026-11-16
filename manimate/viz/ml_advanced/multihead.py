"""Several heads side by side, then concatenated and mixed (README section 6.2)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from manim import DOWN, RIGHT, UP, AnimationGroup, Arrow, Group, Succession, Text, UpdateFromAlphaFunc

from ..core.grids import HeatmapGrid
from ..core.palette import PALETTE
from ._math import attention_head, init_weights, seeded_embeddings

MAX_HEADS = 4
MAX_MH_TOKENS = 8
HEAD_COLORS = ("yellow", "teal", "blue", "orange")


@dataclass(frozen=True)
class MultiHeadData:
    tokens: tuple[str, ...]
    head_weights: tuple[np.ndarray, ...]   # one (n, n) attention matrix per head
    concat: np.ndarray                     # (n, n_heads * d_head)
    Wo: np.ndarray
    output: np.ndarray                     # (n, d_model)


def _fade_in(mob, run_time):
    return UpdateFromAlphaFunc(mob, lambda m, a: m.set_opacity(a), run_time=run_time)


class MultiHeadAttention(Group):
    """Intuition: a panel of readers, each asking a different question of the same sentence. One head may look back at the
    previous word, another may chase a verb's subject. Their findings are laid side by side (concatenated) and mixed by
    one more matrix into a single new representation per word.
    """

    def __init__(self, tokens: list[str], n_heads: int = 3, *, causal: bool = False, d_model: int = 8, d_head: int = 4, seed: int = 7):
        super().__init__()
        n = len(tokens)
        if not 1 <= n <= MAX_MH_TOKENS:
            raise ValueError(f"MultiHeadAttention shows between 1 and {MAX_MH_TOKENS} tokens, got {n}")
        if not 2 <= n_heads <= MAX_HEADS:
            raise ValueError(f"n_heads must be between 2 and {MAX_HEADS}")
        if not 2 <= d_model <= 8 or not 1 <= d_head <= 4:
            raise ValueError("d_model must be 2..8 and d_head 1..4")
        self.tokens, self.n_heads, self.n = tuple(str(t) for t in tokens), n_heads, n
        X = seeded_embeddings(n, d_model, seed)
        heads = [attention_head(X, *(init_weights((d_model, d_head), seed=seed + 10 * h + k) for k in range(3)), causal=causal) for h in range(n_heads)]
        concat = np.concatenate([h.output for h in heads], axis=1)
        Wo = init_weights((n_heads * d_head, d_model), seed=seed + 99)
        out = concat @ Wo
        self._data = MultiHeadData(self.tokens, tuple(h.weights for h in heads), concat, Wo, out)
        cell = min(0.34, 2.2 / n)
        gap = 0.7
        self.head_grids: list[HeatmapGrid] = []
        self.head_group = Group()
        for h, d in enumerate(heads):
            g = HeatmapGrid(d.weights, cmap="viridis", vmin=0.0, vmax=1.0, cell=cell, show_values=False)
            self.head_grids.append(g)
        total_w = sum(g.width for g in self.head_grids) + gap * (n_heads - 1)
        x = -total_w / 2
        for h, g in enumerate(self.head_grids):
            g.move_to((x + g.width / 2, 1.5, 0))
            x += g.width + gap
            title = Text(f"head {h + 1}", font_size=20, color=getattr(PALETTE, HEAD_COLORS[h])).next_to(g, UP, buff=0.2)
            self.head_group.add(Group(g, title))
        ch = float(np.max(np.abs(concat))) or 1.0
        oh = float(np.max(np.abs(out))) or 1.0
        self.concat_grid = HeatmapGrid(concat, cmap="coolwarm", vmin=-ch, vmax=ch, cell=0.3, show_values=False).move_to((-2.4, -1.7, 0))
        self.out_grid = HeatmapGrid(out, cmap="coolwarm", vmin=-oh, vmax=oh, cell=0.3, show_values=False).move_to((3.0, -1.7, 0))
        self.arrow = Arrow(self.concat_grid.get_right() + RIGHT * 0.2, self.out_grid.get_left() + RIGHT * -0.2, buff=0, color=PALETTE.muted, stroke_width=4)
        self.mix_label = Text("× W_O", font_size=22, color=PALETTE.muted).next_to(self.arrow, UP, buff=0.1)
        self.concat_label = Text("heads side by side", font_size=20, color=PALETTE.muted).next_to(self.concat_grid, UP, buff=0.2)
        self.out_label = Text("output per word", font_size=20, color=PALETTE.green).next_to(self.out_grid, UP, buff=0.2)
        self.merge_group = Group(self.concat_grid, self.concat_label, self.arrow, self.mix_label, self.out_grid, self.out_label)
        self.add(self.head_group, self.merge_group)

    def animate_split(self, *, run_time: float = 3.0):
        """Bring in the heads one after another so the viewer sees each pattern on its own."""
        per = run_time / self.n_heads
        return Succession(*(_fade_in(sub, per) for sub in self.head_group))

    def animate_concat(self, *, run_time: float = 2.0):
        """Lay the heads' outputs side by side, then mix them into one output per word."""
        return Succession(_fade_in(Group(self.concat_grid, self.concat_label), run_time * 0.45),
                          _fade_in(Group(self.arrow, self.mix_label), run_time * 0.2),
                          _fade_in(Group(self.out_grid, self.out_label), run_time * 0.35))

    @property
    def data(self) -> MultiHeadData:
        return self._data
