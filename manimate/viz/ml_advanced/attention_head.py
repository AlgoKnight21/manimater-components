"""One whole attention head, shown a stage at a time: project -> scores -> output (README section 6.2)."""
from __future__ import annotations

import numpy as np
from manim import UP, Group, Text, UpdateFromAlphaFunc

from ..core.grids import HeatmapGrid
from ..core.palette import PALETTE
from ._math import HeadData, attention_head, init_weights, seeded_embeddings
from .qkv import MAX_QKV_D_HEAD, MAX_QKV_D_MODEL, MAX_QKV_TOKENS, QKVProjection

STAGES = ("project", "scores", "output")


class _OutputView(Group):
    """weights x V = output, drawn as three small matrices."""

    def __init__(self, data: HeadData):
        super().__init__()
        hi = float(np.max(np.abs(data.V))) or 1.0
        w = HeatmapGrid(data.weights, cmap="viridis", vmin=0.0, vmax=1.0, cell=0.45, show_values=False)
        v = HeatmapGrid(data.V, cmap="coolwarm", vmin=-hi, vmax=hi, cell=0.45, show_values=False)
        o = HeatmapGrid(data.output, cmap="coolwarm", vmin=-hi, vmax=hi, cell=0.45, show_values=False)
        w.move_to((-3.6, 0, 0)), v.move_to((0.0, 0, 0)), o.move_to((3.6, 0, 0))
        self.add(w, v, o)
        for g, name, color in ((w, "weights", PALETTE.yellow), (v, "V", PALETTE.blue), (o, "output", PALETTE.green)):
            self.add(Text(name, font_size=22, color=color).next_to(g, UP, buff=0.3))
        self.add(Text("×", font_size=34, color=PALETTE.muted).move_to((-1.8, 0, 0)), Text("=", font_size=34, color=PALETTE.muted).move_to((1.8, 0, 0)))


class AttentionHead(Group):
    """Intuition: one reader with one question. It turns every word into a query, a key and a value, scores every query
    against every key, and uses those scores to blend the values. This view shows the three stages one at a time so the
    viewer can follow a single thread from words to a new, context-aware word.
    """

    def __init__(self, tokens: list[str], *, causal: bool = False, d_model: int = 6, d_head: int = 4, seed: int = 7, stage: str = "project"):
        super().__init__()
        n = len(tokens)
        if not 1 <= n <= MAX_QKV_TOKENS:
            raise ValueError(f"AttentionHead shows between 1 and {MAX_QKV_TOKENS} tokens, got {n}")
        if not 1 <= d_model <= MAX_QKV_D_MODEL or not 1 <= d_head <= MAX_QKV_D_HEAD:
            raise ValueError(f"d_model must be 1..{MAX_QKV_D_MODEL} and d_head 1..{MAX_QKV_D_HEAD}")
        if stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
        self.tokens, self.causal = tuple(str(t) for t in tokens), causal
        X = seeded_embeddings(n, d_model, seed)
        Ws = tuple(init_weights((d_model, d_head), seed=seed + k) for k in range(3))
        self._data = attention_head(X, *Ws, causal=causal)
        self.views = {"project": QKVProjection(X, d_head, weights=Ws),
                      "scores": HeatmapGrid(self._data.weights, cmap="viridis", vmin=0.0, vmax=1.0, cell=0.6, row_labels=list(tokens), col_labels=list(tokens)),
                      "output": _OutputView(self._data)}
        self.stage = stage
        self.slot = Group(self.views[stage])
        self.add(self.slot)

    def show(self, stage: str) -> "AttentionHead":
        """Switch to ``stage`` right now."""
        if stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
        self.slot.submobjects = [self.views[stage]]
        self.stage = stage
        return self

    def animate_to(self, stage: str, *, run_time: float = 1.4):
        """Fade the current stage out and the next one in."""
        if stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
        if stage == self.stage:
            raise ValueError(f"the head is already showing {stage!r}")
        # One alpha-driven animation: fade the current view out, swap at the midpoint, fade the next view in.
        # Each element's own opacity is remembered so things that start hidden (a cursor, a row sum) stay hidden.
        def snapshot(view):
            return [(m, m.get_fill_opacity(), m.get_stroke_opacity()) for m in view.family_members_with_points()]

        saved = {self.stage: snapshot(self.views[self.stage]), stage: snapshot(self.views[stage])}
        state = {"swapped": False}

        def fade(view_name, f):
            for m, fo, so in saved[view_name]:
                m.set_fill(opacity=fo * f).set_stroke(opacity=so * f)

        def step(mob, alpha):
            if alpha < 0.5:
                fade(self.stage if not state["swapped"] else stage, 1 - 2 * alpha)
            else:
                if not state["swapped"]:
                    fade(self.stage, 1.0)  # leave the outgoing view exactly as it was
                    self.show(stage)
                    state["swapped"] = True
                fade(stage, 2 * alpha - 1)

        return UpdateFromAlphaFunc(self.slot, step, run_time=run_time, rate_func=lambda t: t)

    @property
    def data(self) -> HeadData:
        return self._data
