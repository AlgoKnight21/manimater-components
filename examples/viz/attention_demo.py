"""Attention heatmap + arcs.  Run:  PYTHONPATH=. manim -ql -s examples/viz/attention_demo.py AttentionDemo  (from renderer/)."""
import numpy as np
from manim import Group

from manimate.ctx import Ctx
from manimate.viz.core import VizScene, fade_in_whole
from manimate.viz.ml_advanced._math import attention_weights
from manimate.viz.ml_advanced.attention import AttentionArcs, AttentionHeatmap


class AttentionDemo(VizScene):
    duration = 12.0

    def script(self, ctx: Ctx) -> None:
        toks = ["The", "cat", "sat", "down"]
        g = np.random.default_rng(5)
        Q, K = g.normal(size=(4, 4)), g.normal(size=(4, 4))
        heat = AttentionHeatmap(Q, K, toks, causal=True).move_to((-3.2, 0.6, 0))
        arcs = AttentionArcs(toks, attention_weights(Q, K, causal=True)[3], query_index=3).scale(0.8).move_to((3.2, 0.6, 0))
        fade_in_whole(ctx, Group(heat, arcs), 1.2)
        ctx.play(heat.animate_scale(), run_time=1.0)
        ctx.play(heat.animate_mask(), run_time=1.0)
        ctx.play(heat.animate_softmax(), run_time=1.2)
        ctx.play(arcs.animate_in(), run_time=1.2)
        ctx.finish()
