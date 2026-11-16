"""One frame of the positional-encoding waves.  Run:  manim -ql -s examples/viz/positional_demo.py WavesDemo   (from renderer/).

``base`` is shrunk from the usual 10000 so the slow waves visibly bend inside 12 positions; the maths is unchanged."""
from manim import Group

from manimate.ctx import Ctx
from manimate.viz.core import VizScene, fade_in_whole
from manimate.viz.ml_advanced import PositionalEncodingWaves


class WavesDemo(VizScene):
    duration = 6.0

    def script(self, ctx: Ctx) -> None:
        w = PositionalEncodingWaves(d_model=8, n_positions=12, n_freqs=4, base=60.0, position=4.0)
        fade_in_whole(ctx, Group(w), 1.0)
        ctx.finish()


class SumDemo(VizScene):
    """manim -ql -s examples/viz/positional_demo.py SumDemo"""
    duration = 6.0

    def script(self, ctx: Ctx) -> None:
        from manimate.viz.ml_advanced import PositionalEncodingSum
        fade_in_whole(ctx, Group(PositionalEncodingSum(["The", "cat", "sat", "on", "mat"], 8)), 1.0)
        ctx.finish()


class PEHeatmapDemo(VizScene):
    """manim -ql -s examples/viz/positional_demo.py PEHeatmapDemo"""
    duration = 6.0

    def script(self, ctx: Ctx) -> None:
        from manimate.viz.ml_advanced import PEHeatmap
        h = PEHeatmap(20, 24, base=500.0).set_dim(2).set_position(8)
        fade_in_whole(ctx, Group(h), 1.0)
        ctx.finish()
