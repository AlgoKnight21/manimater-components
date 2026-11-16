"""Multi-head attention frame.  Run:  manim -ql -s examples/viz/multihead_demo.py MultiHeadDemo  (from renderer/)."""
from manimate.ctx import Ctx
from manimate.viz.core import VizScene, fade_in_whole
from manimate.viz.ml_advanced import MultiHeadAttention


class MultiHeadDemo(VizScene):
    duration = 8.0

    def script(self, ctx: Ctx) -> None:
        mh = MultiHeadAttention(["The", "cat", "sat", "on", "mat"], n_heads=3, causal=True)
        ctx.fit(mh)
        fade_in_whole(ctx, mh, 1.0)
        ctx.play(mh.animate_split(), run_time=3.0)
        ctx.play(mh.animate_concat(), run_time=2.0)
        ctx.finish()
