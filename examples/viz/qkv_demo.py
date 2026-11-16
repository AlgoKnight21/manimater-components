"""One frame of the QKV projection.  Run:  manim -ql -s examples/viz/qkv_demo.py QKVDemo   (from renderer/)."""
from manim import Group

from manimate.ctx import Ctx
from manimate.viz.core import VizScene, fade_in_whole
from manimate.viz.ml_advanced import QKVProjection
from manimate.viz.ml_advanced._math import seeded_embeddings


class QKVDemo(VizScene):
    duration = 6.0

    def script(self, ctx: Ctx) -> None:
        q = QKVProjection(seeded_embeddings(5, 6), d_head=3).set_token(1)
        fade_in_whole(ctx, Group(q), 1.0)
        ctx.finish()
