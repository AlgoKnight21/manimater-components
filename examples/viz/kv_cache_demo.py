"""KV cache frames.  Run:  manim -ql -s examples/viz/kv_cache_demo.py CacheDemo  (or TimelineDemo), from renderer/."""
from manim import Group

from manimate.ctx import Ctx
from manimate.viz.core import VizScene, fade_in_whole
from manimate.viz.ml_advanced import DecodeTimeline, KVCacheBuffer


class CacheDemo(VizScene):
    duration = 6.0

    def script(self, ctx: Ctx) -> None:
        fade_in_whole(ctx, Group(KVCacheBuffer(8, window=4).set_step(6)), 1.0)
        ctx.finish()


class TimelineDemo(VizScene):
    duration = 6.0

    def script(self, ctx: Ctx) -> None:
        fade_in_whole(ctx, Group(DecodeTimeline(10)), 1.0)
        ctx.finish()
