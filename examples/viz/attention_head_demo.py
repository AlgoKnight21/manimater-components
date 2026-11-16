"""One attention head, three stages.  Run:  manim -ql -s examples/viz/attention_head_demo.py HeadOutput  (from renderer/)."""
from manimate.ctx import Ctx
from manimate.viz.core import VizScene, fade_in_whole
from manimate.viz.ml_advanced import AttentionHead

TOKENS = ["The", "cat", "sat", "on", "mat"]


class HeadOutput(VizScene):
    duration = 8.0

    def script(self, ctx: Ctx) -> None:
        head = AttentionHead(TOKENS, causal=True)
        ctx.fit(head)
        fade_in_whole(ctx, head, 1.0)
        ctx.play(head.animate_to("scores"), run_time=1.4)
        ctx.play(head.animate_to("output"), run_time=1.4)
        ctx.finish()
