"""One frame with most of viz.core.  Run:  manim -ql -s examples/viz/core_demo.py CoreDemo   (from renderer/)."""
import numpy as np
from manim import Axes, Group, ValueTracker

from manimate.ctx import Ctx
from manimate.viz.core import (AreaBand, Block, FlowField, GlowDot, HeatmapGrid, LiveCurve, PALETTE, Slider, Sphere2p5D,
                               Surface2p5D, VizScene, Wire, fade_in_whole)


class CoreDemo(VizScene):
    duration = 6.0

    def script(self, ctx: Ctx) -> None:
        t = ValueTracker(0.5)
        heat = HeatmapGrid(np.random.default_rng(1).random((4, 4)), row_labels=list("abcd")).scale(0.8).move_to((-4.6, 1.6, 0))
        ax = Axes(x_range=[0, 4, 1], y_range=[0, 4, 1], x_length=3.4, y_length=2.2).move_to((-0.6, 1.8, 0))
        curve = LiveCurve(ax, lambda x, k: 4 * k * np.sin(x) ** 2, tracker=t)
        band = AreaBand(ax, lambda x: 0.5, lambda x: 1.5)
        slider = Slider(t, (0, 1), "gain").move_to((-0.6, -0.1, 0))
        surf = Surface2p5D(lambda u, v: np.exp(-(u * u + v * v)), resolution=(18, 18), projector=None).scale(0.8).move_to((4.4, 1.5, 0))
        globe = Sphere2p5D(0.9).move_to((4.4, -1.6, 0))
        a, b = Block("Plant").move_to((-4.4, -1.6, 0)), Block("Control").move_to((-1.6, -1.6, 0))
        wire = Wire.route(a.port("right"), b.port("left"), arrow=True, color=PALETTE.teal)
        flow = FlowField(wire[0], n=10)
        dot = GlowDot((1.2, -1.6, 0), 0.12)
        frame = [heat, ax, curve, band, slider, surf, globe, a, b, wire, dot]
        fade_in_whole(ctx, Group(*frame), 1.0)  # Group, not VGroup: HeatmapGrid can hold an image
        ctx.add(flow)
        ctx.finish()
