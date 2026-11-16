"""Buck converter + inductor ramp + ratio curve.  Run: PYTHONPATH=. manim -ql -s examples/viz/power_demo.py PowerDemo (from renderer/)."""
from manim import Group, ValueTracker

from manimate.viz.core import VizScene, fade_in_whole
from manimate.viz.power import ConversionRatioCurve, ConverterTopology, InductorRamp


class PowerDemo(VizScene):
    duration = 8.0

    def script(self, ctx):
        duty = ValueTracker(0.4)
        buck = ConverterTopology("buck", vin=12, duty=0.4).bind(duty, "duty").scale(0.8).move_to((-3.6, 0.3, 0))
        ramp = InductorRamp(buck, width=3.4).scale(0.8).move_to((2.2, 0.2, 0))
        curve = ConversionRatioCurve("buck", duty, width=2.2, height=1.5).move_to((-3.7, -2.2, 0))
        fade_in_whole(ctx, Group(buck, ramp, curve), 1.0)
        ctx.play(buck.animate_switching(3), run_time=2.0)
        ctx.play(duty.animate.set_value(0.7), run_time=2.0)
        ctx.finish()
