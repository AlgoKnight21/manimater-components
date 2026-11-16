"""Machines, stability and grid.  Run: PYTHONPATH=. manim -ql -s examples/viz/power_machines_demo.py MachinesDemo (from renderer/)."""
from manim import Group, ValueTracker

from manimate.viz.core import VizScene, fade_in_whole
from manimate.viz.power import DQFrame, EqualAreaCriterion, FaultCurrentTransient, PowerGrid, RotatingMagneticField, TorqueDecomposition


class MachinesDemo(VizScene):
    duration = 8.0

    def script(self, ctx):
        th = ValueTracker(0.6)
        rmf = RotatingMagneticField().scale(0.8).move_to((-5.0, 1.6, 0))
        dq = DQFrame(th).scale(0.6).move_to((-2.0, 1.6, 0))
        tq = TorqueDecomposition(0.3, 0.9, size=1.2).scale(0.8).move_to((1.2, 1.5, 0))
        ea = EqualAreaCriterion().scale(0.55).move_to((-4.6, -1.7, 0))
        fc = FaultCurrentTransient(width=4.6, height=1.9).scale(0.7).move_to((0.1, -1.6, 0))
        grid = PowerGrid.radial(3).scale(0.42).move_to((4.6, 0.2, 0))
        fade_in_whole(ctx, Group(rmf, dq, tq, ea, fc, grid), 1.0)
        ea.set_progress(2.0)
        ctx.play(grid.animate_fault("L2"), run_time=1.5)
        ctx.finish()
