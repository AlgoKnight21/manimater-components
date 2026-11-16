"""Frames from viz.signals (control). Run one, from renderer/:  manim -ql -s examples/viz/signals_control.py LoopAndPID

Classes: LoopAndPID, StepAndTuning, FrequencyResponse, PlaneViews, BlockSpec, DiagramSpec.
"""
from manim import DOWN, VGroup

from manimate.ctx import Ctx
from manimate.viz.core import VizScene, fade_in_whole
from manimate.viz.signals import (BlockDiagram, BodePlot, FeedbackLoop, NyquistPlot, PIDController, PIDTuningDemo, PoleZeroPlot, RootLocus,
                                  StateSpaceDiagram, StepResponsePlot, TransferFunctionBlock)
from manimate.viz.signals import _math as sm


class LoopAndPID(VizScene):
    duration = 8.0

    def script(self, ctx: Ctx) -> None:
        loop = FeedbackLoop(["Controller", "Plant"], ["Sensor"]).scale(0.8).move_to((0, 2.0, 0))
        pid = PIDController(2.0, 1.0, 0.5).scale(0.62).move_to((0, -0.7, 0))
        fade_in_whole(ctx, VGroup(loop, pid), 1.0)
        ctx.play(loop.animate_signal(pulses=1, run_time=2.5), run_time=2.5)
        ctx.play(pid.animate_response(run_time=3.0), run_time=3.0)
        ctx.finish()


class StepAndTuning(VizScene):
    duration = 8.0

    def script(self, ctx: Ctx) -> None:
        demo = PIDTuningDemo("second_order", kp=1.0, ki=0.0, kd=0.0)
        ctx.fit(demo)
        fade_in_whole(ctx, demo, 1.0)
        ctx.play(demo.animate_gains(kp=4.0), run_time=2.0)  # P-only: faster, but a gap and some ringing
        ctx.play(demo.animate_gains(ki=2.0), run_time=2.0)  # I closes the gap
        ctx.play(demo.animate_gains(kd=1.5), run_time=2.0)  # D damps the overshoot
        ctx.finish()


class FrequencyResponse(VizScene):
    duration = 8.0

    def script(self, ctx: Ctx) -> None:
        bode = BodePlot([0, -1, -2], [], 10.0)  # a classic loop whose phase margin is negative at this gain
        ctx.fit(bode)
        fade_in_whole(ctx, bode, 1.0)
        ctx.play(bode.animate_sweep(run_time=3.0), run_time=3.0)
        ctx.play(bode.animate_margins(), run_time=1.2)
        ctx.finish()


class PlaneViews(VizScene):
    duration = 8.0

    def script(self, ctx: Ctx) -> None:
        nyq = NyquistPlot([0, -1, -2], [], 10.0, size=3.4)
        locus = RootLocus([1], [1, 3, 2, 0], 10.0, size=3.4)
        pz = PoleZeroPlot([-0.5 + 2j, -0.5 - 2j, -3], [-1], size=3.0)
        row = VGroup(nyq, locus, pz).arrange(buff=0.6)
        ctx.fit(row)
        fade_in_whole(ctx, row, 1.0)
        ctx.play(nyq.animate_trace(run_time=2.0), run_time=2.0)
        ctx.play(locus.animate_k(run_time=3.0), pz.animate_move_pole(0, 0.3 + 2j, run_time=3.0), run_time=3.0)
        ctx.finish()


class BlockSpec(VizScene):
    duration = 6.0

    def script(self, ctx: Ctx) -> None:
        tf = TransferFunctionBlock([1], [1, 0.6, 1], show_poles_zeros=True).scale(0.8)
        step = StepResponsePlot(sm.tf_from_coeffs([1], [1, 0.6, 1]), width=4.4, height=2.2)
        top = VGroup(tf, step).arrange(buff=0.6)
        ss = StateSpaceDiagram([[0, 1], [-2, -3]], [[0], [1]], [[1, 0]]).scale(0.7)
        group = VGroup(top, ss).arrange(DOWN, buff=0.5)
        ctx.fit(group)
        fade_in_whole(ctx, group, 1.0)
        ctx.play(step.animate_draw(run_time=2.0), run_time=2.0)
        ctx.play(step.animate_annotate(), run_time=1.2)
        ctx.finish()


class DiagramSpec(VizScene):
    duration = 4.0

    def script(self, ctx: Ctx) -> None:
        diagram = BlockDiagram({"r": {"label": "r", "kind": "io"}, "s": {"kind": "sum"}, "c": "Controller", "p": "Plant",
                                "y": {"label": "y", "kind": "io"}},
                               [("r", "s"), ("s", "c"), ("c", "p"), ("p", "y"), ("p", "s")])
        ctx.fit(diagram)
        fade_in_whole(ctx, diagram, 1.0)
        ctx.finish()
