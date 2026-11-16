"""Scenes for ``viz.power`` (README section 9): each ``play_*`` runs as a storyboard component or a standalone Scene.

Every scene follows the house timing rules: the finished static frame fades in whole, then the parts that carry the
meaning move, and the scene ends by filling whatever narration time is left (``ctx.finish``).
"""
from __future__ import annotations

import math

from manim import Group, ValueTracker, VGroup

from ...ctx import Ctx
from ..core.focus import fade_in_whole
from .converters import ConversionRatioCurve, ConverterTopology, HBridge, InductorRamp, PWMGenerator, ThreePhaseInverter
from .grid import PowerGrid, Substation
from .machines import (ClarkeParkTransform, DamperWindings, DQFrame, EqualAreaCriterion, FaultCurrentTransient, FOCBlockDiagram,
                       RotorAngleOscillation, TorqueDecomposition)

CONVERTERS = ("buck", "boost", "buck_boost")
CONVERTER_VIEWS = ("schematic", "waveforms", "both")
INVERTER_CIRCUITS = ("pwm", "h_bridge", "three_phase")
FOC_VIEWS = ("clarke_park", "dq_frame", "torque", "block_diagram")
STABILITY_VIEWS = ("swing", "equal_area", "fault_current")
GRID_LAYOUTS = ("radial", "ring", "single_bus", "double_bus", "ring_bus")


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _slots(ctx: Ctx, *weights: float, floor: float = 0.5):
    """Intuition: cutting the time that is left into shares, so the picture ends when the narration does."""
    left = ctx.remaining
    total = float(sum(weights)) or 1.0
    return [max(floor, left * w / total) for w in weights]


def play_dc_dc_converter(ctx: Ctx, topology: str = "buck", vin: float = 12.0, duty: float = 0.5, view: str = "both") -> None:
    """Intuition: a switch chops the supply and an inductor smooths it; the share of time the switch is on sets the output."""
    if topology not in CONVERTERS:
        raise ValueError(f"topology must be one of {list(CONVERTERS)}")
    if view not in CONVERTER_VIEWS:
        raise ValueError(f"view must be one of {list(CONVERTER_VIEWS)}")
    vin, d0 = _clamp(vin, 1, 1000), _clamp(duty, 0.1, 0.9)
    d = ValueTracker(d0)
    conv = ConverterTopology(topology, vin=vin, duty=d0).bind(d, "duty")
    ramp = InductorRamp(conv, width=3.4)
    curve = ConversionRatioCurve(topology, d, width=2.4, height=1.7)
    if view == "schematic":
        conv.scale(0.95).move_to((-2.6, 0.5, 0))
        curve.move_to((4.0, 0.5, 0))
        shown = Group(conv, curve)
    elif view == "waveforms":
        ramp.scale(1.1).move_to((-2.2, 0.5, 0))
        curve.move_to((4.0, 0.5, 0))
        shown = Group(ramp, curve)  # the schematic still drives the waveforms; it just stays off screen
    else:
        conv.scale(0.8).move_to((-3.6, 0.3, 0))
        ramp.scale(0.8).move_to((2.2, 0.2, 0))
        curve.scale(0.9).move_to((-3.7, -2.1, 0))
        shown = Group(conv, ramp, curve)
    fade_in_whole(ctx, shown, 1.0)
    a, b, c = _slots(ctx, 2, 2, 1)
    if view == "waveforms":
        a, b = a + b / 2, b / 2
    else:
        ctx.play(conv.animate_switching(3, run_time=a), run_time=a)
    ctx.play(d.animate.set_value(_clamp(d0 + 0.3, 0.1, 0.9) if d0 < 0.6 else _clamp(d0 - 0.3, 0.1, 0.9)), run_time=b if view != "waveforms" else a)
    ctx.play(d.animate.set_value(d0), run_time=c)
    ctx.finish()


def play_pwm_inverter(ctx: Ctx, circuit: str = "pwm", modulation: str = "bipolar", modulation_index: float = 0.8,
                      carrier_ratio: int = 15) -> None:
    """Intuition: a fast triangle wave races a slow sine; whenever the sine is higher the switch is on, so pulse widths trace the sine."""
    if circuit not in INVERTER_CIRCUITS:
        raise ValueError(f"circuit must be one of {list(INVERTER_CIRCUITS)}")
    ma = _clamp(modulation_index, 0.1, 1.0)
    ratio = int(_clamp(carrier_ratio, 9, 31))
    if circuit == "pwm":
        mob = PWMGenerator("sine", "triangle", ma, ratio)
        anim = lambda t: mob.animate_build(run_time=t)  # noqa: E731
    elif circuit == "h_bridge":
        if modulation not in ("bipolar", "unipolar"):
            raise ValueError("modulation must be 'bipolar' or 'unipolar'")
        mob = HBridge(modulation, ma, "RL", ratio)
        anim = lambda t: mob.animate_cycle(run_time=t)  # noqa: E731
    else:
        mob = ThreePhaseInverter(400.0, ma, ratio)
        anim = lambda t: mob.animate_build(run_time=t)  # noqa: E731
    ctx.fit(mob)
    fade_in_whole(ctx, mob, 1.0)
    (t,) = _slots(ctx, 1)
    ctx.play(anim(max(1.0, t * 0.92)), run_time=max(1.0, t * 0.92))
    ctx.finish()


def play_foc(ctx: Ctx, view: str = "clarke_park", frequency_hz: float = 50.0) -> None:
    """Intuition: ride on the rotor. Three wobbling currents become two flat ones, and flat is easy to control."""
    if view not in FOC_VIEWS:
        raise ValueError(f"view must be one of {list(FOC_VIEWS)}")
    f = _clamp(frequency_hz, 1, 400)
    if view == "clarke_park":
        mob = ClarkeParkTransform(freq=f)
        ctx.fit(mob)
        fade_in_whole(ctx, mob, 1.0)
        (t,) = _slots(ctx, 1)
        ctx.play(mob.animate_steps(run_time=max(1.5, t * 0.92)), run_time=max(1.5, t * 0.92))
    elif view == "dq_frame":
        theta = ValueTracker(0.0)
        frame = DQFrame(theta)
        ctx.fit(frame)
        fade_in_whole(ctx, frame, 1.0)
        (t,) = _slots(ctx, 1)
        ctx.play(theta.animate.set_value(2 * math.pi), run_time=max(1.5, t * 0.92), rate_func=lambda x: x)
    elif view == "torque":
        tq = TorqueDecomposition(0.0, 0.5, size=2.0)
        ctx.fit(tq)
        fade_in_whole(ctx, tq, 1.0)
        a, b, c = _slots(ctx, 1, 1, 1)
        ctx.play(tq.animate_to(0.0, 1.0, run_time=a), run_time=a)   # more q-axis current: more torque
        ctx.play(tq.animate_to(0.7, 1.0, run_time=b), run_time=b)   # d-axis current adds flux, no torque
        ctx.play(tq.animate_to(0.0, 0.5, run_time=c), run_time=c)
    else:
        mob = FOCBlockDiagram()
        ctx.fit(mob)
        fade_in_whole(ctx, mob, 1.0)
        (t,) = _slots(ctx, 1)
        ctx.play(mob.animate_signal(run_time=max(1.5, t * 0.92)), run_time=max(1.5, t * 0.92))
    ctx.finish()


def play_power_stability(ctx: Ctx, view: str = "swing", inertia_h: float = 5.0, mechanical_power: float = 0.8,
                         clearing_angle_deg: float = 52.0, dc_offset: bool = True) -> None:
    """Intuition: a heavy flywheel on a spring. A shock sets the rotor angle swinging, and the grid either pulls it back or lets it slip."""
    if view not in STABILITY_VIEWS:
        raise ValueError(f"view must be one of {list(STABILITY_VIEWS)}")
    pm = _clamp(mechanical_power, 0.1, 1.2)
    if view == "swing":
        mob = RotorAngleOscillation(H=_clamp(inertia_h, 1, 12), Pm=pm, Pmax=1.8, D=0.05, disturbance=0.5)
        run = lambda t: mob.animate_swing(run_time=t)  # noqa: E731
    elif view == "equal_area":
        mob = EqualAreaCriterion(Pm=pm, Pmax_pre=1.8, Pmax_fault=0.4, Pmax_post=1.4, delta_clear=math.radians(_clamp(clearing_angle_deg, 20, 80)))
        run = lambda t: mob.animate_areas(run_time=t)  # noqa: E731
    else:
        mob = FaultCurrentTransient(dc_offset=bool(dc_offset))
        run = lambda t: mob.animate_draw(run_time=t)  # noqa: E731
    ctx.fit(mob)
    fade_in_whole(ctx, mob, 1.0)
    (t,) = _slots(ctx, 1)
    ctx.play(run(max(1.5, t * 0.92)), run_time=max(1.5, t * 0.92))
    ctx.finish()


def play_power_grid(ctx: Ctx, layout: str = "radial", loads: int = 3, show_fault: bool = True) -> None:
    """Intuition: roads for electricity. Power flows from generators to loads, and a fault is a road closure the breakers isolate."""
    if layout not in GRID_LAYOUTS:
        raise ValueError(f"layout must be one of {list(GRID_LAYOUTS)}")
    loads = int(_clamp(loads, 1, 4))
    if layout in ("radial", "ring"):
        grid = PowerGrid.radial(loads) if layout == "radial" else PowerGrid.ring(max(3, loads))
        ctx.fit(grid)
        fade_in_whole(ctx, grid, 1.0)
        if show_fault:
            node = ("L2" if loads >= 2 else "L1") if layout == "radial" else "B2"
            a, b = _slots(ctx, 2, 1)
            ctx.play(grid.animate_fault(node, run_time=a), run_time=a)
            trip = [e for e in grid.edges if node in e and "G1" not in e]
            if trip:
                ctx.play(grid.animate_trip(trip, run_time=b), run_time=b)
    else:
        scheme = {"single_bus": "single", "double_bus": "double", "ring_bus": "ring"}[layout]
        sub = Substation(scheme, feeders=min(max(loads, 2), 4))
        ctx.fit(sub)
        fade_in_whole(ctx, sub, 1.0)
        path = {"single": ["bus_a", "dis_a1", "brk1", "fdr1"], "double": ["bus_b", "dis_b1", "brk1", "fdr1"],
                "ring": ["ring0", "brk0", "ring1", "brk1", "fdr1"]}[scheme]
        a, b = _slots(ctx, 1, 1)
        ctx.play(sub.animate_switching(path, run_time=a), run_time=a)
        if scheme != "ring":
            alt = ["bus_a", "dis_a0", "brk0", "fdr0"]
            ctx.play(sub.animate_switching(alt, run_time=b), run_time=b)
    ctx.finish()


def play_damper_windings(ctx: Ctx, poles: int = 4, slip: float = 0.05) -> None:
    """Intuition: copper bars in the rotor face. If the rotor lags the field, the bars cut flux and the induced current pulls it back."""
    poles = int(_clamp(poles, 2, 8))
    poles -= poles % 2
    s = _clamp(slip, 0.0, 0.5)
    mob = DamperWindings(poles, s)
    ctx.fit(mob)
    fade_in_whole(ctx, mob, 1.0)
    a, b = _slots(ctx, 1, 1)
    ctx.play(mob.animate_slip(min(0.5, s + 0.2), run_time=a), run_time=a)
    ctx.play(mob.animate_slip(0.0, run_time=b), run_time=b)
    ctx.finish()
