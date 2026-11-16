"""Power electronics, machines and grid adapters (viz README sections 9 and 12.1): thin wrappers over ``viz.power`` scenes."""
from __future__ import annotations

from ..ctx import Ctx
from ..registry import Param as P, component
from ..viz.power import scenes as S

_D = "power-systems"


def _in_range(lo: float, hi: float, what: str = "value"):
    return lambda v: None if lo <= v <= hi else f"{what} must be between {lo:g} and {hi:g}"


@component(
    "dc_dc_converter",
    "DC-DC converter (buck, boost, buck-boost): the switching schematic, the inductor current ramping up and down, and the "
    "output-to-input ratio curve as the duty cycle changes.",
    {"topology": P("string", "converter type", default="buck", choices=list(S.CONVERTERS)),
     "vin": P("number", "input voltage in volts, 1 to 1000", default=12, validator=_in_range(1, 1000, "vin")),
     "duty": P("number", "fraction of each period the switch is on, 0.1 to 0.9", default=0.5, validator=_in_range(0.1, 0.9, "duty")),
     "view": P("string", "which pictures", default="both", choices=list(S.CONVERTER_VIEWS))},
    {"topology": "buck", "vin": 12, "duty": 0.4, "view": "both"},
    domain=_D,
)
def dc_dc_converter(ctx: Ctx, topology: str = "buck", vin: float = 12, duty: float = 0.5, view: str = "both") -> None:
    S.play_dc_dc_converter(ctx, topology, vin, duty, view)


@component(
    "pwm_inverter",
    "Pulse-width modulation: a triangle carrier racing a sine reference to make switching pulses, driving a single-phase "
    "H-bridge (bipolar or unipolar) or a three-phase inverter.",
    {"circuit": P("string", "what to show", default="pwm", choices=list(S.INVERTER_CIRCUITS)),
     "modulation": P("string", "H-bridge switching style (only for h_bridge)", default="bipolar", choices=["bipolar", "unipolar"]),
     "modulation_index": P("number", "reference amplitude over carrier amplitude, 0.1 to 1", default=0.8, validator=_in_range(0.1, 1, "modulation_index")),
     "carrier_ratio": P("integer", "carrier frequency over reference frequency, 9 to 31", default=15, validator=_in_range(9, 31, "carrier_ratio"))},
    {"circuit": "three_phase", "modulation": "bipolar", "modulation_index": 0.9, "carrier_ratio": 21},
    domain=_D,
)
def pwm_inverter(ctx: Ctx, circuit: str = "pwm", modulation: str = "bipolar", modulation_index: float = 0.8, carrier_ratio: int = 15) -> None:
    S.play_pwm_inverter(ctx, circuit, modulation, modulation_index, int(carrier_ratio))


@component(
    "foc_transform",
    "Field-oriented (vector) motor control: Clarke and Park transforms turning three sine currents into flat d and q "
    "values, the rotating d-q frame, torque from the q-axis current, or the full control block diagram.",
    {"view": P("string", "which picture", default="clarke_park", choices=list(S.FOC_VIEWS)),
     "frequency_hz": P("number", "electrical frequency in Hz, 1 to 400", default=50, validator=_in_range(1, 400, "frequency_hz"))},
    {"view": "clarke_park", "frequency_hz": 50},
    domain=_D,
)
def foc_transform(ctx: Ctx, view: str = "clarke_park", frequency_hz: float = 50) -> None:
    S.play_foc(ctx, view, frequency_hz)


@component(
    "power_stability",
    "Power-system stability: the rotor angle swinging after a disturbance, the equal-area criterion for a fault cleared at "
    "a chosen angle, or a generator's short-circuit current decaying from subtransient to steady state.",
    {"view": P("string", "which picture", default="swing", choices=list(S.STABILITY_VIEWS)),
     "inertia_h": P("number", "inertia constant H in seconds, 1 to 12 (swing view)", default=5, validator=_in_range(1, 12, "inertia_h")),
     "mechanical_power": P("number", "mechanical power in per unit, 0.1 to 1.2", default=0.8, validator=_in_range(0.1, 1.2, "mechanical_power")),
     "clearing_angle_deg": P("number", "angle at which the fault is cleared, 20 to 80 degrees (equal_area view)", default=52, validator=_in_range(20, 80, "clearing_angle_deg")),
     "dc_offset": P("boolean", "include the decaying DC offset (fault_current view)", default=True)},
    {"view": "equal_area", "inertia_h": 5, "mechanical_power": 0.8, "clearing_angle_deg": 52, "dc_offset": True},
    domain=_D,
)
def power_stability(ctx: Ctx, view: str = "swing", inertia_h: float = 5, mechanical_power: float = 0.8,
                    clearing_angle_deg: float = 52, dc_offset: bool = True) -> None:
    S.play_power_stability(ctx, view, inertia_h, mechanical_power, clearing_angle_deg, dc_offset)


@component(
    "power_grid",
    "An electrical grid: generator, buses, lines and loads with power-flow arrows and a fault that breakers isolate, or "
    "substation bus layouts (single bus, double bus, breaker ring) with the route the current takes.",
    {"layout": P("string", "grid or substation layout", default="radial", choices=list(S.GRID_LAYOUTS)),
     "loads": P("integer", "loads or feeders, 1 to 4", default=3, validator=_in_range(1, 4, "loads")),
     "show_fault": P("boolean", "show a fault and the breakers clearing it", default=True)},
    {"layout": "radial", "loads": 3, "show_fault": True},
    domain=_D,
)
def power_grid(ctx: Ctx, layout: str = "radial", loads: int = 3, show_fault: bool = True) -> None:
    S.play_power_grid(ctx, layout, int(loads), show_fault)


@component(
    "damper_windings",
    "Damper (amortisseur) windings in a synchronous machine's rotor: how slip makes bars cut flux and the induced current pulls the rotor back in step.",
    {"poles": P("integer", "number of poles, an even number from 2 to 8", default=4, validator=_in_range(2, 8, "poles")),
     "slip": P("number", "starting slip, 0 to 0.5", default=0.05, validator=_in_range(0, 0.5, "slip"))},
    {"poles": 4, "slip": 0.05},
    domain=_D,
)
def damper_windings(ctx: Ctx, poles: int = 4, slip: float = 0.05) -> None:
    S.play_damper_windings(ctx, int(poles), slip)
