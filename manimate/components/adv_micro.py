"""Microelectronics and computer architecture adapters (viz README section 12.1): thin wrappers over ``viz.microelectronics`` scenes.
"""
from __future__ import annotations

from ..ctx import Ctx
from ..registry import Param as P, component
from ..viz.microelectronics import (play_branch_prediction, play_cache_access, play_cpu_pipeline, play_memory_system,
                                    play_mosfet_channel, play_pn_junction, play_serial_bus)


def _in_range(lo: float, hi: float):
    return lambda v: None if lo <= v <= hi else f"must be between {lo:g} and {hi:g}"


def _bits(v: list) -> str | None:
    if not v:
        return "needs at least one outcome"
    return None if all(x in (0, 1) for x in v) else "outcomes must each be 0 (not taken) or 1 (taken)"


def _bytes_ok(limit: int):
    def check(v: list) -> str | None:
        if not v:
            return "needs at least one value"
        if len(v) > limit:
            return f"takes at most {limit} values"
        return None if all(0 <= x <= 255 and float(x).is_integer() for x in v) else "values must be whole numbers from 0 to 255"
    return check


def _pow2(lo: int, hi: int):
    return lambda v: None if lo <= v <= hi and (v & (v - 1)) == 0 else f"must be a power of two between {lo} and {hi}"


@component(
    "pn_junction",
    "PN junction: electrons and holes diffusing across the junction, a depletion region that widens in reverse bias and "
    "narrows in forward bias. Views: carriers, band diagram, I-V curve.",
    {"bias_v": P("number", "bias voltage in volts, -1 (reverse) to 0.65 (forward)", default=0.0, validator=_in_range(-1.0, 0.65)),
     "view": P("string", "which picture", default="carriers", choices=["carriers", "bands", "iv"])},
    {"bias_v": 0.0, "view": "carriers"},
    domain="microelectronics",
)
def pn_junction(ctx: Ctx, bias_v: float = 0.0, view: str = "carriers") -> None:
    play_pn_junction(ctx, view=view, bias_v=bias_v)


@component(
    "mosfet_channel",
    "MOSFET cross-section: the inversion channel forms as VGS passes the threshold and pinches off as VDS rises, with the "
    "drain current. Views: cross-section, or the I_D-V_DS characteristics.",
    {"kind": P("string", "transistor type (voltages are magnitudes for pmos)", default="nmos", choices=["nmos", "pmos"]),
     "vgs": P("number", "starting gate-source voltage in volts", default=0.0, validator=_in_range(0.0, 4.0)),
     "vds": P("number", "starting drain-source voltage in volts", default=0.0, validator=_in_range(0.0, 4.0)),
     "view": P("string", "which picture", default="cross_section", choices=["cross_section", "characteristics"])},
    {"kind": "nmos", "vgs": 0.0, "vds": 0.0, "view": "cross_section"},
    domain="microelectronics",
)
def mosfet_channel(ctx: Ctx, kind: str = "nmos", vgs: float = 0.0, vds: float = 0.0, view: str = "cross_section") -> None:
    play_mosfet_channel(ctx, kind=kind, vgs=vgs, vds=vds, view=view)


@component(
    "cpu_pipeline",
    "5-stage CPU pipeline (IF ID EX MEM WB) as a cycle-by-cycle grid: stall bubbles labelled with their cause, forwarding "
    "arrows, and branch flushes.",
    {"program": P("string", "instruction sequence", default="raw_hazard", choices=["raw_hazard", "load_use", "independent", "branch"]),
     "forwarding": P("boolean", "forward results between stages instead of stalling", default=True),
     "branch_policy": P("string", "how branches are handled", default="stall", choices=["stall", "predict_not_taken", "predict_taken"])},
    {"program": "raw_hazard", "forwarding": False, "branch_policy": "stall"},
    domain="microelectronics",
)
def cpu_pipeline(ctx: Ctx, program: str = "raw_hazard", forwarding: bool = True, branch_policy: str = "stall") -> None:
    play_cpu_pipeline(ctx, program=program, forwarding=forwarding, branch_policy=branch_policy)


@component(
    "branch_prediction",
    "Branch prediction: the predicted path versus the real one, the flush penalty on a miss, and the saturating-counter state machine.",
    {"predictor": P("string", "predictor type", default="2bit", choices=["static", "1bit", "2bit"]),
     "outcomes": P("number[]", "branch outcomes in order, 1 = taken, 0 = not taken (up to 8)", default=[1, 1, 0, 1, 1, 0], validator=_bits)},
    {"predictor": "2bit", "outcomes": [1, 1, 0, 1]},
    domain="microelectronics",
)
def branch_prediction(ctx: Ctx, predictor: str = "2bit", outcomes: list | None = None) -> None:
    play_branch_prediction(ctx, predictor=predictor, outcomes=[int(o) for o in (outcomes or [1, 1, 0, 1, 1, 0])])


@component(
    "cache_access",
    "Set-associative cache: each address is split into tag, index and offset, the set is searched, and the access hits or "
    "replaces a line (LRU or FIFO).",
    {"addresses": P("number[]", "byte addresses 0-255 to look up in order (up to 8)", default=[0, 4, 0, 32, 64, 0, 4],
                    validator=_bytes_ok(8)),
     "sets": P("integer", "number of sets (1, 2, 4 or 8)", default=4, validator=_pow2(1, 8)),
     "ways": P("integer", "ways per set, 1 to 4", default=2, validator=_in_range(1, 4)),
     "policy": P("string", "replacement policy", default="lru", choices=["lru", "fifo"])},
    {"addresses": [0, 4, 0, 32, 64], "sets": 4, "ways": 2, "policy": "lru"},
    domain="microelectronics",
)
def cache_access(ctx: Ctx, addresses: list | None = None, sets: int = 4, ways: int = 2, policy: str = "lru") -> None:
    play_cache_access(ctx, addresses=[int(a) for a in addresses] if addresses else None, sets=int(sets), ways=int(ways), policy=policy)


@component(
    "memory_system",
    "Memory system: the cache hierarchy with latencies and average access time, the address and data buses with RAM row/column "
    "selection, or DMA moving a block while the CPU stays free.",
    {"view": P("string", "which picture", default="hierarchy", choices=["hierarchy", "bus", "dma"]),
     "levels": P("integer", "cache levels in the hierarchy view, 1 to 3", default=3, validator=_in_range(1, 3)),
     "words": P("integer", "words moved in the dma view, 1 to 8", default=8, validator=_in_range(1, 8))},
    {"view": "hierarchy", "levels": 3, "words": 8},
    domain="microelectronics",
)
def memory_system(ctx: Ctx, view: str = "hierarchy", levels: int = 3, words: int = 8) -> None:
    play_memory_system(ctx, view=view, levels=int(levels), words=int(words))


@component(
    "serial_bus",
    "Serial bus waveforms with the sampling instants and frame fields: SPI (all four modes), I2C, UART, USB (NRZI and bit "
    "stuffing), CAN (with arbitration), and a differential pair that cancels noise.",
    {"variant": P("string", "protocol", default="spi", choices=["spi", "i2c", "uart", "usb", "can", "diff"]),
     "data": P("number[]", "bytes to send, 0-255 (at most 2)", default=[165], validator=_bytes_ok(2)),
     "mode": P("integer", "SPI mode 0-3 (CPOL, CPHA); for UART 0 = no parity, 1 = even, 2 = odd", default=0, validator=_in_range(0, 3))},
    {"variant": "spi", "data": [165], "mode": 0},
    domain="microelectronics",
)
def serial_bus(ctx: Ctx, variant: str = "spi", data: list | None = None, mode: int = 0) -> None:
    play_serial_bus(ctx, variant=variant, data=[int(b) for b in (data or [165])], mode=int(mode))
