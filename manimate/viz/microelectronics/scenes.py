"""Scenes for ``viz.microelectronics`` (README section 7): each ``play_*`` runs as a storyboard component or a standalone Scene.

Every scene follows the house timing rules: the static frame fades in whole, then the parts that carry the meaning move,
and the scene ends by filling whatever narration time is left (``ctx.finish``).
"""
from __future__ import annotations

from manim import FadeIn, FadeOut, ValueTracker, VGroup

from ...ctx import Ctx
from ..core.focus import FocusRing, fade_in_whole
from ..core.live import LiveValue, Slider
from ..core.palette import PALETTE
from .buses import (CANArbitration, CANBus, DifferentialPair, I2CBus, SPIBus, UARTFrame, USBLine)
from ._math import can_frame
from .cpu import BranchPredictionPath, PipelineDiagram, PredictorFSM
from .memory import AddressBus, CacheArray, CacheHierarchy, DataBus, DMAController, RAMArray
from .semiconductor import (DiodeIVCurve, DrainCurrentVectors, MOSFETCharacteristics, MOSFETCrossSection, PNBandDiagram,
                            PNJunction)

PN_VIEWS = ("carriers", "bands", "iv")
MOS_VIEWS = ("cross_section", "characteristics")


def _share(ctx: Ctx):
    """Hand out slices of the time that is left when the opening fade ends."""
    left = ctx.remaining

    def take(fraction: float, floor: float = 0.6) -> float:
        return max(floor, fraction * left)

    return take


def play_pn_junction(ctx: Ctx, view: str = "carriers", bias_v: float = 0.0) -> None:
    """Intuition: two crowds mixing until a charged moat stops them; bias widens or narrows the moat."""
    if view not in PN_VIEWS:
        raise ValueError(f"view must be one of {list(PN_VIEWS)}")
    bias_v = max(-1.0, min(0.65, float(bias_v)))
    bias = ValueTracker(bias_v)
    junction = PNJunction(bias=bias_v).bind(bias)
    slider = Slider(bias, (-1.0, 0.8), "bias", unit="V")
    if view == "carriers":
        junction.scale(0.78).move_to((0, 1.0, 0))
        slider.move_to((0, -2.0, 0))
        frame = VGroup(junction, slider)
    elif view == "bands":
        bands = PNBandDiagram(junction)
        junction.scale(0.6).move_to((0, 2.0, 0))
        bands.scale(0.6).move_to((0, -0.1, 0))
        slider.move_to((0, -2.1, 0))
        frame = VGroup(junction, bands, slider)
    else:
        iv = DiodeIVCurve(v_range=(-1.0, 0.8)).bind(bias)
        junction.scale(0.6).move_to((-3.1, 1.0, 0))
        iv.scale(0.85).move_to((3.2, 0.9, 0))
        slider.move_to((0, -1.9, 0))
        frame = VGroup(junction, iv, slider)
    fade_in_whole(ctx, frame, 1.3)
    take = _share(ctx)
    if view == "carriers":
        ctx.play(junction.animate_diffusion(run_time=take(0.1)), run_time=take(0.1))
        ctx.play(junction.animate_recombination(run_time=take(0.12)), run_time=take(0.12))
        ctx.play(junction.animate_recombination(run_time=take(0.09)), run_time=take(0.09))
    ctx.play(bias.animate.set_value(0.65), run_time=take(0.24))  # forward: the moat narrows
    ctx.play(bias.animate.set_value(-0.9), run_time=take(0.28))  # reverse: the moat widens
    ctx.play(bias.animate.set_value(bias_v), run_time=take(0.1))
    ctx.finish()


def play_mosfet_channel(ctx: Ctx, kind: str = "nmos", vgs: float = 0.0, vds: float = 0.0, view: str = "cross_section",
                        vth: float = 1.0, k: float = 0.5) -> None:
    """Intuition: a plate that pulls carriers into a bridge under the oxide; the drain voltage tilts it until it pinches off."""
    if view not in MOS_VIEWS:
        raise ValueError(f"view must be one of {list(MOS_VIEWS)}")
    device = MOSFETCrossSection(kind, max(0.0, vgs), max(0.0, vds), vth, k=k)
    vgs_s = Slider(device.vgs, (0.0, 4.0), "VGS", unit="V")
    vds_s = Slider(device.vds, (0.0, 4.0), "VDS", unit="V")
    current = LiveValue("ID", device.current, decimals=2)
    vectors = DrainCurrentVectors(device)
    if view == "cross_section":
        device.scale(0.82).move_to((-1.6, 0.8, 0))
        vgs_s.move_to((4.0, 1.9, 0))
        vds_s.move_to((4.0, 0.5, 0))
        current.move_to((4.0, -0.8, 0))
        frame = VGroup(device, vgs_s, vds_s, current, vectors)
    else:
        device.scale(0.58).move_to((-3.5, 1.0, 0))
        charts = MOSFETCharacteristics(vth, k).link(device).scale(0.8).move_to((3.0, 0.9, 0))
        vgs_s.move_to((-3.6, -1.0, 0))
        vds_s.move_to((-3.6, -1.85, 0))
        frame = VGroup(device, charts, vgs_s, vds_s, vectors)
    fade_in_whole(ctx, frame, 1.3)
    take = _share(ctx)
    vgs_hi = max(float(vgs), vth + 1.8)
    ctx.play(device.vgs.animate.set_value(vgs_hi), run_time=take(0.34))  # the bridge forms as VGS crosses Vth
    ctx.play(device.vds.animate.set_value(min(4.0, vgs_hi - vth + 1.5)), run_time=take(0.38))  # and pinches off
    ctx.finish()


# --------------------------------------------------------------------------------------------------------------------
# CPU pipeline and memory scenes (README section 7.3)
# --------------------------------------------------------------------------------------------------------------------
def play_cpu_pipeline(ctx: Ctx, program="raw_hazard", forwarding: bool = True, branch_policy: str = "stall") -> None:
    """Intuition: a laundromat line. One instruction enters per cycle; a hazard makes the one behind it wait, and forwarding removes the wait."""
    d = PipelineDiagram(program, forwarding=forwarding, branch_policy=branch_policy)
    ctx.fit(d, max_w=11.8, max_h=4.4)
    fade_in_whole(ctx, d, 1.3)
    n = d.schedule.total_cycles
    per = max(0.25, 0.5 * ctx.remaining / n)
    for c in range(1, n + 1):
        ctx.play(d.animate_cycle(c, run_time=per), run_time=per)
    take = _share(ctx)
    first = next((i for i in range(len(d.instrs)) if d.stall_cells(i)), None)
    if first is not None:
        ctx.play(d.animate_stall(first, run_time=take(0.3)), run_time=take(0.3))
    for a, b, *_ in d.forward_edges()[:2]:
        ctx.play(d.animate_forward(a, b, run_time=take(0.2)), run_time=take(0.2))
    ctx.finish()


def play_branch_prediction(ctx: Ctx, predictor: str = "2bit", outcomes=(1, 1, 0, 1), program="branch") -> None:
    """Intuition: the CPU guesses a fork in the road; a wrong guess throws away the steps it already took."""
    outcomes = tuple(outcomes)
    path = BranchPredictionPath(program, predictor, outcomes).scale(0.78).move_to((0, 1.35, 0))
    fsm = None
    if predictor != "static":
        bits = 2 if predictor == "2bit" else 1
        fsm = PredictorFSM(bits, state=1 if bits == 2 else 0).scale(0.72).move_to((0, -1.35, 0))
    frame = VGroup(path, fsm) if fsm is not None else path
    fade_in_whole(ctx, frame, 1.3)
    per = max(0.9, 0.85 * ctx.remaining / len(outcomes))
    for i, o in enumerate(outcomes):
        anims = [path.animate_outcome(i, run_time=per)]
        if fsm is not None:
            anims.append(fsm.animate_run([o], run_time=per))
        ctx.play(*anims, run_time=per)
    ctx.finish()


def play_cache_access(ctx: Ctx, addresses=None, sets: int = 4, ways: int = 2, policy: str = "lru", block: int = 4) -> None:
    """Intuition: small shelves of recently used books; the index picks the shelf, the tag says which book, hits are instant."""
    addrs = [int(a) % 256 for a in (addresses or (0x00, 0x04, 0x00, 0x20, 0x40, 0x00, 0x04))][:8]
    cache = CacheArray(sets, ways, block, policy)
    ctx.fit(cache, max_w=10.5, max_h=4.6)
    fade_in_whole(ctx, cache, 1.3)
    per = max(1.0, 0.85 * ctx.remaining / len(addrs))
    for a in addrs:
        ctx.play(cache.animate_access(a, run_time=per), run_time=per)
    ctx.finish()


def play_memory_system(ctx: Ctx, view: str = "hierarchy", levels: int = 3, words: int = 8, cpu_busy: float = 0.5,
                       mode: str = "steal") -> None:
    """Intuition: pantries of growing size and delay (hierarchy), wires that name and carry a word (bus), a courier that moves blocks (dma)."""
    if view not in ("hierarchy", "bus", "dma"):
        raise ValueError("view must be 'hierarchy', 'bus' or 'dma'")
    if view == "hierarchy":
        levels = max(1, min(3, int(levels)))
        names = ("L1", "L2", "L3")[:levels]
        lat = (1, 10, 40)[:levels] + (200,)
        h = CacheHierarchy(names, True, lat)
        ctx.fit(h, max_w=9.0, max_h=4.8)
        fade_in_whole(ctx, h, 1.3)
        patterns = [tuple(i == j for i in range(levels)) for j in range(levels)] + [tuple(False for _ in range(levels))]
        per = max(1.2, 0.85 * ctx.remaining / len(patterns))
        for p in patterns:
            ctx.play(h.animate_lookup(hits=p, run_time=per), run_time=per)
    elif view == "bus":
        addr, data, ram = AddressBus(8), DataBus(8), RAMArray(4, 4)
        addr.scale(0.8).move_to((0, 2.75, 0))
        ram.move_to((0, 0.55, 0))
        data.scale(0.8).move_to((0, -1.65, 0))
        fade_in_whole(ctx, VGroup(addr, ram, data), 1.3)
        take = _share(ctx)
        ctx.play(addr.animate_transfer(0x2A, run_time=take(0.25)), run_time=take(0.25))
        ctx.play(ram.animate_select(2, 2, run_time=take(0.3)), run_time=take(0.3))  # 0x2A = row 2, column 2
        ctx.play(data.animate_transfer(0xC3, run_time=take(0.25)), run_time=take(0.25))
    else:
        dma = DMAController(max(1, min(8, int(words))), cpu_busy, mode)
        ctx.fit(dma, max_w=11.8, max_h=5.0)
        fade_in_whole(ctx, dma, 1.3)
        t = max(1.5, 0.85 * ctx.remaining)
        ctx.play(dma.animate_transfer(run_time=t), run_time=t)
    ctx.finish()


# --------------------------------------------------------------------------------------------------------------------
# Bus protocols (README section 7.4)
# --------------------------------------------------------------------------------------------------------------------
SERIAL_VARIANTS = ("spi", "i2c", "uart", "usb", "can", "diff")


def _bytes(data, limit: int) -> list[int]:
    out = [int(b) & 0xFF for b in (data or [0xA5])][:limit]
    return out or [0xA5]


def play_serial_bus(ctx: Ctx, variant: str = "spi", data=None, mode: int = 0) -> None:
    """Intuition: a few wires, a shared beat and an agreement about when to look; each protocol is a different agreement."""
    if variant not in SERIAL_VARIANTS:
        raise ValueError(f"variant must be one of {list(SERIAL_VARIANTS)}")
    mode = int(mode)
    if variant == "spi":
        d = _bytes(data, 2)
        pic = SPIBus(d, [b ^ 0xFF for b in d], (mode >> 1) & 1, mode & 1)
    elif variant == "i2c":
        pic = I2CBus(0x48, _bytes(data, 2), "w")
    elif variant == "uart":
        pic = UARTFrame(_bytes(data, 1)[0], ("none", "even", "odd")[mode % 3], 1)
    elif variant == "usb":
        bits = [(b >> (7 - i)) & 1 for b in _bytes(data, 2) for i in range(8)]
        pic = USBLine(bits)
    elif variant == "can":
        pic = CANBus(can_frame(0x123, _bytes(data, 2)))
    else:
        pic = DifferentialPair([(b >> (7 - i)) & 1 for b in _bytes(data, 1) for i in range(8)])
    ctx.fit(pic, max_w=11.8, max_h=4.9)
    fade_in_whole(ctx, pic, 1.3)
    take = _share(ctx)
    if variant in ("spi", "i2c", "uart"):
        t = take(0.8, 1.5)
        ctx.play(pic.animate_sampling(run_time=t), run_time=t)
    elif variant == "usb":
        if pic.data.stuffed:
            ctx.play(pic.animate_stuffing(run_time=take(0.5)), run_time=take(0.5))
    elif variant == "can":
        t = take(0.4, 1.5)
        ctx.play(pic.animate_scan(run_time=t), run_time=t)
        arb = CANArbitration([0x120, 0x100, 0x180])
        ctx.play(FadeOut(pic), run_time=0.5)
        ctx.fit(arb, max_w=11.8, max_h=4.9)
        ctx.play(FadeIn(arb), run_time=0.8)
        t = max(1.5, 0.8 * ctx.remaining)
        ctx.play(arb.animate_arbitration(run_time=t), run_time=t)
    else:
        t = take(0.8, 1.5)
        ctx.play(pic.animate_noise(run_time=t), run_time=t)
    ctx.finish()
