"""Pure maths for ``viz.microelectronics`` (README section 7.1). No Manim imports: everything here is plain Python and NumPy,
so the numbers can be tested exactly and read by the pictures through their ``.data`` properties.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..core.numeric import rng

VT_300K = 0.02585  # thermal voltage kT/q at 300 K, in volts
DEFAULT_VBI = 0.7  # a silicon-like built-in potential, in volts


# --------------------------------------------------------------------------------------------------------------------
# Semiconductor physics
# --------------------------------------------------------------------------------------------------------------------
def depletion_width(v_bias: float, na: float = 1.0, nd: float = 1.0, normalise: bool = True,
                    vbi: float = DEFAULT_VBI) -> tuple[float, float]:
    """Abrupt-junction depletion width and built-in potential.

    ``v_bias`` is positive for forward bias. The width scales as ``sqrt(Vbi - V)``, so it grows monotonically as the
    bias goes from forward to reverse. Forward bias is clamped just below ``Vbi`` (the model stops being valid there).
    With ``normalise`` the width is divided by its zero-bias value for the same doping, so the result is 1.0 at V = 0.
    """
    if na <= 0 or nd <= 0:
        raise ValueError("doping levels na and nd must be positive")
    if vbi <= 0:
        raise ValueError("vbi must be positive")
    drop = max(vbi - float(v_bias), 0.05 * vbi)
    scale = math.sqrt(1.0 / na + 1.0 / nd)
    w = math.sqrt(drop) * scale
    if normalise:
        w /= math.sqrt(vbi) * scale
    return w, vbi


def depletion_split(width: float, na: float = 1.0, nd: float = 1.0) -> tuple[float, float]:
    """Split a total depletion width into ``(x_p, x_n)``: charge balance ``na * x_p = nd * x_n`` puts more of it in the lighter side."""
    if na <= 0 or nd <= 0:
        raise ValueError("doping levels na and nd must be positive")
    return width * nd / (na + nd), width * na / (na + nd)


def diode_current(v: float, i_s: float = 1e-12, n: float = 1.0, vt: float = VT_300K) -> float:
    """Shockley diode equation ``I = Is (exp(V / (n Vt)) - 1)``; the exponent is clamped so the maths never overflows."""
    if n <= 0 or vt <= 0:
        raise ValueError("n and vt must be positive")
    return i_s * (math.exp(min(float(v) / (n * vt), 80.0)) - 1.0)


def mos_region(vgs: float, vds: float, vth: float) -> str:
    """Operating region of an n-channel MOSFET: ``"cutoff"``, ``"triode"`` or ``"saturation"`` (boundary at ``vds = vgs - vth``)."""
    vov = vgs - vth
    if vov <= 0:
        return "cutoff"
    return "triode" if vds < vov else "saturation"


def drain_current(vgs: float, vds: float, vth: float, k: float = 0.5) -> float:
    """Square-law drain current. ``k`` lumps mobility, oxide capacitance and W/L together."""
    vov = vgs - vth
    region = mos_region(vgs, vds, vth)
    if region == "cutoff":
        return 0.0
    if region == "triode":
        return k * (vov * vds - 0.5 * vds * vds)
    return 0.5 * k * vov * vov


def pinch_point(vgs: float, vds: float, vth: float) -> float | None:
    """Fraction of the channel (0 = source, 1 = drain) where the inversion layer pinches off; ``None`` when there is no channel.

    The channel potential rises as ``vds * x``; the layer vanishes where it reaches the overdrive ``vgs - vth``.
    """
    vov = vgs - vth
    if vov <= 0:
        return None
    if vds <= 0:
        return 1.0
    return min(1.0, vov / vds)


def channel_profile(vgs: float, vds: float, vth: float, n: int = 21, vov_ref: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Inversion-layer depth along the channel, ``(xs, depth)``. ``depth`` is normalised by ``vov_ref`` and clamped to [0, 1].

    Depth is proportional to ``vov - vds * x``: uniform at ``vds = 0``, tapering toward the drain, and zero past the pinch-off point.
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    xs = np.linspace(0.0, 1.0, n)
    vov = vgs - vth
    if vov <= 0:
        return xs, np.zeros(n)
    depth = np.clip((vov - max(vds, 0.0) * xs) / vov_ref, 0.0, 1.0)
    return xs, depth


# --------------------------------------------------------------------------------------------------------------------
# A tiny instruction set and the 5-stage pipeline
# --------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Instr:
    text: str
    op: str  # "alu", "load", "store" or "branch"
    dest: str | None
    srcs: tuple[str, ...]
    taken: bool = False


PROGRAMS: dict[str, list[str]] = {
    "raw_hazard": ["add r1, r2, r3", "sub r4, r1, r5", "and r6, r1, r7", "or r8, r1, r9"],
    "load_use": ["lw r1, 0(r2)", "add r3, r1, r4", "sub r5, r3, r6", "sw r5, 4(r2)"],
    "independent": ["add r1, r2, r3", "sub r4, r5, r6", "and r7, r8, r9", "or r10, r11, r12"],
    "branch": ["add r1, r2, r3", "beq r1, r4, L1 (taken)", "lw r5, 0(r6)", "add r7, r5, r8"],
}
_REG = re.compile(r"\br\d+\b")


def parse_instr(line: str) -> Instr:
    """Parse ``"add r1, r2, r3"``, ``"lw r1, 0(r2)"``, ``"sw r1, 4(r2)"`` or ``"beq r1, r2, L (taken)"``."""
    taken = "(taken)" in line
    text = line.replace("(taken)", "").strip()
    if not text:
        raise ValueError("empty instruction")
    mnemonic = text.split()[0].lower()
    regs = _REG.findall(text)
    if mnemonic in ("lw", "ld", "lb"):
        if len(regs) < 2:
            raise ValueError(f"load needs a destination and a base register: {line!r}")
        return Instr(text, "load", regs[0], tuple(regs[1:]))
    if mnemonic in ("sw", "sd", "sb"):
        return Instr(text, "store", None, tuple(regs))
    if mnemonic in ("beq", "bne", "blt", "bge"):
        return Instr(text, "branch", None, tuple(regs), taken)
    if not regs:
        raise ValueError(f"no registers found in instruction {line!r}")
    return Instr(text, "alu", regs[0], tuple(regs[1:]))


def resolve_program(program: str | Sequence[str]) -> list[Instr]:
    """A preset name from ``PROGRAMS`` or a list of instruction strings (at most 8) becomes a list of ``Instr``."""
    lines = PROGRAMS.get(program) if isinstance(program, str) else list(program)
    if lines is None:
        raise ValueError(f"unknown program {program!r}; choose one of {sorted(PROGRAMS)} or pass a list of instructions")
    if not 1 <= len(lines) <= 8:
        raise ValueError("a program needs between 1 and 8 instructions")
    return [parse_instr(s) for s in lines]


@dataclass
class Cell:
    cycle: int
    label: str  # "IF", "ID", ..., "stall", "X"
    kind: str  # "stage", "stall", "hold" or "flush"
    cause: str | None = None  # "RAW", "load-use" or "control" for stall cells


@dataclass
class Row:
    text: str
    squashed: bool
    cells: list[Cell]


@dataclass
class PipelineSchedule:
    stages: tuple[str, ...]
    rows: list[Row]
    total_cycles: int
    stall_cycles: int
    flush_cycles: int
    forwards: list[tuple[int, int, int, int]] = field(default_factory=list)  # (src_row, dst_row, src_cycle, dst_cycle)


BRANCH_POLICIES = ("stall", "predict_not_taken", "predict_taken")


def pipeline_schedule(instrs: Sequence[Instr], forwarding: bool = True, branch_policy: str = "stall",
                      stages: tuple[str, ...] = ("IF", "ID", "EX", "MEM", "WB")) -> PipelineSchedule:
    """Cycle-by-cycle occupancy of a classic in-order 5-stage pipeline.

    Rules: the register file is written in the first half of WB and read in the second half of ID, so a consumer's ID may
    share a cycle with the producer's WB. With ``forwarding`` an ALU result is usable by the next EX, a load result only
    after MEM (so a load-use pair costs one bubble). Branches resolve at the end of EX (two wasted fetches) except that
    ``predict_taken`` knows the target after ID (one wasted fetch when right).
    """
    if tuple(stages) != ("IF", "ID", "EX", "MEM", "WB"):
        raise ValueError("only the classic five stages IF, ID, EX, MEM, WB are supported")
    if branch_policy not in BRANCH_POLICIES:
        raise ValueError(f"branch_policy must be one of {list(BRANCH_POLICIES)}")
    rows: list[Row] = []
    writers: dict[str, tuple[int, Instr, int, int, int]] = {}  # reg -> (row index, instr, EX cycle, MEM cycle, WB cycle)
    forwards: list[tuple[int, int, int, int]] = []
    next_fetch, prev_id = 1, 0
    stall_total = flush_total = 0
    for ins in instrs:
        f = next_fetch
        d_min = max(f + 1, prev_id + 1)
        d, cause = d_min, None
        for pos, reg in enumerate(ins.srcs):
            w = writers.get(reg)
            if w is None:
                continue
            _, prod, p_ex, p_mem, p_wb = w
            store_data = ins.op == "store" and pos == 0  # a store's data is only needed in MEM
            offset = 2 if store_data else 1
            if forwarding:
                avail = p_mem if prod.op == "load" else p_ex
                need_id = avail + 1 - offset
            else:
                need_id = p_wb
            if need_id > d:
                d, cause = need_id, ("load-use" if (forwarding and prod.op == "load") else "RAW")
        ex, mem, wb = d + 1, d + 2, d + 3
        cells = [Cell(f, "IF", "stage")]
        cells += [Cell(c, "IF", "hold") for c in range(f + 1, d_min)]
        cells += [Cell(c, "ID", "stall", cause) for c in range(d_min, d)]
        cells += [Cell(d, "ID", "stage"), Cell(ex, "EX", "stage"), Cell(mem, "MEM", "stage"), Cell(wb, "WB", "stage")]
        idx = len(rows)
        rows.append(Row(ins.text, False, cells))
        stall_total += d - d_min
        for pos, reg in enumerate(ins.srcs):  # record forwarding edges that were actually needed
            w = writers.get(reg)
            if forwarding and w is not None and d < w[4]:
                prod = w[1]
                src_c = w[3] if prod.op == "load" else w[2]
                store_data = ins.op == "store" and pos == 0
                forwards.append((w[0], idx, src_c, mem if store_data else ex))
        if ins.dest:
            writers[ins.dest] = (idx, ins, ex, mem, wb)
        prev_id = d
        next_fetch = d_min  # the next instruction enters IF as this one leaves it
        if ins.op == "branch":
            if branch_policy == "stall":
                next_fetch = ex + 1
                stall_total += ex - f
            else:
                mispredict = ins.taken if branch_policy == "predict_not_taken" else not ins.taken
                resolve = ex if mispredict else (d if branch_policy == "predict_taken" else None)
                if resolve is not None:  # wrong-path fetches are squashed when the branch resolves
                    for j in range(1, min(2, resolve - f) + 1):
                        c0 = f + j
                        sq = [Cell(c0, "IF", "flush")] + [Cell(c, "ID", "flush") for c in range(c0 + 1, resolve + 1)]
                        sq.append(Cell(resolve + 1, "X", "flush"))
                        rows.append(Row("(wrong path)", True, sq))
                    next_fetch = resolve + 1
                    flush_total += resolve - f
    if branch_policy == "stall":  # relabel: the instruction after a branch waits in IF for the branch to resolve
        _label_control_waits(rows, instrs)
    total = max(c.cycle for r in rows for c in r.cells)
    return PipelineSchedule(tuple(stages), rows, total, stall_total, flush_total, forwards)


def _label_control_waits(rows: list[Row], instrs: Sequence[Instr]) -> None:
    """With the ``stall`` policy the instruction fetched after a branch shows its wait as ``control`` stall cells in IF."""
    for i, ins in enumerate(instrs[:-1]):
        if ins.op != "branch":
            continue
        branch_if = rows[i].cells[0].cycle
        nxt = rows[i + 1]
        first = nxt.cells[0]
        if first.cycle - branch_if > 1:  # gap between the two fetches: fill it with stall cells
            gap = [Cell(c, "IF", "stall", "control") for c in range(branch_if + 1, first.cycle)]
            nxt.cells[0:0] = gap


# --------------------------------------------------------------------------------------------------------------------
# Branch prediction
# --------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PredictorData:
    kind: str
    states: tuple[int, ...]  # counter state *before* each outcome
    predictions: tuple[int, ...]
    hits: tuple[bool, ...]
    final_state: int
    mispredicts: int


def predictor_run(history: Sequence[int], kind: str = "2bit", initial: int | None = None) -> PredictorData:
    """Run ``"static"`` (always not taken), ``"1bit"`` or ``"2bit"`` (saturating counter) predictors over taken/not-taken outcomes."""
    if kind not in ("static", "1bit", "2bit"):
        raise ValueError("kind must be 'static', '1bit' or '2bit'")
    top = {"static": 0, "1bit": 1, "2bit": 3}[kind]
    state = (1 if kind == "2bit" else 0) if initial is None else int(initial)
    if not 0 <= state <= top:
        raise ValueError(f"initial state must be between 0 and {top}")
    states, preds, hits = [], [], []
    for o in history:
        if o not in (0, 1):
            raise ValueError("outcomes must be 0 (not taken) or 1 (taken)")
        states.append(state)
        pred = 0 if kind == "static" else (1 if state >= (top + 1) / 2 else 0)
        preds.append(pred)
        hits.append(pred == o)
        if kind != "static":
            state = min(top, state + 1) if o else max(0, state - 1)
    return PredictorData(kind, tuple(states), tuple(preds), tuple(hits), state, sum(1 for h in hits if not h))


def two_bit_predictor(history: Sequence[int], initial: int = 1) -> PredictorData:
    """State sequence of a 2-bit saturating counter (0, 1 predict not taken; 2, 3 predict taken)."""
    return predictor_run(history, "2bit", initial)


# --------------------------------------------------------------------------------------------------------------------
# Caches and memory
# --------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Access:
    address: int
    tag: int
    index: int
    offset: int
    hit: bool
    way: int
    evicted_tag: int | None
    order: tuple[int, ...]  # ways of this set, victim-first (LRU or oldest) to safest
    state: tuple[tuple[int | None, ...], ...]  # tag per way per set, after the access (None = invalid)


def _is_pow2(x: int) -> bool:
    return x >= 1 and (x & (x - 1)) == 0


class CacheModel:
    """A set-associative cache you can step one access at a time. Addresses are byte addresses; ``block`` is in bytes."""

    def __init__(self, sets: int = 4, ways: int = 2, block: int = 4, policy: str = "lru"):
        if not (_is_pow2(sets) and _is_pow2(block)):
            raise ValueError("sets and block must be powers of two")
        if ways < 1:
            raise ValueError("ways must be at least 1")
        if policy not in ("lru", "fifo"):
            raise ValueError("policy must be 'lru' or 'fifo'")
        self.sets, self.ways, self.block, self.policy = sets, ways, block, policy
        self.offset_bits, self.index_bits = int(math.log2(block)), int(math.log2(sets))
        self.lines: list[list[int | None]] = [[None] * ways for _ in range(sets)]
        self.order: list[list[int]] = [list(range(ways)) for _ in range(sets)]  # victim-first
        self.history: list[Access] = []

    def split(self, address: int) -> tuple[int, int, int]:
        if address < 0:
            raise ValueError("addresses must be non-negative")
        offset = address & (self.block - 1)
        index = (address >> self.offset_bits) & (self.sets - 1)
        return address >> (self.offset_bits + self.index_bits), index, offset

    def access(self, address: int) -> Access:
        tag, index, offset = self.split(int(address))
        lines, order = self.lines[index], self.order[index]
        way = next((w for w in range(self.ways) if lines[w] == tag), None)
        hit = way is not None
        evicted = None
        if hit:
            if self.policy == "lru":
                order.remove(way)
                order.append(way)
        else:
            invalid = [w for w in order if lines[w] is None]
            way = invalid[0] if invalid else order[0]
            evicted = lines[way]
            lines[way] = tag
            order.remove(way)
            order.append(way)
        acc = Access(int(address), tag, index, offset, hit, way, evicted, tuple(order),
                     tuple(tuple(s) for s in self.lines))
        self.history.append(acc)
        return acc


@dataclass(frozen=True)
class CacheResult:
    accesses: tuple[Access, ...]
    hits: int
    misses: int

    @property
    def hit_rate(self) -> float:
        return self.hits / max(1, len(self.accesses))


def cache_simulate(addresses: Sequence[int], sets: int = 4, ways: int = 2, block: int = 4, policy: str = "lru") -> CacheResult:
    """Replay ``addresses`` through a cache: per access the tag, index, offset, hit or miss, evicted tag and replacement order."""
    model = CacheModel(sets, ways, block, policy)
    out = tuple(model.access(a) for a in addresses)
    hits = sum(1 for a in out if a.hit)
    return CacheResult(out, hits, len(out) - hits)


def amat(hit_time, miss_rate, miss_penalty: float) -> float:
    """Average memory access time.

    Single level: ``hit_time + miss_rate * miss_penalty``. Multi-level: pass lists of per-level hit times and *local*
    miss rates, and ``miss_penalty`` as the main-memory time: ``h1 + m1 (h2 + m2 (... + mn * mem))``.
    """
    if isinstance(hit_time, (int, float)):
        if not 0.0 <= miss_rate <= 1.0:
            raise ValueError("miss_rate must be between 0 and 1")
        return hit_time + miss_rate * miss_penalty
    hits, misses = list(hit_time), list(miss_rate)
    if len(hits) != len(misses) or not hits:
        raise ValueError("hit_time and miss_rate need the same, non-zero number of levels")
    total = float(miss_penalty)
    for h, m in zip(reversed(hits), reversed(misses)):
        if not 0.0 <= m <= 1.0:
            raise ValueError("miss rates must be between 0 and 1")
        total = h + m * total
    return total


@dataclass(frozen=True)
class DMASchedule:
    owners: tuple[str, ...]  # who owns the bus in each cycle: "CPU", "DMA" or "idle"
    dma_cycles: int
    finish_cycle: int


def dma_schedule(words: int, cpu_busy: float = 0.5, mode: str = "steal") -> DMASchedule:
    """Bus ownership per cycle while DMA moves ``words`` words.

    ``burst`` takes the bus for ``words`` cycles in a row. ``steal`` gives DMA every cycle the CPU does not want
    (``cpu_busy`` is the fraction it wants) and guarantees DMA at least every fourth cycle so it always finishes.
    """
    if not 1 <= words <= 64:
        raise ValueError("words must be between 1 and 64")
    if not 0.0 <= cpu_busy <= 1.0:
        raise ValueError("cpu_busy must be between 0 and 1")
    if mode not in ("burst", "steal"):
        raise ValueError("mode must be 'burst' or 'steal'")
    if mode == "burst":
        return DMASchedule(("DMA",) * words, words, words)
    owners: list[str] = []
    left, i = words, 0
    while left > 0:
        wants = math.floor((i + 1) * cpu_busy) > math.floor(i * cpu_busy)
        if not wants or i % 4 == 3:
            owners.append("DMA")
            left -= 1
        else:
            owners.append("CPU")
        i += 1
    return DMASchedule(tuple(owners), words, len(owners))


# --------------------------------------------------------------------------------------------------------------------
# Serial buses. Every protocol returns a BusTimeline: step signals in bit-time units plus labelled fields and sample marks.
# --------------------------------------------------------------------------------------------------------------------
@dataclass
class BusTimeline:
    signals: dict[str, list[tuple[float, float]]]  # name -> [(time, level)], a step function starting at t = 0
    duration: float
    fields: list[tuple[float, float, str]] = field(default_factory=list)  # (t0, t1, label)
    samples: list[tuple[float, int, str]] = field(default_factory=list)  # (time, bit, signal name)
    bits: list[int] = field(default_factory=list)


def level_at(points: list[tuple[float, float]], t: float) -> float:
    """Value of a step function at time ``t`` (the last breakpoint at or before ``t``)."""
    cur = points[0][1]
    for pt, lv in points:
        if pt <= t + 1e-9:
            cur = lv
        else:
            break
    return cur


def _step(events: list[tuple[float, float]], start: float) -> list[tuple[float, float]]:
    """Collapse a list of (time, level) change events into a clean breakpoint list beginning at t = 0."""
    pts = [(0.0, start)]
    for t, lv in sorted(events, key=lambda e: e[0]):
        if abs(pts[-1][1] - lv) > 1e-12 or t <= 0:
            if t <= 0:
                pts[0] = (0.0, lv)
            else:
                pts.append((t, lv))
    return pts


def _to_bits(data: int | Sequence[int], bits: int) -> list[int]:
    items = [data] if isinstance(data, int) else list(data)
    if not items:
        raise ValueError("data must not be empty")
    out: list[int] = []
    for b in items:
        if not 0 <= b < (1 << bits):
            raise ValueError(f"each value must fit in {bits} bits")
        out += [(b >> (bits - 1 - i)) & 1 for i in range(bits)]
    return out


def spi_frame(data: int | Sequence[int], cpol: int = 0, cpha: int = 0, bits: int = 8,
              miso: int | Sequence[int] | None = None) -> BusTimeline:
    """Edge-accurate SPI frame (MSB first). Time is in bit periods.

    The first (leading) clock edge is at ``t = 1 + k`` for bit ``k`` and the trailing edge half a period later. With
    ``cpha = 0`` data changes half a period before the leading edge and is sampled on it; with ``cpha = 1`` data changes on
    the leading edge and is sampled on the trailing edge. ``cpol`` is the idle level of SCLK.
    """
    if cpol not in (0, 1) or cpha not in (0, 1):
        raise ValueError("cpol and cpha must each be 0 or 1")
    out = _to_bits(data, bits)
    n = len(out)
    other = _to_bits(miso, bits) if miso is not None else None
    if other is not None and len(other) != n:
        raise ValueError("miso must carry as many bits as mosi")
    clk: list[tuple[float, float]] = []
    lead, trail = 1 - cpol, cpol
    for k in range(n):
        clk += [(1.0 + k, lead), (1.5 + k, trail)]

    def line(bitlist: list[int]) -> list[tuple[float, float]]:
        ev = []
        for k, b in enumerate(bitlist):
            ev.append(((k + 0.5) if cpha == 0 else (k + 1.0), b))
        ev.append(((n + 0.5) if cpha == 0 else (n + 1.0), 0))  # bus parks low after the last bit
        return _step(ev, bitlist[0] if cpha == 0 else 0)

    cs = _step([(0.5, 0), (n + 1.5, 1)], 1)
    signals = {"SCLK": _step(clk, cpol), "MOSI": line(out)}
    if other is not None:
        signals["MISO"] = line(other)
    signals["CS"] = cs
    sample_t = [(1.0 + k) if cpha == 0 else (1.5 + k) for k in range(n)]
    samples = [(t, b, "MOSI") for t, b in zip(sample_t, out)]
    if other is not None:
        samples += [(t, b, "MISO") for t, b in zip(sample_t, other)]
    return BusTimeline(signals, n + 2.0, [(0.5, n + 1.5, "frame")], samples, out)


def i2c_frame(addr: int, data: int | Sequence[int], rw: str | int = "w") -> BusTimeline:
    """I2C transaction: START, 7-bit address, R/W, ACK, data bytes each followed by ACK, STOP. Time is in bit periods.

    SDA only changes while SCL is low, except for START (SDA falls while SCL is high) and STOP (SDA rises while SCL is high).
    On a read the slave drives the data and the master ACKs every byte but the last.
    """
    if not 0 <= addr < 128:
        raise ValueError("addr must be a 7-bit address (0 to 127)")
    read = rw in ("r", "read", 1)
    bytes_ = [data] if isinstance(data, int) else list(data)
    if not 1 <= len(bytes_) <= 4:
        raise ValueError("give between 1 and 4 data bytes")
    slots: list[tuple[int, str]] = []  # (SDA level, field label) per 1-bit slot
    slots += [((addr >> (6 - i)) & 1, "ADDR") for i in range(7)]
    slots += [(1 if read else 0, "R/W"), (0, "ACK")]
    for bi, byte in enumerate(bytes_):
        slots += [((byte >> (7 - i)) & 1, "DATA") for i in range(8)]
        nack = read and bi == len(bytes_) - 1
        slots.append((1 if nack else 0, "NACK" if nack else "ACK"))
    t0 = 1.0
    sda: list[tuple[float, float]] = [(0.5, 0)]  # START: SDA falls while SCL is high
    scl: list[tuple[float, float]] = [(1.0, 0)]
    samples = []
    for j, (lv, _) in enumerate(slots):
        t = t0 + j
        sda.append((t, lv))
        scl += [(t + 0.25, 1), (t + 0.75, 0)]
        samples.append((t + 0.5, lv, "SDA"))
    end = t0 + len(slots)
    sda.append((end, 0))
    scl.append((end + 0.25, 1))
    sda.append((end + 0.5, 1))  # STOP: SDA rises while SCL is high
    fields: list[tuple[float, float, str]] = [(0.5, 1.0, "START")]
    j = 0
    while j < len(slots):
        label = slots[j][1]
        k = j
        while k < len(slots) and slots[k][1] == label and (label != "ADDR" or k - j < 7):
            k += 1
        if label == "ADDR":
            text = f"ADDR 0x{addr:02X}"
        elif label == "DATA":
            val = bytes_[sum(1 for f in fields if f[2].startswith("DATA"))]
            text = f"DATA 0x{val:02X}"
        else:
            text = label
        fields.append((t0 + j, t0 + k, text))
        j = k
    fields.append((end + 0.25, end + 1.0, "STOP"))
    return BusTimeline({"SCL": _step(scl, 1), "SDA": _step(sda, 1)}, end + 1.25, fields, samples,
                       [s[0] for s in slots])


def uart_frame(byte: int, parity: str = "none", stop: int = 1, bits: int = 8) -> BusTimeline:
    """One UART frame, LSB first: idle (high), start (low), data, optional parity, stop bit(s). Time is in bit periods; sampled mid-bit."""
    if not 5 <= bits <= 9:
        raise ValueError("bits must be between 5 and 9")
    if not 0 <= byte < (1 << bits):
        raise ValueError(f"byte must fit in {bits} bits")
    if parity not in ("none", "even", "odd"):
        raise ValueError("parity must be 'none', 'even' or 'odd'")
    if stop not in (1, 2):
        raise ValueError("stop must be 1 or 2")
    data = [(byte >> i) & 1 for i in range(bits)]
    ones = sum(data)
    seq: list[tuple[int, str]] = [(0, "START")] + [(b, f"D{i}") for i, b in enumerate(data)]
    if parity != "none":
        seq.append(((ones % 2) if parity == "even" else (1 - ones % 2), "PAR"))
    seq += [(1, "STOP")] * stop
    ev, fields, samples = [], [(0.0, 1.0, "IDLE")], []
    for i, (lv, name) in enumerate(seq):
        t = 1.0 + i
        ev.append((t, lv))
        fields.append((t, t + 1, name))
        samples.append((t + 0.5, lv, "TX"))
    end = 1.0 + len(seq)
    return BusTimeline({"TX": _step(ev, 1)}, end + 1.0, fields + [(end, end + 1.0, "IDLE")], samples, [s[0] for s in seq])


@dataclass(frozen=True)
class NRZIResult:
    data: tuple[int, ...]  # the bit stream after stuffing
    stuffed: tuple[int, ...]  # indices into ``data`` that are stuffed bits
    levels: tuple[int, ...]  # line level (1 = J, 0 = K) during each bit


def nrzi_encode(bits: Sequence[int], stuffing: bool = True) -> NRZIResult:
    """USB-style NRZI: a 0 toggles the line, a 1 keeps it. With ``stuffing`` a 0 is inserted after every six 1s in a row, so
    a transition is guaranteed often enough for the receiver to keep its clock (a run of 1s never exceeds six)."""
    out: list[int] = []
    stuffed: list[int] = []
    run = 0
    for b in bits:
        if b not in (0, 1):
            raise ValueError("bits must be 0 or 1")
        out.append(b)
        run = run + 1 if b == 1 else 0
        if stuffing and run == 6:
            stuffed.append(len(out))
            out.append(0)
            run = 0
    level, levels = 1, []
    for b in out:
        if b == 0:
            level ^= 1
        levels.append(level)
    return NRZIResult(tuple(out), tuple(stuffed), tuple(levels))


_CAN_POLY = 0x4599


def can_crc15(bits: Sequence[int]) -> int:
    """CAN CRC-15 (polynomial 0x4599) over the bits from SOF to the end of the data field."""
    crc = 0
    for b in bits:
        nxt = b ^ ((crc >> 14) & 1)
        crc = (crc << 1) & 0x7FFF
        if nxt:
            crc ^= _CAN_POLY
    return crc


@dataclass(frozen=True)
class CANFrame:
    bits: tuple[int, ...]  # the full stuffed bit stream, 0 = dominant, 1 = recessive
    fields: tuple[tuple[int, int, str], ...]  # (first index, one past last, name) over ``bits``
    stuffed: tuple[int, ...]  # indices of stuff bits
    crc: int


def can_frame(can_id: int, data: Sequence[int] = (), stuffing: bool = True, ack: bool = True) -> CANFrame:
    """A standard (11-bit identifier) CAN data frame with bit stuffing. Stuffing runs from SOF through the CRC: after five equal
    bits the opposite bit is inserted, so six equal bits in a row never occur there."""
    if not 0 <= can_id < 2048:
        raise ValueError("can_id must fit in 11 bits (0 to 2047)")
    payload = list(data)
    if len(payload) > 8 or any(not 0 <= b < 256 for b in payload):
        raise ValueError("a CAN frame carries at most 8 data bytes, each 0 to 255")
    head = [("SOF", [0]), ("ID", [(can_id >> (10 - i)) & 1 for i in range(11)]), ("RTR", [0]), ("IDE", [0]), ("r0", [0]),
            ("DLC", [(len(payload) >> (3 - i)) & 1 for i in range(4)])]
    if payload:
        head.append(("DATA", [(b >> (7 - i)) & 1 for b in payload for i in range(8)]))
    crc = can_crc15([b for _, bs in head for b in bs])
    head.append(("CRC", [(crc >> (14 - i)) & 1 for i in range(15)]))
    out: list[int] = []
    fields: list[tuple[int, int, str]] = []
    stuffed: list[int] = []
    run, last = 0, None
    for name, bs in head:
        start = len(out)
        for b in bs:
            out.append(b)
            run = run + 1 if b == last else 1
            last = b
            if stuffing and run == 5:
                stuffed.append(len(out))
                out.append(1 - b)
                last, run = 1 - b, 1
        fields.append((start, len(out), name))
    tail = [("CRC del", [1]), ("ACK", [0 if ack else 1]), ("ACK del", [1]), ("EOF", [1] * 7), ("IFS", [1] * 3)]
    for name, bs in tail:
        fields.append((len(out), len(out) + len(bs), name))
        out += bs
    return CANFrame(tuple(out), tuple(fields), tuple(stuffed), crc)


@dataclass(frozen=True)
class ArbitrationResult:
    ids: tuple[int, ...]
    bits: tuple[tuple[int, ...], ...]  # per node, its 11 identifier bits
    bus: tuple[int, ...]  # the wired-AND level on the bus at each bit
    lost_at: tuple[int | None, ...]  # per node, the bit where it saw a dominant bus while sending recessive
    winner: int


def can_arbitration(ids: Sequence[int]) -> ArbitrationResult:
    """Bitwise CAN arbitration: dominant (0) beats recessive (1), so the lowest identifier wins and the others drop out."""
    ids = tuple(ids)
    if not 2 <= len(ids) <= 4:
        raise ValueError("give between 2 and 4 node identifiers")
    if len(set(ids)) != len(ids):
        raise ValueError("identifiers must be unique: two nodes sending the same ID would collide")
    if any(not 0 <= i < 2048 for i in ids):
        raise ValueError("identifiers must fit in 11 bits (0 to 2047)")
    bits = tuple(tuple((i >> (10 - k)) & 1 for k in range(11)) for i in ids)
    active = set(range(len(ids)))
    lost: list[int | None] = [None] * len(ids)
    bus = []
    for k in range(11):
        level = min(bits[n][k] for n in active)
        bus.append(level)
        for n in list(active):
            if bits[n][k] == 1 and level == 0:
                lost[n] = k
                active.discard(n)
    return ArbitrationResult(ids, bits, tuple(bus), tuple(lost), min(range(len(ids)), key=lambda n: ids[n]))


@dataclass(frozen=True)
class DifferentialData:
    t: np.ndarray
    dplus: np.ndarray
    dminus: np.ndarray
    diff: np.ndarray
    common: np.ndarray


def differential(bits: Sequence[int], noise: float = 0.0, seed: int = 7, samples_per_bit: int = 8,
                 common_mode: float = 1.0) -> DifferentialData:
    """D+ and D- for a bit sequence plus equal noise on both lines. The difference ``D+ - D-`` is noise-free; the noise lives entirely in the common mode."""
    if not bits or any(b not in (0, 1) for b in bits):
        raise ValueError("bits must be a non-empty list of 0 and 1")
    if samples_per_bit < 2:
        raise ValueError("samples_per_bit must be at least 2")
    d = np.repeat([1.0 if b else -1.0 for b in bits], samples_per_bit)
    n = len(d)
    t = np.arange(n) / samples_per_bit
    if noise > 0:
        raw = rng(seed).normal(0.0, 1.0, n + 6)
        smooth = np.convolve(raw, np.ones(4) / 4.0, mode="valid")[:n]
        nz = noise * smooth / max(float(np.std(smooth)), 1e-9) * 0.5
    else:
        nz = np.zeros(n)
    dp, dm = common_mode + 0.5 * d + nz, common_mode - 0.5 * d + nz
    return DifferentialData(t, dp, dm, dp - dm, 0.5 * (dp + dm))
