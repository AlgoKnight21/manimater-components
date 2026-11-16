"""CPU pipeline, hazards, branch prediction and the datapath (README section 7.3)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, AnimationGroup, Arrow, Create, CurvedArrow, FadeIn, FadeOut, Indicate, Line, Rectangle,
                   Succession, Transform, VGroup, Circle, MoveAlongPath, TAU)

from ...ctx import label
from ..core.glow import GlowDot
from ..core.palette import PALETTE
from ._frame import Framed
from ._math import (PipelineSchedule, PredictorData, pipeline_schedule, predictor_run, resolve_program)

STAGE_COLOR = {"IF": PALETTE.blue, "ID": PALETTE.teal, "EX": PALETTE.purple, "MEM": PALETTE.green, "WB": PALETTE.muted}
CAUSE_TEXT = {"RAW": "RAW", "load-use": "LU", "control": "CTL"}
LABEL_W = 2.7


@dataclass
class _Cell:
    rect: Rectangle
    text: object
    kind: str
    fill: float
    stroke: float
    row: int
    cycle: int


class PipelineDiagram(Framed, VGroup):
    """Intuition: a laundromat line. Each instruction moves one stage per cycle; a hazard makes the one behind it wait.

    Rows are instructions (top to bottom), columns are clock cycles. Orange cells are bubbles labelled with their cause
    (RAW data hazard, LU load-use, CTL control); red cells are wrong-path instructions squashed after a mispredicted branch.
    Cells start hidden: ``animate_cycle(n)`` advances a yellow clock marker and reveals what happens in cycle ``n``.
    """

    def __init__(self, program="raw_hazard", stages=("IF", "ID", "EX", "MEM", "WB"), forwarding: bool = True,
                 branch_policy: str = "stall", *, hidden: bool = True, cell_w: float = 0.6, cell_h: float = 0.46):
        super().__init__()
        self._init_frame()
        self.instrs = resolve_program(program)
        self.schedule: PipelineSchedule = pipeline_schedule(self.instrs, forwarding, branch_policy, tuple(stages))
        self.cw, self.ch, self.forwarding = cell_w, cell_h, forwarding
        sch = self.schedule
        n_rows = len(sch.rows)
        self.row_of_instr = [i for i, r in enumerate(sch.rows) if not r.squashed]
        self.cells: dict[tuple[int, int], _Cell] = {}
        self.cycle_cells: dict[int, list[_Cell]] = {c: [] for c in range(1, sch.total_cycles + 1)}
        head = VGroup(label("cycle", 18, PALETTE.muted).move_to((LABEL_W - 0.55, 0, 0)))
        for c in range(1, sch.total_cycles + 1):
            head.add(label(str(c), 18, PALETTE.muted).move_to((self.col_x(c), 0, 0)))
        names = VGroup()
        for r, row in enumerate(sch.rows):
            t = label(row.text, 20, PALETTE.red if row.squashed else PALETTE.fg)
            t.move_to((LABEL_W - 0.2 - t.width / 2 - 0.35, self.row_y(r), 0))
            names.add(t)
        self.add(head, names)
        for r, row in enumerate(sch.rows):
            for cell in row.cells:
                self._add_cell(r, cell, hidden)
        self.clock = Rectangle(width=cell_w, height=cell_h * (n_rows + 1) + 0.1, stroke_color=PALETTE.yellow, stroke_width=3,
                               fill_opacity=0).set_stroke(opacity=0.0)
        self.clock.move_to((self.col_x(1), -(n_rows * cell_h) / 2 - 0.0, 0))
        self.add(self.clock)
        self.move_to((0, 0, 0))

    # -- layout ---------------------------------------------------------------------------------------------------------
    def col_x(self, cycle: int) -> float:
        return LABEL_W + (cycle - 0.5) * self.cw

    def row_y(self, row: int) -> float:
        return -(row + 1.2) * self.ch

    def _add_cell(self, r: int, cell, hidden: bool) -> None:
        kind = cell.kind
        if kind == "stage":
            col, fill = STAGE_COLOR[cell.label], 0.4
        elif kind == "stall":
            col, fill = PALETTE.orange, 0.55
        elif kind == "hold":
            col, fill = STAGE_COLOR["IF"], 0.12
        else:  # flush
            col, fill = PALETTE.red, 0.35
        txt = CAUSE_TEXT.get(cell.cause, cell.cause) if kind == "stall" else cell.label
        rect = Rectangle(width=self.cw - 0.06, height=self.ch - 0.07, stroke_color=col, stroke_width=2, fill_color=col,
                         fill_opacity=fill)
        rect.move_to((self.col_x(cell.cycle), self.row_y(r), 0))
        text = label(txt, 16, PALETTE.fg).move_to(rect)
        c = _Cell(rect, text, kind, fill, 1.0, r, cell.cycle)
        self.cells[(r, cell.cycle)] = c
        self.cycle_cells[cell.cycle].append(c)
        if hidden:
            rect.set_fill(opacity=0).set_stroke(opacity=0)
            text.set_opacity(0)
        self.add(rect, text)

    # -- animations -----------------------------------------------------------------------------------------------------
    def _reveal(self, cells: list[_Cell]) -> list:
        out = []
        for c in cells:
            out += [c.rect.animate.set_fill(opacity=c.fill).set_stroke(opacity=c.stroke), c.text.animate.set_opacity(1.0)]
        return out

    def animate_cycle(self, n: int, *, run_time: float = 0.8):
        """Advance the clock to cycle ``n`` (1-based) and reveal that cycle's cells."""
        if not 1 <= n <= self.schedule.total_cycles:
            raise ValueError(f"cycle must be between 1 and {self.schedule.total_cycles}")
        target = self.pt(self.col_x(n), self.clock_center_y())
        return AnimationGroup(self.clock.animate.move_to(target).set_stroke(opacity=1.0), *self._reveal(self.cycle_cells[n]),
                              run_time=run_time)

    def clock_center_y(self) -> float:
        return -(len(self.schedule.rows) * self.ch) / 2 - 0.1

    def reveal_all(self, *, run_time: float = 1.0):
        return AnimationGroup(*self._reveal([c for c in self.cells.values()]), run_time=run_time)

    def stall_cells(self, instr: int) -> list[_Cell]:
        r = self.row_of_instr[instr]
        return [c for (rr, _), c in sorted(self.cells.items(), key=lambda kv: kv[0][1]) if rr == r and c.kind == "stall"]

    def animate_stall(self, instr: int, cycles: int | None = None, *, run_time: float = 1.0):
        """Show the bubbles that hold instruction ``instr`` (program order, 0-based), labelled with their cause."""
        if not 0 <= instr < len(self.instrs):
            raise ValueError("instr is out of range")
        cells = self.stall_cells(instr)
        if not cells:
            raise ValueError(f"instruction {instr} does not stall under this configuration")
        if cycles is not None:
            cells = cells[:max(1, cycles)]
        return AnimationGroup(*self._reveal(cells), *[Indicate(c.rect, color=PALETTE.orange) for c in cells], run_time=run_time)

    def forward_edges(self) -> list[tuple[int, int, int, int]]:
        """Forwarding edges as ``(producer instr, consumer instr, from cycle, to cycle)`` in program order."""
        inv = {r: i for i, r in enumerate(self.row_of_instr)}
        return [(inv[a], inv[b], c1, c2) for a, b, c1, c2 in self.schedule.forwards]

    def forward_arrow(self, src: int, dst: int):
        edge = next((e for e in self.schedule.forwards
                     if (self.row_of_instr[src], self.row_of_instr[dst]) == (e[0], e[1])), None)
        if edge is None:
            raise ValueError(f"no forwarding path from instruction {src} to {dst} in this schedule")
        a, b = self.cells[(edge[0], edge[2])], self.cells[(edge[1], edge[3])]
        arrow = CurvedArrow(a.rect.get_bottom(), b.rect.get_top(), angle=-TAU / 8, color=PALETTE.yellow, stroke_width=3, tip_length=0.13)
        return arrow, (a, b)

    def animate_forward(self, src: int, dst: int, *, run_time: float = 0.9):
        """Draw the forwarding path that hands the result of ``src`` straight to ``dst``, skipping the register file."""
        arrow, cells = self.forward_arrow(src, dst)
        return AnimationGroup(*self._reveal(list(cells)), Create(arrow), run_time=run_time)

    @property
    def data(self) -> PipelineSchedule:
        return self.schedule


class HazardTimeline(VGroup):
    """Intuition: the same program run twice, without and with forwarding, so the saved cycles are visible side by side.

    Both pipelines are fully drawn. Yellow arrows on the lower one are the forwarding paths that removed the stalls.
    """

    def __init__(self, program="raw_hazard", *, scale: float = 0.78):
        super().__init__()
        self.slow = PipelineDiagram(program, forwarding=False, hidden=False)
        self.fast = PipelineDiagram(program, forwarding=True, hidden=False)
        for d in (self.slow, self.fast):
            d.scale(scale)
        self.title_slow = label(f"no forwarding: {self.slow.schedule.total_cycles} cycles", 24, PALETTE.orange)
        self.title_fast = label(f"with forwarding: {self.fast.schedule.total_cycles} cycles", 24, PALETTE.green)
        self.slow.next_to(self.title_slow, DOWN, buff=0.2)
        top = VGroup(self.title_slow, self.slow)
        self.fast.next_to(self.title_fast, DOWN, buff=0.2)
        bottom = VGroup(self.title_fast, self.fast)
        VGroup(top, bottom).arrange(DOWN, buff=0.45)
        self.arrows = VGroup()
        for a, b, *_ in self.fast.schedule.forwards:
            inv = {r: i for i, r in enumerate(self.fast.row_of_instr)}
            arrow, _ = self.fast.forward_arrow(inv[a], inv[b])
            self.arrows.add(arrow)
        self.add(top, bottom, self.arrows)
        self.move_to((0, 0, 0))

    @property
    def saved_cycles(self) -> int:
        return self.slow.schedule.total_cycles - self.fast.schedule.total_cycles

    @property
    def data(self) -> dict:
        return {"slow_cycles": self.slow.schedule.total_cycles, "fast_cycles": self.fast.schedule.total_cycles,
                "saved": self.saved_cycles, "slow_stalls": self.slow.schedule.stall_cycles,
                "fast_stalls": self.fast.schedule.stall_cycles}

    def animate_compare(self, *, run_time: float = 2.0):
        """Pulse the slow total, then the fast one, then the forwarding arrows."""
        t = run_time / 3
        return Succession(Indicate(self.title_slow, color=PALETTE.orange, run_time=t), Indicate(self.title_fast, run_time=t),
                          Indicate(self.arrows, color=PALETTE.yellow, run_time=t) if len(self.arrows) else Indicate(self.title_fast, run_time=t))


def _chip(text: str, color: str, w: float = 1.55) -> VGroup:
    rect = Rectangle(width=w, height=0.46, stroke_color=color, stroke_width=2, fill_color=color, fill_opacity=0.3)
    return VGroup(rect, label(text, 18, PALETTE.fg).move_to(rect))


class _Arm(VGroup):
    """One side of a branch fork: a line out of the branch and the instructions that would follow on that path."""

    def __init__(self, start, bend, texts: list[str], direction_y: float, name: str):
        super().__init__()
        self.line = Line(start, bend, color=PALETTE.muted, stroke_width=4)
        self.chips = VGroup(*[_chip(t, PALETTE.muted) for t in texts]).arrange(RIGHT, buff=0.25)
        self.chips.next_to(np.array(bend), RIGHT, buff=0.25)
        self.tag = label(name, 20, PALETTE.muted).next_to(self.line, UP if direction_y > 0 else DOWN, buff=0.1)
        self.add(self.line, self.chips, self.tag)

    def paint(self, color: str, faded: bool = False) -> list:
        fade = 0.25 if faded else 1.0
        anims = [self.line.animate.set_stroke(color=color, opacity=fade), self.tag.animate.set_color(color).set_opacity(fade)]
        for chip in self.chips:
            rect, text = chip
            anims += [rect.animate.set_stroke(color=color, opacity=fade).set_fill(color=color, opacity=0.3 * fade),
                      text.animate.set_opacity(fade)]
        return anims


class BranchPredictionPath(Framed, VGroup):
    """Intuition: the CPU guesses a fork in the road and starts walking before it knows. A right guess costs nothing; a wrong one throws the steps away.

    The blue arm is the prediction, green the path that was actually taken. If they disagree the predicted path's
    instructions are squashed (red, faded) and the status line shows the two-cycle penalty.
    """

    PENALTY = 2

    def __init__(self, program="branch", predictor: str = "2bit", outcomes=(1, 1, 0, 1)):
        super().__init__()
        self._init_frame()
        outcomes = tuple(outcomes)
        if not 1 <= len(outcomes) <= 8:
            raise ValueError("give between 1 and 8 outcomes")
        self.instrs = resolve_program(program)
        self.run: PredictorData = predictor_run(outcomes, predictor)
        self.outcomes = outcomes
        idx = next((i for i, ins in enumerate(self.instrs) if ins.op == "branch"), 0)
        follow = [ins.text[:14] for ins in self.instrs[idx + 1: idx + 3]] + ["next", "next"]
        branch = _chip(self.instrs[idx].text[:14], PALETTE.yellow, 1.7).move_to((-3.0, 0, 0))
        stream = Line((-5.6, 0, 0), (-3.85, 0, 0), color=PALETTE.muted, stroke_width=4)
        self.taken = _Arm((-2.2, 0.0, 0), (-1.2, 1.3, 0), ["target 1", "target 2"], 1, "taken")
        self.not_taken = _Arm((-2.2, 0.0, 0), (-1.2, -1.3, 0), follow[:2], -1, "not taken")
        self.status = label(f"predictor: {predictor}", 24, PALETTE.fg).move_to((0, -2.45, 0))
        self.add(stream, branch, self.taken, self.not_taken, self.status)
        self.move_to((0, 0, 0))

    @property
    def data(self) -> PredictorData:
        return self.run

    @property
    def penalty_cycles(self) -> int:
        """Cycles wasted over the whole run (two per misprediction)."""
        return self.PENALTY * self.run.mispredicts

    def animate_outcome(self, i: int, *, run_time: float = 2.0):
        """Play branch ``i``: show the prediction, then the real outcome, then either a tick or a flush."""
        if not 0 <= i < len(self.outcomes):
            raise ValueError("outcome index out of range")
        pred, actual = self.run.predictions[i], self.outcomes[i]
        arm = lambda t: self.taken if t else self.not_taken
        word = lambda t: "taken" if t else "not taken"
        hit = pred == actual
        verdict = "correct: no bubbles" if hit else f"wrong: {self.PENALTY} cycles flushed"
        text = label(f"branch {i + 1}: predicted {word(pred)}, actual {word(actual)} - {verdict}", 22,
                     PALETTE.green if hit else PALETTE.red).scale(self.unit).move_to(self.status)
        t = run_time / 3
        guess = AnimationGroup(*arm(pred).paint(PALETTE.blue), *arm(1 - pred).paint(PALETTE.muted), run_time=t)
        if hit:
            outcome = AnimationGroup(*arm(pred).paint(PALETTE.green), run_time=t)
        else:
            outcome = AnimationGroup(*arm(pred).paint(PALETTE.red, faded=True), *arm(actual).paint(PALETTE.green), run_time=t)
        return Succession(guess, Transform(self.status, text, run_time=t), outcome)


class PredictorFSM(VGroup):
    """Intuition: a confidence meter. Each taken branch nudges it toward "taken", each not-taken branch nudges it back; it takes two misses to change its mind.

    The 2-bit version has four states (the top bit is the prediction); the 1-bit version flips every time it is wrong.
    A yellow ring marks the current state.
    """

    def __init__(self, bits: int = 2, state: int = 0):
        super().__init__()
        if bits not in (1, 2):
            raise ValueError("bits must be 1 or 2")
        self.bits, self.n = bits, 2 ** bits
        names = ["strongly not taken", "weakly not taken", "weakly taken", "strongly taken"] if bits == 2 else ["predict not taken", "predict taken"]
        gap = 2.2
        xs = [(i - (self.n - 1) / 2) * gap for i in range(self.n)]
        self.circles = VGroup()
        self.t_arrows, self.nt_arrows = [], []
        for i, x in enumerate(xs):
            taken_side = i >= self.n / 2
            col = PALETTE.teal if taken_side else PALETTE.blue
            c = Circle(radius=0.5, color=col, stroke_width=3, fill_color=col, fill_opacity=0.15).move_to((x, 0, 0))
            self.circles.add(VGroup(c, label(format(i, f"0{bits}b"), 28, PALETTE.fg).move_to(c),
                                    label(names[i], 16, PALETTE.muted).move_to((x, -0.85, 0))))
        self.arrows = VGroup()
        for i in range(self.n - 1):
            a = Arrow((xs[i] + 0.5, 0.2, 0), (xs[i + 1] - 0.5, 0.2, 0), buff=0.05, color=PALETTE.green, stroke_width=3,
                      max_tip_length_to_length_ratio=0.2)
            b = Arrow((xs[i + 1] - 0.5, -0.2, 0), (xs[i] + 0.5, -0.2, 0), buff=0.05, color=PALETTE.red, stroke_width=3,
                      max_tip_length_to_length_ratio=0.2)
            self.t_arrows.append(a)
            self.nt_arrows.append(b)
            self.arrows.add(a, b, label("T", 16, PALETTE.green).move_to(((xs[i] + xs[i + 1]) / 2, 0.45, 0)),
                            label("NT", 16, PALETTE.red).move_to(((xs[i] + xs[i + 1]) / 2, -0.45, 0)))
        self.loops = VGroup(label("T stays", 16, PALETTE.green).move_to((xs[-1], 0.85, 0)),
                            label("NT stays", 16, PALETTE.red).move_to((xs[0], 0.85, 0)))
        self.marker = Circle(radius=0.62, color=PALETTE.yellow, stroke_width=5)
        self.state = max(0, min(self.n - 1, state))
        self.marker.move_to(self.circles[self.state][0].get_center())
        self.add(self.circles, self.arrows, self.loops, self.marker)
        self.move_to((0, 0, 0))
        self.marker.move_to(self.circles[self.state][0].get_center())

    def set_state(self, state: int) -> "PredictorFSM":
        if not 0 <= state < self.n:
            raise ValueError(f"state must be between 0 and {self.n - 1}")
        self.state = state
        self.marker.move_to(self.circles[state][0].get_center())
        return self

    def animate_to(self, state: int, *, run_time: float = 0.6):
        if not 0 <= state < self.n:
            raise ValueError(f"state must be between 0 and {self.n - 1}")
        self.state = state
        return self.marker.animate.move_to(self.circles[state][0].get_center()).build().set_run_time(run_time)

    def animate_run(self, outcomes, *, run_time: float = 4.0):
        """Feed a list of branch outcomes (1 = taken) through the counter, hopping the marker with each one."""
        outcomes = list(outcomes)
        if not outcomes:
            raise ValueError("outcomes must not be empty")
        data = predictor_run(outcomes, "2bit" if self.bits == 2 else "1bit", self.state)
        states = list(data.states[1:]) + [data.final_state]
        per = run_time / len(outcomes)
        steps, cur = [], self.state
        for o, nxt in zip(outcomes, states):
            moved = nxt != cur
            lo = min(cur, nxt)
            arrow = (self.t_arrows[lo] if o else self.nt_arrows[lo]) if moved else None
            parts = [self.marker.animate.move_to(self.circles[nxt][0].get_center())]
            if arrow is not None:
                parts.append(Indicate(arrow, color=PALETTE.yellow))
            steps.append(AnimationGroup(*parts, run_time=per))
            cur = nxt
        self.state = cur
        return Succession(*steps)


def _block(name: str, x: float, y: float, color: str, w: float = 1.8, h: float = 1.1) -> VGroup:
    rect = Rectangle(width=w, height=h, stroke_color=color, stroke_width=3, fill_color=color, fill_opacity=0.15)
    return VGroup(rect, label(name, 22, PALETTE.fg).move_to(rect)).move_to((x, y, 0))


class CPUDatapath(VGroup):
    """Intuition: an assembly line for one instruction. It is fetched, decoded, executed, touches memory, and the result is written back.

    ``animate_instruction`` sends a glowing token through PC, instruction memory, registers, ALU and data memory, then back
    along the write-back wire to the registers. ``show_control`` adds the control unit and its signals (dashed).
    """

    NAMES = ("PC", "Instr Mem", "Registers", "ALU", "Data Mem")

    def __init__(self, program="raw_hazard", show_control: bool = False):
        super().__init__()
        self.instrs = resolve_program(program)
        self.show_control = show_control
        xs = [-4.8, -2.4, 0.0, 2.4, 4.8]
        colors = [PALETTE.blue, PALETTE.teal, PALETTE.purple, PALETTE.orange, PALETTE.green]
        self.blocks = [_block(n, x, 0.0, c) for n, x, c in zip(self.NAMES, xs, colors)]
        self.wires = VGroup(*[Arrow((xs[i] + 0.9, 0, 0), (xs[i + 1] - 0.9, 0, 0), buff=0.04, color=PALETTE.muted, stroke_width=4,
                                    max_tip_length_to_length_ratio=0.35) for i in range(4)])
        self.plus4 = _block("+4", xs[0], 1.7, PALETTE.muted, 0.9, 0.55)
        self.pc_loop = VGroup(Arrow((xs[0] - 0.3, 0.55, 0), (xs[0] - 0.3, 1.4, 0), buff=0.03, color=PALETTE.muted, stroke_width=3),
                              Arrow((xs[0] + 0.3, 1.4, 0), (xs[0] + 0.3, 0.55, 0), buff=0.03, color=PALETTE.muted, stroke_width=3))
        self.writeback = VGroup(Line((xs[4], -0.55, 0), (xs[4], -1.45, 0)), Line((xs[4], -1.45, 0), (xs[2] + 0.3, -1.45, 0)),
                                Arrow((xs[2] + 0.3, -1.45, 0), (xs[2] + 0.3, -0.58, 0), buff=0.0, max_tip_length_to_length_ratio=0.3))
        self.writeback.set_stroke(color=PALETTE.green, width=4)
        self.wb_label = label("write back", 20, PALETTE.green).move_to(((xs[2] + xs[4]) / 2 + 0.3, -1.75, 0))
        self.add(self.wires, self.pc_loop, self.plus4, self.writeback, self.wb_label, *self.blocks)
        self.control = VGroup()
        if show_control:
            ctl = _block("Control", 0.0, 2.2, PALETTE.yellow, 2.2, 0.7)
            sig = VGroup(*[Line((x, 1.85, 0), (x, 0.55, 0), color=PALETTE.yellow, stroke_width=2, stroke_opacity=0.7)
                           for x in (xs[2], xs[3], xs[4])])
            self.control.add(ctl, sig)
            self.add(self.control)
        self.token = GlowDot((xs[0] - 1.25, 0, 0), radius=0.13, color=PALETTE.yellow)
        self.token_tag = label(self.instrs[0].text.split()[0], 20, PALETTE.yellow)
        self.token_tag.add_updater(lambda m: m.next_to(self.token, UP, buff=0.18))
        self.add(self.token, self.token_tag)
        self.move_to((0, 0, 0))
        self._start = self.token.get_center().copy()

    @property
    def data(self) -> dict:
        return {"instruction": self.instrs[0].text, "path": list(self.NAMES), "control": self.show_control}

    def animate_instruction(self, *, run_time: float = 4.0):
        """Send the first instruction of the program once through all five blocks and back along the write-back wire."""
        steps = len(self.blocks) + 2
        t = run_time / steps
        anims = []
        for blk in self.blocks:
            anims.append(AnimationGroup(self.token.animate.move_to(blk.get_center()), Indicate(blk[0], color=PALETTE.yellow, scale_factor=1.06),
                                        run_time=t))
        down = self.blocks[4].get_center() + DOWN * 1.45
        anims.append(self.token.animate.move_to(down).build().set_run_time(t))
        anims.append(self.token.animate.move_to(self.blocks[2].get_center() + np.array([0.3, -1.45, 0])).build().set_run_time(t * 0.6))
        anims.append(AnimationGroup(Indicate(self.blocks[2][0], color=PALETTE.green, scale_factor=1.06),
                                    self.token.animate.move_to(self.blocks[2].get_center()), run_time=t * 0.6))
        return Succession(*anims)
