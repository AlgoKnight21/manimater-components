"""Core visual devices from ``manimate.viz.core`` as storyboard components (viz README sections 5 and 12).

The helpers (Spotlight, FocusRing, Block, Wire, pulse_along) already existed; until now the planner could not ask for them because
nothing registered them. Each component here is a thin adapter: it lays the items out, plays the device, and fills the scene time.
"""
from __future__ import annotations

from manim import RIGHT, FadeIn, LaggedStart, Line, VGroup

from ..ctx import Ctx
from ..registry import Param as P, component
from ..viz.core import Block, FocusRing, Spotlight, Wire
from ..viz.core.flow import PulseTrain
from ..viz.core.palette import PALETTE

_D = "general"
_ITEMS_HINT = "2 to 6 short labels (a few words each)"
_COLORS = [PALETTE.blue, PALETTE.green, PALETTE.yellow, PALETTE.red, PALETTE.blue, PALETTE.green]


def _labels(v: list) -> str | None:
    if not 2 <= len(v) <= 6:
        return "needs 2 to 6 items"
    return None if all(isinstance(t, str) and 0 < len(t.strip()) <= 28 for t in v) else "each item must be a short non-empty string (28 characters at most)"


def _cards(items: list[str]) -> list[Block]:
    return [Block(t.strip(), w=2.2, h=1.1, color=_COLORS[i % len(_COLORS)], size=24) for i, t in enumerate(items)]


def _row(cards: list[Block]) -> VGroup:
    return VGroup(*cards).arrange(RIGHT, buff=0.45)


def _clamp(i: int, n: int) -> int:
    return max(0, min(n - 1, int(i)))


def _title(ctx: Ctx, title: str):
    if not title.strip():
        return None
    h = ctx.heading(title.strip())
    ctx.play(FadeIn(h), run_time=0.5)
    return h


@component(
    "spotlight",
    "Stage-light focus: a row of ideas appears, then everything dims except the one the narration is about, then the lights come back up. "
    "Good for 'of all these, this is the one that matters'.",
    {"items": P("string[]", f"the ideas shown side by side ({_ITEMS_HINT})", required=True, validator=_labels),
     "focus": P("integer", "which item to light up, counting from 0", default=0),
     "title": P("string", "optional heading above the row", default="")},
    {"items": ["Input", "Hidden layer", "Output"], "focus": 1, "title": "Where the learning happens"},
    domain=_D,
)
def spotlight(ctx: Ctx, items: list | None = None, focus: int = 0, title: str = "") -> None:
    items = items or ["One", "Two"]
    row = ctx.fit(_row(_cards(items)), max_h=3.0)
    _title(ctx, title)
    beat = Ctx.beat(ctx.remaining, 3)
    ctx.play(LaggedStart(*(FadeIn(c) for c in row), lag_ratio=0.25), run_time=beat)
    cards = list(row)
    light = Spotlight(cards[_clamp(focus, len(cards))], cards)
    ctx.play(light.animate_on(), run_time=0.7)
    ctx.wait(max(0.0, ctx.remaining - 0.9))
    ctx.play(light.animate_off(), run_time=0.7)
    ctx.finish()


@component(
    "focus_ring",
    "A pulsing ring drawn round one idea in a row to say 'look here', optionally moving on to a second one.",
    {"items": P("string[]", f"the ideas shown side by side ({_ITEMS_HINT})", required=True, validator=_labels),
     "focus": P("integer", "which item to ring first, counting from 0", default=0),
     "then": P("integer", "optional second item to move the ring to, counting from 0; -1 for none", default=-1),
     "title": P("string", "optional heading above the row", default="")},
    {"items": ["Request", "Cache", "Database"], "focus": 1, "then": 2, "title": ""},
    domain=_D,
)
def focus_ring(ctx: Ctx, items: list | None = None, focus: int = 0, then: int = -1, title: str = "") -> None:
    items = items or ["One", "Two"]
    row = ctx.fit(_row(_cards(items)), max_h=3.0)
    _title(ctx, title)
    ctx.play(LaggedStart(*(FadeIn(c) for c in row), lag_ratio=0.25), run_time=Ctx.beat(ctx.remaining, 3))
    cards = list(row)
    ring = FocusRing(cards[_clamp(focus, len(cards))])
    ctx.play(ring.animate_in(), run_time=0.6)
    if int(then) >= 0:
        ctx.wait(max(0.0, ctx.remaining / 2 - 0.4))
        ctx.play(ring.animate.move_to(cards[_clamp(then, len(cards))]), run_time=0.7)
    ctx.finish()


@component(
    "block_flow",
    "A left-to-right chain of labelled blocks joined by arrows, with a bright pulse travelling through it to show how a signal, "
    "request or piece of data moves from the first block to the last.",
    {"blocks": P("string[]", f"the stages in order ({_ITEMS_HINT})", required=True, validator=_labels),
     "title": P("string", "optional heading above the chain", default="")},
    {"blocks": ["Sensor", "Filter", "Controller", "Motor"], "title": "How the signal flows"},
    domain=_D,
)
def block_flow(ctx: Ctx, blocks: list | None = None, title: str = "") -> None:
    blocks = blocks or ["Input", "Output"]
    cards = _cards(blocks)
    for c in cards:
        c.scale(0.8)
    row = _row(cards).arrange(RIGHT, buff=0.8)
    ctx.fit(row, max_h=3.0)
    wires = [Wire.route(a.port("right"), b.port("left"), arrow=True) for a, b in zip(cards, cards[1:])]
    _title(ctx, title)
    ctx.play(LaggedStart(*(FadeIn(c) for c in cards), lag_ratio=0.2), run_time=Ctx.beat(ctx.remaining, 4))
    ctx.play(*(FadeIn(w) for w in wires), run_time=0.6)
    paths = [Line(a.port("right"), b.port("left")) for a, b in zip(cards, cards[1:])]  # plain lines for the pulse: an arrow is an outline
    ctx.play(PulseTrain(paths, delays=[0.5 * i for i in range(len(paths))], run_time=0.9), run_time=0.9 + 0.5 * (len(paths) - 1))
    ctx.finish()
