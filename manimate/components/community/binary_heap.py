"""Animated min/max binary heap: insertion, sift-up, extraction, and sift-down."""
from __future__ import annotations

import math

from manim import DOWN, Circle, FadeIn, FadeOut, Indicate, Line, Rectangle, Text, VGroup

from ... import theme
from ...ctx import Ctx
from ...registry import Param as P, component


def _values_ok(values: list[float]) -> str | None:
    if any(not math.isfinite(float(v)) for v in values):
        return "values must all be finite numbers"
    if any(abs(float(v)) > 9999 for v in values):
        return "values must be between -9999 and 9999"
    return None


def _count_ok(count: int) -> str | None:
    return "must be non-negative" if count < 0 else None


def _comes_first(a: float, b: float, kind: str) -> bool:
    return a < b if kind == "min" else a > b


def _is_heap(values: list[float], kind: str) -> bool:
    return all(not _comes_first(values[i], values[(i - 1) // 2], kind) for i in range(1, len(values)))


def _build_events(values: list[float], extract_count: int, kind: str) -> tuple[list[dict], list[float], list[float]]:
    """Return an event trace, extracted roots, and final heap (without Manim)."""
    heap: list[float] = []
    events: list[dict] = []
    for value in values:
        before = heap.copy()
        heap.append(value)
        events.append({"phase": "insert", "before": before, "after": heap.copy(),
                       "compared": [], "swapped": None, "message": f"Insert {value:g}"})
        i = len(heap) - 1
        while i > 0:
            parent = (i - 1) // 2
            before = heap.copy()
            swap = _comes_first(heap[i], heap[parent], kind)
            if swap:
                heap[parent], heap[i] = heap[i], heap[parent]
            events.append({"phase": "sift-up", "before": before, "after": heap.copy(),
                           "compared": [parent, i], "swapped": [parent, i] if swap else None,
                           "message": (f"Swap {before[i]:g} with {before[parent]:g}" if swap
                                       else f"Heap property holds at {before[parent]:g}")})
            if not swap:
                break
            i = parent

    extracted: list[float] = []
    for _ in range(min(extract_count, len(heap))):
        before = heap.copy()
        removed = heap[0]
        extracted.append(removed)
        if len(heap) == 1:
            heap.pop()
            events.append({"phase": "extract", "before": before, "after": [], "compared": [0],
                           "swapped": None, "message": f"Extract {removed:g}"})
            continue
        last = heap.pop()
        heap[0] = last
        events.append({"phase": "extract", "before": before, "after": heap.copy(),
                       "compared": [0, len(heap)], "swapped": None,
                       "message": f"Extract {removed:g}; move {last:g} to root"})
        i = 0
        while True:
            left, right = 2 * i + 1, 2 * i + 2
            if left >= len(heap):
                break
            best = left
            compared = [i, left]
            if right < len(heap):
                compared.append(right)
                if _comes_first(heap[right], heap[left], kind):
                    best = right
            before = heap.copy()
            swap = _comes_first(heap[best], heap[i], kind)
            if swap:
                heap[i], heap[best] = heap[best], heap[i]
            events.append({"phase": "sift-down", "before": before, "after": heap.copy(),
                           "compared": compared, "swapped": [i, best] if swap else None,
                           "message": (f"Swap {before[i]:g} with {before[best]:g}" if swap
                                       else f"Heap property holds at {before[i]:g}")})
            if not swap:
                break
            i = best
    return events, extracted, heap


def _tree_xy(index: int) -> tuple[float, float]:
    depth = (index + 1).bit_length() - 1
    first = (1 << depth) - 1
    slot = index - first
    slots = 1 << depth
    width = min(8.8, 2.25 * slots)
    return ((slot + 0.5) / slots - 0.5) * width, 1.35 - depth * 0.95


def _draw_board(values: list[float], kind: str, compared: list[int] | None = None,
                message: str = "") -> VGroup:
    compared = compared or []
    parts = []
    positions = [_tree_xy(i) for i in range(len(values))]
    for i in range(1, len(values)):
        p = (i - 1) // 2
        parts.append(Line([positions[p][0], positions[p][1], 0],
                          [positions[i][0], positions[i][1], 0], color=theme.MUTED, stroke_width=2))
    for i, value in enumerate(values):
        circle = Circle(radius=0.31, color=theme.YELLOW if i in compared else theme.ACCENT,
                        stroke_width=4 if i in compared else 2.5,
                        fill_color=theme.PANEL, fill_opacity=1).move_to([*positions[i], 0])
        number = Text(f"{value:g}", font_size=19, color=theme.FG).move_to(circle)
        parts.extend([circle, number])
    if values:
        y, cell_w = -1.65, min(0.92, 9.2 / len(values))
        start_x = -(len(values) * cell_w) / 2
        for i, value in enumerate(values):
            x = start_x + (i + 0.5) * cell_w
            rect = Rectangle(width=cell_w - 0.06, height=0.52,
                             color=theme.YELLOW if i in compared else theme.ACCENT,
                             stroke_width=3 if i in compared else 1.5,
                             fill_color=theme.PANEL, fill_opacity=1).move_to([x, y, 0])
            number = Text(f"{value:g}", font_size=18, color=theme.FG).move_to(rect)
            idx = Text(str(i), font_size=12, color=theme.MUTED).next_to(rect, DOWN, buff=0.03)
            parts.extend([rect, number, idx])
    else:
        parts.append(Text("Empty heap", font_size=25, color=theme.MUTED).move_to([0, -0.2, 0]))
    parts.append(Text(f"{kind.upper()}-HEAP ARRAY", font_size=16, color=theme.MUTED).move_to([0, -2.35, 0]))
    if message:
        parts.append(Text(message, font_size=18, color=theme.GOOD).move_to([0, 2.2, 0]))
    return VGroup(*parts)


@component(
    "binary_heap",
    "Visualize a min-heap or max-heap built by insertion, then remove the root using sift-down. Shows the heap tree and matching array with comparisons highlighted.",
    {
        "values": P("number[]", "Numbers to insert in order (up to 8); duplicates and negatives are supported",
                     default=[10, 20, 15, 5, 8], validator=_values_ok),
        "extract_count": P("integer", "Number of roots to extract after insertion; values above heap size are clamped",
                           default=2, validator=_count_ok),
        "kind": P("string", "Heap ordering", default="min", choices=["min", "max"]),
    },
    {"values": [10, 20, 15, 5, 8], "extract_count": 2, "kind": "min"},
    domain="computer-science",
)
def binary_heap(ctx: Ctx, values: list[float] | None = None, extract_count: int = 2, kind: str = "min") -> None:
    # An explicit [] means an empty heap; do not replace it with the default.
    items = list([10, 20, 15, 5, 8] if values is None else values[:8])
    events, extracted, final_heap = _build_events(items, max(0, int(extract_count)), kind)
    heading = ctx.heading(f"{kind.title()} Binary Heap")
    board = _draw_board([], kind, message="Insert values in the given order")
    ctx.fit(board, max_h=4.7)
    ctx.add(heading)
    ctx.play(FadeIn(board), run_time=0.8)
    if events:
        beat = max(0.1, min(0.25, (ctx.remaining - 0.4) / (len(events) + 1)))
        for event in events:
            target = _draw_board(event["after"], kind, event["compared"], event["message"])
            ctx.fit(target, max_h=4.7)
            ctx.play(FadeOut(board), FadeIn(target), run_time=beat)
            board = target
    else:
        ctx.play(Indicate(board, color=theme.ACCENT), run_time=min(0.5, max(0.1, ctx.remaining)))
    extracted_text = ", ".join(f"{v:g}" for v in extracted) if extracted else "none"
    footer = Text(f"Extracted: {extracted_text}   |   Final heap: {final_heap}",
                  font_size=15, color=theme.FG).move_to([0, -3.0, 0])
    ctx.play(FadeIn(footer), run_time=min(0.35, max(0.1, ctx.remaining)))
