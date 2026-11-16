"""Computer-science pack: algorithms and core ideas, drawn with plain Manim (no community plugin needed).

Every algorithm is computed here in ordinary Python (``binary_search_steps``, ``fib_tree``, ``build_buckets``, ...), and
the scene only plays the real steps back, so what the video shows is always what the algorithm actually does.
"""
from __future__ import annotations

import math
import re

import numpy as np
from manim import (
    DOWN, LEFT, RIGHT, UP, Arrow, Axes, Create, DashedLine, Dot, FadeIn, FadeOut, FadeTransform, Line, RoundedRectangle,
    Rectangle, Text, Triangle, VGroup, Write,
)

from .. import theme
from ..ctx import Ctx, label
from ..registry import Param as P, component

DOMAIN = "computer-science"


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _box(text, w: float = 1.0, h: float = 0.7, color: str = theme.FG, fill: str = theme.PANEL, size: int = 26,
         fill_opacity: float = 1.0, stroke: float = 3) -> VGroup:
    """A rounded box with centred text. ``group[0]`` is the rectangle and ``group[1]`` the text."""
    rect = RoundedRectangle(corner_radius=0.1, width=w, height=h, color=color, fill_color=fill, fill_opacity=fill_opacity,
                            stroke_width=stroke)
    txt = Text(str(text), font_size=size, color=theme.FG)
    txt.move_to(rect)
    return VGroup(rect, txt)


def _marker(name: str, color: str) -> VGroup:
    """A small upward arrow-head with a name under it, used to point at array cells."""
    tri = Triangle(color=color, fill_color=color, fill_opacity=1, stroke_width=0).scale(0.11)
    return VGroup(tri, Text(name, font_size=18, color=color)).arrange(DOWN, buff=0.05)


# --------------------------------------------------------------------------------------------------------------
# Binary search
# --------------------------------------------------------------------------------------------------------------
def binary_search_steps(values: list[int], target: int) -> list[tuple[int, int, int, str]]:
    """Each probe as ``(lo, mid, hi, verdict)``. ``values`` must be sorted. verdict: 'found', 'right' (target is larger,
    so keep the right half) or 'left' (target is smaller, so keep the left half)."""
    lo, hi, steps = 0, len(values) - 1, []
    while lo <= hi:
        mid = (lo + hi) // 2
        if values[mid] == target:
            steps.append((lo, mid, hi, "found"))
            break
        if values[mid] < target:
            steps.append((lo, mid, hi, "right"))
            lo = mid + 1
        else:
            steps.append((lo, mid, hi, "left"))
            hi = mid - 1
    return steps


def _bs_values_ok(v: list) -> str | None:
    if not 3 <= len(v) <= 8:
        return "needs 3 to 8 values"
    if any(not float(x).is_integer() or not 0 <= x <= 99 for x in v):
        return "values must be whole numbers from 0 to 99"
    return None


def _0_99(v) -> str | None:
    return None if 0 <= v <= 99 else "must be from 0 to 99"


@component(
    "binary_search",
    "Binary search in a sorted array: look at the middle, discard the half that cannot hold the target, repeat. Shows why it takes about log2(n) steps.",
    {"values": P("number[]", "3-8 whole numbers 0-99 (sorted for you), e.g. [3, 8, 15, 23, 42, 57, 71]", required=True, validator=_bs_values_ok),
     "target": P("integer", "the number to look for, 0-99", required=True, validator=_0_99)},
    {"values": [3, 8, 15, 23, 42, 57, 71], "target": 57},
    domain=DOMAIN,
)
def binary_search(ctx: Ctx, values: list[float] | None = None, target: int = 0) -> None:
    vals = sorted(int(v) for v in (values or [3, 8, 15, 23, 42, 57, 71]))
    n = len(vals)
    steps = binary_search_steps(vals, target)
    gap = min(1.35, 11.0 / n)
    xs = [(i - (n - 1) / 2) * gap for i in range(n)]
    y_arr = 1.15
    boxes = [_box(v, w=gap * 0.86, h=0.95, size=30).move_to((xs[i], y_arr, 0)) for i, v in enumerate(vals)]
    idx = [Text(str(i), font_size=20, color=theme.MUTED).move_to((xs[i], y_arr - 0.75, 0)) for i in range(n)]
    head = ctx.heading(f"Binary search for {target}")
    rows = {"lo": -0.15, "mid": -0.72, "hi": -1.29}
    lo_m, mid_m, hi_m = _marker("lo", theme.ACCENT), _marker("mid", theme.YELLOW), _marker("hi", theme.PURPLE)
    lo_m.move_to((xs[0], rows["lo"], 0))
    hi_m.move_to((xs[-1], rows["hi"], 0))
    status = label(f"Sorted array of {n}. Is {target} in it?", 26, theme.FG).move_to((0, -2.05, 0))

    ctx.play(Write(head), run_time=0.6)
    ctx.play(*[FadeIn(b, shift=DOWN * 0.15) for b in boxes], FadeIn(VGroup(*idx)), run_time=0.8)
    ctx.play(FadeIn(lo_m), FadeIn(hi_m), FadeIn(status), run_time=0.5)
    per = _clamp((ctx.remaining - 1.2) / max(1, len(steps)), 1.9, 3.2)
    found = False
    for k, (lo, mid, hi, verdict) in enumerate(steps):
        mid_to = (xs[mid], rows["mid"], 0)
        if k == 0:
            mid_m.move_to(mid_to)
            ctx.play(FadeIn(mid_m), boxes[mid][0].animate.set_stroke(theme.YELLOW, 6), run_time=0.6)
        else:
            ctx.play(mid_m.animate.move_to(mid_to), boxes[mid][0].animate.set_stroke(theme.YELLOW, 6), run_time=0.6)
        word = {"found": f"{vals[mid]} = {target}: found it at index {mid}!",
                "right": f"{vals[mid]} < {target}: too small, keep the right half",
                "left": f"{vals[mid]} > {target}: too big, keep the left half"}[verdict]
        new_status = label(word, 26, theme.GOOD if verdict == "found" else theme.FG).move_to(status)
        ctx.play(FadeTransform(status, new_status), run_time=0.5)
        status = new_status
        ctx.wait(max(0.1, per - 2.0))
        if verdict == "found":
            found = True
            ctx.play(boxes[mid][0].animate.set_stroke(theme.GOOD, 7).set_fill(theme.GOOD, 0.35), run_time=0.6)
            break
        gone = list(range(lo, mid + 1)) if verdict == "right" else list(range(mid, hi + 1))
        moves = [boxes[i].animate.set_stroke(theme.FG, 3).set_opacity(0.22) for i in gone]
        if verdict == "right":
            moves.append(lo_m.animate.move_to((xs[min(mid + 1, n - 1)], rows["lo"], 0)))
        else:
            moves.append(hi_m.animate.move_to((xs[max(mid - 1, 0)], rows["hi"], 0)))
        ctx.play(*moves, run_time=0.7)
    if not found:
        miss = label(f"Nothing left to check: {target} is not in the array", 26, theme.DANGER).move_to(status)
        ctx.play(FadeTransform(status, miss), run_time=0.5)
    ctx.wait(ctx.remaining)


# --------------------------------------------------------------------------------------------------------------
# Big-O growth
# --------------------------------------------------------------------------------------------------------------
_BIGO = {  # key -> (name shown, f(n), colour); the dict order is slowest-growing first
    "1": ("O(1)", lambda n: 1.0, theme.GOOD),
    "log n": ("O(log n)", lambda n: math.log2(max(n, 1.0)), theme.ACCENT),
    "n": ("O(n)", lambda n: n, theme.YELLOW),
    "n log n": ("O(n log n)", lambda n: n * math.log2(max(n, 1.0)), theme.PURPLE),
    "n^2": ("O(n²)", lambda n: n * n, theme.WARN),
    "2^n": ("O(2ⁿ)", lambda n: 2.0 ** n, theme.DANGER),
}
_SUPER = str.maketrans({"²": "^2", "ⁿ": "^n", "·": "*"})


def norm_bigo(raw: str) -> str | None:
    """Map what a person (or a small model) might write to one of the keys of ``_BIGO``; None if it is not recognised."""
    s = str(raw).lower().translate(_SUPER).strip()
    s = re.sub(r"^o\((.*)\)$", r"\1", s).strip()
    s = s.replace("log(n)", "log n").replace("logn", "log n").replace("*", " ").replace("**", "^")
    s = re.sub(r"\s+", " ", s)
    s = {"constant": "1", "linear": "n", "logarithmic": "log n", "quadratic": "n^2", "exponential": "2^n",
         "n2": "n^2", "n squared": "n^2", "nlog n": "n log n", "n log(n)": "n log n", "nlogn": "n log n",
         "log2 n": "log n", "2n": "2^n", "n^n": "2^n"}.get(s, s)
    return s if s in _BIGO else None


def _bigo_ok(v: list) -> str | None:
    if not 2 <= len(v) <= 6:
        return "needs 2 to 6 growth rates"
    bad = [x for x in v if norm_bigo(x) is None]
    if bad:
        return f"unknown growth rate {bad[0]!r}; use any of {list(_BIGO)}"
    return None


@component(
    "big_o_growth",
    "Plots how the work grows with input size n for several Big-O classes (O(1), O(log n), O(n), O(n log n), O(n^2), O(2^n)), so the gap between them is obvious.",
    {"functions": P("string[]", "2-6 of: '1', 'log n', 'n', 'n log n', 'n^2', '2^n'", default=["1", "log n", "n", "n log n", "n^2"], validator=_bigo_ok)},
    {"functions": ["log n", "n", "n log n", "n^2", "2^n"]},
    domain=DOMAIN,
)
def big_o_growth(ctx: Ctx, functions: list[str] | None = None) -> None:
    keys = [k for k in _BIGO if k in {norm_bigo(f) for f in (functions or ["1", "log n", "n", "n log n", "n^2"])}]
    axes = Axes(x_range=[0, 10, 2], y_range=[0, 50, 10], x_length=9.2, y_length=3.5,
                axis_config={"color": theme.MUTED, "include_numbers": False, "include_tip": True, "stroke_width": 2,
                            "tip_width": 0.16, "tip_height": 0.16}).move_to((-0.9, 0.55, 0))
    xl = Text("input size n", font_size=22, color=theme.MUTED).next_to(axes, DOWN, buff=0.12).align_to(axes, RIGHT)
    yl = Text("work done", font_size=22, color=theme.MUTED).rotate(math.pi / 2).next_to(axes, LEFT, buff=0.15)
    head = ctx.heading("How fast does the work grow?")
    ctx.play(Write(head), Create(axes), FadeIn(xl), FadeIn(yl), run_time=1.0)

    curves, tags = [], []
    ends: list[tuple[float, VGroup, bool]] = []
    for key in keys:
        name, f, color = _BIGO[key]
        x_end = 10.0
        while f(x_end) > 50 and x_end > 0.5:
            x_end -= 0.02
        curve = axes.plot(f, x_range=[0, x_end, 0.05], color=color, stroke_width=5, use_smoothing=False)
        tag = Text(name, font_size=24, color=color, weight="BOLD")
        end = axes.c2p(x_end, f(x_end))
        clipped = x_end < 9.99
        tag.move_to(end + (UP * 0.3 if clipped else RIGHT * (0.35 + tag.width / 2)))
        curves.append(curve)
        tags.append(tag)
        ends.append((end[1], tag, clipped))
    side = sorted((e for e in ends if not e[2]), key=lambda e: e[0])  # nudge labels that sit on the right edge apart
    for i in range(1, len(side)):
        gap = side[i][1].get_center()[1] - side[i - 1][1].get_center()[1]
        if gap < 0.38:
            side[i][1].shift(UP * (0.38 - gap))
    per = _clamp((ctx.remaining - 3.2) / max(1, len(keys)), 0.5, 1.4)
    for curve, tag in zip(curves, tags):
        ctx.play(Create(curve), FadeIn(tag), run_time=per)
    order = "  <  ".join(_BIGO[k][0] for k in keys)
    note = label(f"Slowest growth to fastest:  {order}", 20, theme.YELLOW, width=100).move_to((0, -2.1, 0))
    ctx.play(FadeIn(note), run_time=0.6)
    ctx.wait(ctx.remaining)


# --------------------------------------------------------------------------------------------------------------
# Recursion tree (naive Fibonacci)
# --------------------------------------------------------------------------------------------------------------
def fib_tree(n: int) -> list[dict]:
    """Call tree of the naive ``fib(n)``. Nodes come back breadth-first as dicts with ``id``, ``parent``, ``value``,
    ``depth`` and ``x`` (leaves are numbered left to right, a parent sits over the middle of its children)."""
    nodes: list[dict] = []

    def build(v: int, depth: int, parent: int | None) -> int:
        nid = len(nodes)
        nodes.append({"id": nid, "parent": parent, "value": v, "depth": depth, "kids": []})
        if v >= 2:
            nodes[nid]["kids"] = [build(v - 1, depth + 1, nid), build(v - 2, depth + 1, nid)]
        return nid

    build(n, 0, None)
    leaf = [0]

    def place(nid: int) -> float:
        kids = nodes[nid]["kids"]
        if not kids:
            nodes[nid]["x"] = float(leaf[0])
            leaf[0] += 1
        else:
            xs = [place(k) for k in kids]
            nodes[nid]["x"] = sum(xs) / len(xs)
        return nodes[nid]["x"]

    place(0)
    return sorted(nodes, key=lambda d: (d["depth"], d["id"]))


def repeated_calls(nodes: list[dict]) -> dict[int, int]:
    """Values (2 and up) that the tree computes more than once, with how many times."""
    counts: dict[int, int] = {}
    for d in nodes:
        counts[d["value"]] = counts.get(d["value"], 0) + 1
    return {v: c for v, c in sorted(counts.items(), reverse=True) if v >= 2 and c > 1}


@component(
    "recursion_tree",
    "Call tree of the naive recursive Fibonacci fib(n): the tree is built level by level, then the repeated sub-calls are highlighted to show why plain recursion is wasteful (and why memoization helps).",
    {"n": P("integer", "which Fibonacci number to compute, 2-5", default=5, choices=[2, 3, 4, 5])},
    {"n": 5},
    domain=DOMAIN,
)
def recursion_tree(ctx: Ctx, n: int = 5) -> None:
    n = int(_clamp(n, 2, 5))
    nodes = fib_tree(n)
    leaves = sum(1 for d in nodes if not d["kids"])
    depth = max(d["depth"] for d in nodes)
    dx = min(1.35, 11.4 / max(leaves, 1))
    dy = min(0.95, 3.7 / max(depth, 1))
    x0 = -(leaves - 1) * dx / 2
    boxes: dict[int, VGroup] = {}
    for d in nodes:
        base = d["value"] < 2
        b = _box(f"fib({d['value']})", w=min(1.2, dx * 0.88), h=0.52, size=22, color=theme.GOOD if base else theme.ACCENT, stroke=2)
        b.move_to((x0 + d["x"] * dx, 2.45 - d["depth"] * dy, 0))
        boxes[d["id"]] = b
    edges = {d["id"]: Line(boxes[d["parent"]].get_bottom(), boxes[d["id"]].get_top(), color=theme.MUTED, stroke_width=2)
             for d in nodes if d["parent"] is not None}
    head = ctx.heading(f"Computing fib({n}) by recursion")
    ctx.play(Write(head), run_time=0.6)
    per = _clamp((ctx.remaining - 4.2) / (depth + 1), 0.4, 1.0)
    for level in range(depth + 1):
        row = [d for d in nodes if d["depth"] == level]
        ctx.play(*[FadeIn(edges[d["id"]]) for d in row if d["id"] in edges], *[FadeIn(boxes[d["id"]], scale=0.8) for d in row], run_time=per)
    dups = repeated_calls(nodes)
    palette = [theme.WARN, theme.PURPLE, theme.YELLOW, theme.DANGER]
    colours = {v: palette[i % len(palette)] for i, v in enumerate(dups)}
    marks = [boxes[d["id"]][0].animate.set_stroke(colours[d["value"]], 5).set_fill(colours[d["value"]], 0.3)
             for d in nodes if d["value"] in colours]
    if marks:
        ctx.play(*marks, run_time=0.9)
    summary = "   ".join(f"fib({v}) x{c}" for v, c in dups.items())
    lines = [label(f"{len(nodes)} calls, but only {n + 1} different values (fib(0) to fib({n}))", 22, theme.YELLOW)]
    if summary:
        lines.append(label(f"Same work, done again:  {summary}", 22, theme.WARN))
    note = VGroup(*lines).arrange(DOWN, buff=0.12).move_to((0, -2.0, 0))
    ctx.play(FadeIn(note), run_time=0.6)
    ctx.wait(ctx.remaining)


# --------------------------------------------------------------------------------------------------------------
# Hash table with chaining
# --------------------------------------------------------------------------------------------------------------
def hash_key(key: str, size: int) -> int:
    """The toy hash used in the video: the sum of the character codes, modulo the number of buckets."""
    return sum(ord(c) for c in key) % size


def build_buckets(keys: list[str], size: int) -> tuple[list[list[str]], list[bool]]:
    """Insert ``keys`` in order with chaining. Returns the final buckets and, per key, whether it collided."""
    buckets: list[list[str]] = [[] for _ in range(size)]
    collided: list[bool] = []
    for k in keys:
        b = hash_key(k, size)
        collided.append(bool(buckets[b]))
        buckets[b].append(k)
    return buckets, collided


def _keys_ok(v: list) -> str | None:
    if not 2 <= len(v) <= 6:
        return "needs 2 to 6 keys"
    if any(not s.strip() or len(s.strip()) > 10 or not s.strip().isascii() for s in v):
        return "each key must be 1-10 plain ASCII characters"
    return None


@component(
    "hash_table",
    "Hash table with chaining: each key is hashed to a bucket (sum of character codes mod table size) and collisions form a chain. Shows O(1) lookup and what a collision is.",
    {"keys": P("string[]", "2-6 short keys (1-10 ASCII characters), e.g. ['cat', 'dog', 'bee', 'ant']", required=True, validator=_keys_ok),
     "size": P("integer", "number of buckets, 4-8", default=5, choices=[4, 5, 6, 7, 8])},
    {"keys": ["cat", "dog", "act", "bee"], "size": 5},
    domain=DOMAIN,
)
def hash_table(ctx: Ctx, keys: list[str] | None = None, size: int = 5) -> None:
    ks = [s.strip() for s in (keys or ["cat", "dog", "act", "bee"])][:6]
    size = int(_clamp(size, 4, 8))
    pitch = min(0.78, 4.3 / size)
    top = 1.95
    bx = -4.2                                   # x of the bucket column
    head = ctx.heading("Hash table with chaining")
    heads = [_box(i, w=0.9, h=pitch * 0.84, size=26, color=theme.ACCENT, stroke=2).move_to((bx, top - i * pitch, 0)) for i in range(size)]
    ctx.play(Write(head), FadeIn(VGroup(*heads)), run_time=0.8)
    per = _clamp((ctx.remaining - 1.6) / len(ks), 1.3, 2.6)
    last: list = [None] * size                  # the newest node in each bucket's chain
    counts = [0] * size
    formula = None
    collisions = 0
    for key in ks:
        h = hash_key(key, size)
        codes = [ord(c) for c in key]
        shown = " + ".join(str(c) for c in codes) if len(key) <= 4 else str(sum(codes))
        text = f"hash('{key}') = ({shown}) mod {size} = {h}" if len(key) <= 4 else f"hash('{key}') = {shown} mod {size} = {h}"
        new_f = label(text, 30, theme.YELLOW).move_to((0, 2.75, 0))
        ctx.play(FadeIn(new_f) if formula is None else FadeTransform(formula, new_f), run_time=0.5)
        formula = new_f
        clash = counts[h] > 0
        node = _box(key, w=max(1.2, 0.22 * len(key) + 0.7), h=pitch * 0.84, size=26, color=theme.WARN if clash else theme.GOOD, stroke=3)
        node.move_to((bx + 2.0 + counts[h] * 2.3, top - h * pitch, 0))
        src = heads[h].get_right() if last[h] is None else last[h].get_right()
        link = Arrow(src, node.get_left(), buff=0.03, stroke_width=3, color=theme.MUTED, max_tip_length_to_length_ratio=0.3)
        ctx.play(heads[h][0].animate.set_stroke(theme.YELLOW, 5), FadeIn(node, shift=RIGHT * 0.3), FadeIn(link), run_time=0.6)
        last[h] = node
        if clash:
            collisions += 1
            warn = label("collision: same bucket, so it joins the chain", 24, theme.WARN).move_to((0, -2.1, 0))
            ctx.play(FadeIn(warn), run_time=0.3)
            ctx.wait(max(0.1, per - 1.7))
            ctx.play(FadeOut(warn), heads[h][0].animate.set_stroke(theme.ACCENT, 2), run_time=0.3)
        else:
            ctx.wait(max(0.1, per - 1.4))
            ctx.play(heads[h][0].animate.set_stroke(theme.ACCENT, 2), run_time=0.3)
        counts[h] += 1
    done = label(f"{len(ks)} keys, {collisions} collision{'s' if collisions != 1 else ''}. Lookup = hash, then scan one short chain.",
                 24, theme.FG, width=80).move_to((0, -2.1, 0))
    ctx.play(FadeIn(done), run_time=0.5)
    ctx.wait(ctx.remaining)


# --------------------------------------------------------------------------------------------------------------
# Linked list reversal
# --------------------------------------------------------------------------------------------------------------
def reverse_steps(n: int) -> list[tuple[int | None, int, int | None]]:
    """The ``(prev, curr, next)`` node indexes at the start of each pass of the classic iterative reversal. ``None`` is null."""
    return [(k - 1 if k > 0 else None, k, k + 1 if k + 1 < n else None) for k in range(n)]


def reverse_list(values: list) -> list:
    """Reverse by running the exact pointer steps above (used to check the animation ends where it should)."""
    nxt = {i: (i + 1 if i + 1 < len(values) else None) for i in range(len(values))}
    prev = None
    curr = 0 if values else None
    while curr is not None:
        after = nxt[curr]
        nxt[curr] = prev
        prev, curr = curr, after
    out, node = [], prev
    while node is not None:
        out.append(values[node])
        node = nxt[node]
    return out


def _ll_values_ok(v: list) -> str | None:
    if not 3 <= len(v) <= 6:
        return "needs 3 to 6 values"
    if any(not s.strip() or len(s.strip()) > 4 for s in v):
        return "each value must be 1-4 characters"
    return None


@component(
    "linked_list_reverse",
    "Reverses a singly linked list in place with the prev / curr / next pointers, flipping one arrow per step.",
    {"values": P("string[]", "3-6 short node values (1-4 characters), e.g. ['1', '2', '3', '4']", required=True, validator=_ll_values_ok)},
    {"values": ["1", "2", "3", "4"]},
    domain=DOMAIN,
)
def linked_list_reverse(ctx: Ctx, values: list[str] | None = None) -> None:
    vals = [s.strip() for s in (values or ["1", "2", "3", "4"])][:6]
    n = len(vals)
    gap = min(1.9, 10.2 / n)
    xs = [(i - (n - 1) / 2) * gap for i in range(n)]
    y = 1.0
    nodes = [_box(v, w=1.0, h=0.85, size=30, color=theme.ACCENT).move_to((xs[i], y, 0)) for i, v in enumerate(vals)]
    none_l = Text("None", font_size=22, color=theme.MUTED).move_to((xs[0] - gap * 0.95, y, 0))
    none_r = Text("None", font_size=22, color=theme.MUTED).move_to((xs[-1] + gap * 0.95, y, 0))

    def arrow(a, b, color):
        return Arrow(a, b, buff=0.06, stroke_width=4, color=color, max_tip_length_to_length_ratio=0.3)

    orig = [arrow(nodes[k].get_right(), (nodes[k + 1] if k + 1 < n else none_r).get_left(), theme.FG) for k in range(n)]
    rev = [arrow(nodes[0].get_left(), none_l.get_right(), theme.GOOD)] + [arrow(nodes[k].get_left(), nodes[k - 1].get_right(), theme.GOOD) for k in range(1, n)]
    head_tag = Text("head", font_size=22, color=theme.YELLOW, weight="BOLD").move_to((xs[0], y + 0.95, 0))
    rows = {"prev": -0.25, "curr": -0.83, "next": -1.41}

    def spot(i: int | None, side: str) -> float:          # x position for a pointer to node i (None: the null on that side)
        return xs[i] if i is not None else (none_l.get_center()[0] if side == "l" else none_r.get_center()[0])

    prev_m, curr_m, next_m = _marker("prev", theme.ACCENT), _marker("curr", theme.YELLOW), _marker("next", theme.PURPLE)
    prev_m.move_to((spot(None, "l"), rows["prev"], 0))
    curr_m.move_to((spot(0, "l"), rows["curr"], 0))
    next_m.move_to((spot(1 if n > 1 else None, "r"), rows["next"], 0))
    head = ctx.heading("Reversing a linked list")
    status = label("Each arrow must end up pointing the other way.", 26, theme.FG).move_to((0, -2.2, 0))

    ctx.play(Write(head), run_time=0.6)
    ctx.play(*[FadeIn(nd, shift=DOWN * 0.15) for nd in nodes], FadeIn(none_l), FadeIn(none_r), run_time=0.7)
    ctx.play(*[Create(a) for a in orig], FadeIn(head_tag), FadeIn(status), run_time=0.8)
    ctx.play(FadeIn(prev_m), FadeIn(curr_m), run_time=0.4)
    s = _clamp((ctx.remaining - 3.0) / max(1, n) / 2.0, 0.3, 1.0)
    for k, (p, c, nx) in enumerate(reverse_steps(n)):
        ctx.play(FadeIn(next_m) if k == 0 else next_m.animate.move_to((spot(nx, "r"), rows["next"], 0)), run_time=0.5 * s)
        back = vals[p] if p is not None else "None"
        new_status = label(f"{vals[c]} now points back to {back}", 26, theme.FG).move_to(status)
        ctx.play(FadeTransform(status, new_status), FadeOut(orig[c]), FadeIn(rev[c]), run_time=0.8 * s)
        status = new_status
        ctx.play(prev_m.animate.move_to((spot(c, "l"), rows["prev"], 0)), curr_m.animate.move_to((spot(nx, "r"), rows["curr"], 0)), run_time=0.7 * s)
    done = label(f"Reversed: {' -> '.join(reverse_list(vals))}", 26, theme.GOOD).move_to(status)
    ctx.play(FadeTransform(status, done), head_tag.animate.move_to((xs[-1], y + 0.95, 0)), run_time=0.8)
    ctx.wait(ctx.remaining)


# --------------------------------------------------------------------------------------------------------------
# Dynamic programming: longest common subsequence
# --------------------------------------------------------------------------------------------------------------
def lcs_table(a: str, b: str) -> list[list[int]]:
    """The DP table: ``L[i][j]`` is the LCS length of ``a[:i]`` and ``b[:j]``."""
    L = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            L[i][j] = L[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(L[i - 1][j], L[i][j - 1])
    return L


def lcs_backtrack(a: str, b: str, L: list[list[int]]) -> tuple[str, list[tuple[int, int]]]:
    """Walk back from the corner. Returns the subsequence and the cells visited (corner first)."""
    i, j, out, path = len(a), len(b), [], []
    while i > 0 and j > 0:
        path.append((i, j))
        if a[i - 1] == b[j - 1]:
            out.append(a[i - 1])
            i, j = i - 1, j - 1
        elif L[i - 1][j] >= L[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return "".join(reversed(out)), path


def _word_ok(v: str) -> str | None:
    if not 2 <= len(v.strip()) <= 6 or not v.strip().isalpha() or not v.strip().isascii():
        return "must be 2-6 plain letters"
    return None


@component(
    "dp_table",
    "Dynamic programming on the longest common subsequence of two short words: the table is filled cell by cell from smaller sub-problems, then traced back to read the answer.",
    {"a": P("string", "first word, 2-6 letters, e.g. 'ABCBD'", required=True, validator=_word_ok),
     "b": P("string", "second word, 2-6 letters, e.g. 'BDCAB'", required=True, validator=_word_ok)},
    {"a": "ABCBD", "b": "BDCAB"},
    domain=DOMAIN,
)
def dp_table(ctx: Ctx, a: str = "ABCBD", b: str = "BDCAB") -> None:
    a, b = a.strip().upper(), b.strip().upper()
    m, n = len(a), len(b)
    L = lcs_table(a, b)
    cw, ch = 0.82, 0.58
    cx, cy = -1.7, 0.45
    ox, oy = cx - n * cw / 2, cy + m * ch / 2                 # centre of cell (0, 0)
    pos = lambda i, j: (ox + j * cw, oy - i * ch, 0)
    rects = [[Rectangle(width=cw, height=ch, stroke_width=2, color=theme.MUTED, fill_color=theme.PANEL, fill_opacity=0.6).move_to(pos(i, j))
              for j in range(n + 1)] for i in range(m + 1)]
    nums = [[Text(str(L[i][j]), font_size=24, color=theme.FG).move_to(pos(i, j)) for j in range(n + 1)] for i in range(m + 1)]
    cols = [Text(ch_, font_size=26, color=theme.ACCENT, weight="BOLD").move_to((pos(0, j + 1)[0], oy + ch * 0.95, 0)) for j, ch_ in enumerate(b)]
    rows = [Text(ch_, font_size=26, color=theme.PURPLE, weight="BOLD").move_to((ox - cw * 0.95, pos(i + 1, 0)[1], 0)) for i, ch_ in enumerate(a)]
    head = ctx.heading(f"Longest common subsequence: {a} and {b}")
    legend = VGroup(
        label("green: letters match,\ntake diagonal + 1", 20, theme.GOOD, width=26, align="left"),
        label("otherwise: max(up, left)", 20, theme.FG, width=26, align="left"),
        label("purple outline: cells being read", 20, theme.PURPLE, width=30, align="left"),
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
    legend.move_to((4.3, 1.5, 0)).align_to((2.9, 0, 0), LEFT)

    border = [rects[i][j] for i in range(m + 1) for j in range(n + 1)]
    ctx.play(Write(head), run_time=0.6)
    ctx.play(*[FadeIn(r) for r in border], *[FadeIn(t) for t in cols + rows], FadeIn(legend),
             *[FadeIn(nums[i][j]) for i in range(m + 1) for j in range(n + 1) if i == 0 or j == 0], run_time=0.9)
    inner = [(i, j) for i in range(1, m + 1) for j in range(1, n + 1)]
    t = (ctx.remaining - 4.0) / max(1, len(inner))
    if t >= 0.22:
        prev: tuple | None = None                             # (cell, was a match, cells that were read) from the step before
        for i, j in inner:
            match = a[i - 1] == b[j - 1]
            src = [(i - 1, j - 1)] if match else [(i - 1, j), (i, j - 1)]
            anims = [rects[i][j].animate.set_fill(theme.YELLOW, 0.4), *[rects[x][y].animate.set_stroke(theme.PURPLE, 5) for x, y in src],
                     FadeIn(nums[i][j])]
            if prev:                                          # settle the previous cell in the same play, so there is one play per cell
                (pi, pj), pmatch, psrc = prev
                anims += [rects[pi][pj].animate.set_fill(theme.GOOD if pmatch else theme.PANEL, 0.4 if pmatch else 0.6),
                          *[rects[x][y].animate.set_stroke(theme.MUTED, 2) for x, y in psrc if (x, y) not in src]]
            ctx.play(*anims, run_time=t)
            prev = ((i, j), match, src)
        (pi, pj), pmatch, psrc = prev
        ctx.play(rects[pi][pj].animate.set_fill(theme.GOOD if pmatch else theme.PANEL, 0.4 if pmatch else 0.6),
                 *[rects[x][y].animate.set_stroke(theme.MUTED, 2) for x, y in psrc], run_time=0.3)
    else:                                                     # not enough time for every cell: reveal a row at a time
        per_row = _clamp((ctx.remaining - 4.0) / m, 0.3, 0.9)
        for i in range(1, m + 1):
            anims = []
            for j in range(1, n + 1):
                anims.append(FadeIn(nums[i][j]))
                if a[i - 1] == b[j - 1]:
                    anims.append(rects[i][j].animate.set_fill(theme.GOOD, 0.4))
            ctx.play(*anims, run_time=per_row)
    word, path = lcs_backtrack(a, b, L)
    ctx.play(*[rects[i][j].animate.set_stroke(theme.GOOD, 6) for i, j in path], run_time=_clamp(ctx.remaining - 2.2, 0.6, 1.4))
    if word:
        result = VGroup(Text(f"LCS = {word}", font_size=32, color=theme.GOOD, weight="BOLD"),
                        Text(f"length {L[m][n]}", font_size=24, color=theme.MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
    else:
        result = Text("no common subsequence", font_size=26, color=theme.DANGER)
    result.move_to((4.3, -0.9, 0)).align_to((2.9, 0, 0), LEFT)
    ctx.play(FadeIn(result), run_time=0.5)
    ctx.wait(ctx.remaining)


# --------------------------------------------------------------------------------------------------------------
# Sequence diagram (protocols, API calls, handshakes)
# --------------------------------------------------------------------------------------------------------------
_MSG = re.compile(r"^\s*(.+?)\s*(-->|->)\s*(.+?)\s*:\s*(.+?)\s*$")


def parse_messages(raw: list[str]) -> list[tuple[str, str, str, bool]]:
    """``'A->B: text'`` is a request and ``'B-->A: text'`` a reply (drawn dashed). Returns (src, dst, text, is_reply)."""
    out = []
    for text in raw:
        m = _MSG.match(text)
        if not m:
            raise ValueError(f"cannot read message {text!r}; write 'Client->Server: GET /' (use --> for a reply)")
        src, arrow, dst, body = m.groups()
        if src.strip().lower() == dst.strip().lower():
            raise ValueError(f"message {text!r} goes from {src!r} to itself")
        out.append((src.strip(), dst.strip(), body.strip(), arrow == "-->"))
    return out


def _messages_ok(v: list) -> str | None:
    try:
        parse_messages(v)
    except ValueError as e:
        return str(e)
    return None


def _actors_ok(v: list) -> str | None:
    if not 2 <= len(v) <= 5:
        return "needs 2 to 5 participants"
    if any(not s.strip() or len(s.strip()) > 14 for s in v):
        return "each participant name must be 1-14 characters"
    return None


@component(
    "sequence_diagram",
    "Sequence diagram: 2-5 participants with lifelines and up to 8 messages in order (requests solid, replies dashed). For TCP handshakes, HTTP requests, login flows, API calls.",
    {"actors": P("string[]", "2-5 participant names, left to right, e.g. ['Client', 'Server']", required=True, validator=_actors_ok),
     "messages": P("string[]", "up to 8 messages: 'Client->Server: SYN' for a request, 'Server-->Client: SYN-ACK' (two dashes) for a reply", required=True, validator=_messages_ok),
     "title": P("string", "Heading", default="")},
    {"actors": ["Client", "Server"], "messages": ["Client->Server: SYN", "Server-->Client: SYN-ACK", "Client->Server: ACK"], "title": "TCP three-way handshake"},
    domain=DOMAIN,
)
def sequence_diagram(ctx: Ctx, actors: list[str] | None = None, messages: list[str] | None = None, title: str = "") -> None:
    names = [s.strip() for s in (actors or ["Client", "Server"])][:5]
    msgs = []
    for src, dst, text, reply in parse_messages(messages or ["Client->Server: request", "Server-->Client: response"]):
        low = {x.lower(): x for x in names}
        for who in (src, dst):                               # a name the planner forgot to list still gets a lifeline
            if who.lower() not in low and len(names) < 5:
                names.append(who)
        low = {x.lower(): i for i, x in enumerate(names)}
        if src.lower() in low and dst.lower() in low:
            msgs.append((low[src.lower()], low[dst.lower()], text, reply))
    msgs = msgs[:8] or [(0, 1, "request", False)]
    k = len(names)
    span = min(10.0, 3.6 * (k - 1))
    xs = [(i - (k - 1) / 2) * (span / max(k - 1, 1)) for i in range(k)]
    top = 2.65 if title else 2.85
    heads = [_box(nm, w=min(2.2, 9.0 / k), h=0.55, size=22, color=theme.ACCENT).move_to((xs[i], top, 0)) for i, nm in enumerate(names)]
    bottom = -2.15
    lifelines = [DashedLine((xs[i], top - 0.3, 0), (xs[i], bottom, 0), color=theme.MUTED, stroke_width=2, dash_length=0.12) for i in range(k)]
    step = _clamp((top - 0.3 - 0.45 - bottom) / len(msgs), 0.45, 0.85)
    y0 = top - 0.3 - step * 0.75
    if title:
        ctx.play(Write(ctx.heading(title)), run_time=0.6)
    ctx.play(*[FadeIn(h) for h in heads], *[Create(l) for l in lifelines], run_time=0.9)
    per = _clamp((ctx.remaining - 1.2) / len(msgs), 0.5, 1.4)
    for n_, (s, d, text, reply) in enumerate(msgs):
        y = y0 - n_ * step
        a, b = (xs[s], y, 0), (xs[d], y, 0)
        color = theme.GOOD if reply else theme.ACCENT
        if reply:
            line = DashedLine(a, b, color=color, stroke_width=4, dash_length=0.14)
            line.add_tip(tip_length=0.2)
        else:
            line = Arrow(a, b, buff=0, color=color, stroke_width=4, max_tip_length_to_length_ratio=0.15)
        txt = label(text, 20, theme.FG, width=30).move_to(((xs[s] + xs[d]) / 2, y + 0.24, 0))
        if txt.width > abs(xs[d] - xs[s]) - 0.2:
            txt.scale_to_fit_width(max(0.8, abs(xs[d] - xs[s]) - 0.2))
        ctx.play(Create(line), FadeIn(txt), run_time=min(per, 0.9))
        ctx.wait(max(0.0, per - 0.9))
    ctx.wait(ctx.remaining)


# --------------------------------------------------------------------------------------------------------------
# Binary representation
# --------------------------------------------------------------------------------------------------------------
PLACES = [128, 64, 32, 16, 8, 4, 2, 1]


def to_bits(value: int) -> list[tuple[int, int, int]]:
    """For each place value, high to low: ``(place, bit, remainder_after)``, by greedy subtraction."""
    out, rem = [], int(value)
    for p in PLACES:
        bit = 1 if rem >= p else 0
        rem -= p * bit
        out.append((p, bit, rem))
    return out


@component(
    "binary_representation",
    "Shows how a number from 0 to 255 is written in binary: eight bits with place values 128 to 1, deciding each bit by subtraction, then adding the lit places back up.",
    {"value": P("integer", "a whole number from 0 to 255", required=True, validator=lambda v: None if 0 <= v <= 255 else "must be from 0 to 255")},
    {"value": 170},
    domain=DOMAIN,
)
def binary_representation(ctx: Ctx, value: int = 170) -> None:
    value = int(_clamp(value, 0, 255))
    bits = to_bits(value)
    gap, y_p, y_b = 1.3, 1.75, 0.85
    xs = [(i - 3.5) * gap for i in range(8)]
    places = [Text(str(p), font_size=26, color=theme.MUTED).move_to((xs[i], y_p, 0)) for i, p in enumerate(PLACES)]
    rects = [RoundedRectangle(corner_radius=0.1, width=1.05, height=0.95, color=theme.MUTED, fill_color=theme.PANEL, fill_opacity=1,
                              stroke_width=3).move_to((xs[i], y_b, 0)) for i in range(8)]
    digits = [Text(str(b), font_size=40, color=theme.FG, weight="BOLD").move_to((xs[i], y_b, 0)) for i, (_, b, _) in enumerate(bits)]
    head = ctx.heading(f"{value} in binary")
    status = label(f"Start with {value}. For each place value, from the biggest: does it fit?", 26, theme.FG, width=70).move_to((0, -0.35, 0))
    ctx.play(Write(head), run_time=0.6)
    ctx.play(*[FadeIn(r) for r in rects], *[FadeIn(p) for p in places], FadeIn(status), run_time=0.8)
    per = _clamp((ctx.remaining - 3.0) / 8, 0.3, 0.9)
    left = value
    for i, (p, bit, after) in enumerate(bits):
        verdict = f"{left} >= {p}: yes, bit = 1, {after} left" if bit else f"{left} < {p}: no, bit = 0"
        new_status = label(verdict, 26, theme.GOOD if bit else theme.MUTED).move_to(status)
        anims = [FadeTransform(status, new_status), FadeIn(digits[i])]
        if bit:
            anims.append(rects[i].animate.set_stroke(theme.GOOD, 5).set_fill(theme.GOOD, 0.3))
        ctx.play(*anims, run_time=per)
        status = new_status
        left = after
    lit = [str(p) for p, b, _ in bits if b]
    total = label(f"{' + '.join(lit) if lit else '0'} = {value}", 30, theme.YELLOW).move_to((0, -1.35, 0))
    word = label("".join(str(b) for _, b, _ in bits), 44, theme.GOOD, weight="BOLD").move_to((0, -2.0, 0))
    ctx.play(FadeOut(status), FadeIn(total), run_time=0.6)
    ctx.play(FadeIn(word), run_time=0.6)
    ctx.wait(ctx.remaining)
