"""Components drawn with the community library manim-extensions (https://github.com/ExploreMaths/manim_extensions, MIT).

Each one is an ``ExtSpec`` (see ``_ext.py``): what to build, and which beats to play. Economics (filed under the
``finance`` domain), concept maps and trees, a file tree, a gear pair and a queue.
"""
from __future__ import annotations

import re
from typing import Any

from manim import RIGHT, Indicate, Rotate, Text, VGroup, linear

from .. import theme
from ..ctx import label
from ..registry import Param as P
from ._ext import Beat, Built, ExtSpec, choose, clamp, lib, plain_text_ok, register, split_labelled

# --------------------------------------------------------------------------------------------------------------
# Economics
# --------------------------------------------------------------------------------------------------------------


def _two_stage(call):
    """Many economics methods return ``[short_run, long_run]``. Return two functions, so each gets its own beat/note."""
    rest: list = []

    def first():
        anims = list(call())
        rest.extend(anims[1:])
        return anims[0]

    return first, lambda: list(rest)


# ---- supply and demand -----------------------------------------------------------------------------------
SD_EVENTS = {"demand_increase": (1, 0), "demand_decrease": (-1, 0), "supply_increase": (0, 1), "supply_decrease": (0, -1)}
SD_TEXT = {
    "demand_increase": "Buyers want more at every price: demand shifts right",
    "demand_decrease": "Buyers want less at every price: demand shifts left",
    "supply_increase": "Sellers offer more at every price: supply shifts right",
    "supply_decrease": "Sellers offer less at every price: supply shifts left",
}


def sd_shifts(event: str, size: float) -> tuple[float, float]:
    """Sideways shift (in quantity units) of the demand and supply curves."""
    d, s = SD_EVENTS[choose(event, SD_EVENTS, "demand_increase")]
    size = clamp(size, 1, 3, 2)
    return d * size, s * size


def sd_equilibrium(demand_shift: float = 0.0, supply_shift: float = 0.0) -> tuple[float, float]:
    """(quantity, price) where D: P = 8 - 0.5(Q - ds) meets S: P = 2 + 0.5(Q - ss). Without shifts that is (6, 5)."""
    return 6 + 0.5 * (demand_shift + supply_shift), 5 + 0.25 * (demand_shift - supply_shift)


def _g(x: float) -> str:
    return f"{x:g}"


def _build_supply_demand(p: dict[str, Any]) -> Built:
    ed = lib("economics")
    event = choose(p.get("event"), SD_EVENTS, "demand_increase")
    ds, ss = sd_shifts(event, p.get("size", 2))
    d = ed.SupplyDemandDiagram()
    (q0, p0), (q1, p1) = sd_equilibrium(), sd_equilibrium(ds, ss)
    curve = "demand" if ds else "supply"
    shift = ds or ss
    if curve == "demand":
        func = lambda q: 8 - 0.5 * (q - shift)  # noqa: E731
    else:
        func = lambda q: 2 + 0.5 * (q - shift)  # noqa: E731
    note = f"{SD_TEXT[event]}. Price {_g(p0)} to {_g(round(p1, 2))}, quantity {_g(q0)} to {_g(round(q1, 2))}"
    return Built(d, [Beat(lambda: d.get_shift_animation(curve, func, show_arrows=True), note)],
                 note="Where buyers and sellers agree: the equilibrium price and quantity")


register(ExtSpec(
    "supply_demand",
    "Supply and demand curves; one curve shifts and the equilibrium price and quantity move.",
    "finance",
    {"event": P("string", "what happens", choices=list(SD_EVENTS), default="demand_increase"),
     "size": P("number", "how far the curve shifts, 1 to 3", default=2)},
    {"event": "demand_increase", "size": 2},
    lambda p: "Supply and demand",
    _build_supply_demand,
))

# ---- AD-AS -------------------------------------------------------------------------------------------------
ADAS_EVENTS = {
    "positive_demand_shock": ("positive_demand_shock", {"m": 30}, "Short run: demand rises, output goes above potential",
                              "Long run: prices catch up and output returns to potential at a higher price level"),
    "negative_demand_shock": ("negative_demand_shock", {"m": 12}, "Short run: demand falls, output drops below potential",
                              "Long run: prices fall and output returns to potential at a lower price level"),
    "adverse_supply_shock": ("adverse_supply_shock", {"sras_price": 6}, "Short run: costs rise, so prices rise and output falls",
                             "Long run: the economy self-corrects and the price level comes back down"),
    "positive_supply_shock": ("positive_supply_shock", {"sras_price": 2}, "Short run: costs fall, so prices fall and output rises",
                              "Long run: the economy self-corrects and the price level comes back up"),
}


def _build_ad_as(p: dict[str, Any]) -> Built:
    ed = lib("economics")
    event = choose(p.get("event"), ADAS_EVENTS, "positive_demand_shock")
    method, kw, short, long_ = ADAS_EVENTS[event]
    long_run = bool(p.get("long_run", True))
    d = ed.ADASDiagram()
    call = lambda: getattr(d, method)(long_run=long_run, show_arrows=True, **kw)  # noqa: E731
    if not long_run:
        return Built(d, [Beat(lambda: call()[0], short)], note="Aggregate demand (AD), short-run supply (SRAS), long-run supply (LRAS)")
    first, rest = _two_stage(call)
    return Built(d, [Beat(first, short), Beat(rest, long_, weight=1.3)],
                 note="Aggregate demand (AD), short-run supply (SRAS), long-run supply (LRAS)")


register(ExtSpec(
    "ad_as_model",
    "Aggregate demand and supply: a demand or supply shock, then the long-run return to potential output.",
    "finance",
    {"event": P("string", "which shock", choices=list(ADAS_EVENTS), default="positive_demand_shock"),
     "long_run": P("boolean", "also show the long-run adjustment", default=True)},
    {"event": "positive_demand_shock", "long_run": True},
    lambda p: "Aggregate demand and aggregate supply",
    _build_ad_as,
))

# ---- IS-LM -------------------------------------------------------------------------------------------------
ISLM_POLICIES = {
    "monetary_expansion": ({"ms": 3}, "Central bank adds money: LM shifts right, interest rate falls, output rises"),
    "monetary_contraction": ({"ms": 1}, "Central bank removes money: LM shifts left, interest rate rises, output falls"),
    "fiscal_expansion": ({"a": 11.0}, "Government spends more: IS shifts right, interest rate and output both rise"),
    "fiscal_contraction": ({"a": 8.0}, "Government spends less: IS shifts left, interest rate and output both fall"),
}


def _build_is_lm(p: dict[str, Any]) -> Built:
    ed = lib("economics")
    policy = choose(p.get("policy"), ISLM_POLICIES, "monetary_expansion")
    kw, note = ISLM_POLICIES[policy]
    d = ed.ISLMDiagram()
    return Built(d, [Beat(lambda: getattr(d, policy)(show_arrows=True, **kw), note)],
                 note="IS: goods market is in balance. LM: money market is in balance")


register(ExtSpec(
    "is_lm_model",
    "IS-LM diagram: a monetary or fiscal policy change shifts one curve and moves interest rate and output.",
    "finance",
    {"policy": P("string", "which policy", choices=list(ISLM_POLICIES), default="monetary_expansion")},
    {"policy": "monetary_expansion"},
    lambda p: "The IS-LM model",
    _build_is_lm,
))

# ---- IS-LM next to AD-AS ---------------------------------------------------------------------------------
LINKED = {
    "monetary_expansion": ({"ms": 3}, "More money shifts LM and AD together: output rises in both views"),
    "monetary_contraction": ({"ms": 1}, "Less money shifts LM and AD together: output falls in both views"),
    "fiscal_expansion": ({"a": 11.0}, "More government spending shifts IS and AD: output rises in both views"),
    "fiscal_contraction": ({"a": 8.0}, "Less government spending shifts IS and AD: output falls in both views"),
    "adverse_supply_shock": ({"sras_price": 6}, "Higher costs shift supply up: prices rise and output falls"),
    "positive_supply_shock": ({"sras_price": 2}, "Lower costs shift supply down: prices fall and output rises"),
}


def _build_linked(p: dict[str, Any]) -> Built:
    ed = lib("economics")
    policy = choose(p.get("policy"), LINKED, "monetary_expansion")
    kw, note = LINKED[policy]
    d = ed.LinkedISLM_ADAS()
    return Built(d, [Beat(lambda: getattr(d, policy)(**kw), note)], note="IS-LM (left) and AD-AS (right) describe the same economy")


register(ExtSpec(
    "is_lm_ad_as",
    "IS-LM and AD-AS side by side, linked: one policy or supply shock moves both diagrams.",
    "finance",
    {"policy": P("string", "which change", choices=list(LINKED), default="monetary_expansion")},
    {"policy": "monetary_expansion"},
    lambda p: "IS-LM and AD-AS together",
    _build_linked,
))

# ---- Solow growth ------------------------------------------------------------------------------------------
SOLOW = {
    "increase_savings": ({"s": 0.4}, "People save more: the savings curve rises and the steady state moves up"),
    "decrease_savings": ({"s": 0.2}, "People save less: the savings curve falls and the steady state moves down"),
    "population_growth": ({"n": 0.1}, "Population grows faster: the break-even line steepens and the steady state falls"),
    "increase_depreciation": ({"delta": 0.15}, "Capital wears out faster: the break-even line steepens and the steady state falls"),
}


def _build_solow(p: dict[str, Any]) -> Built:
    ed = lib("economics")
    change = choose(p.get("change"), SOLOW, "increase_savings")
    kw, note = SOLOW[change]
    d = ed.SolowDiagram()
    return Built(d, [Beat(lambda: getattr(d, change)(show_arrows=True, **kw), note)],
                 note="Capital per worker settles where saving equals break-even investment")


register(ExtSpec(
    "solow_growth",
    "Solow growth model: change savings, population growth or depreciation and watch the steady state move.",
    "finance",
    {"change": P("string", "what changes", choices=list(SOLOW), default="increase_savings")},
    {"change": "increase_savings"},
    lambda p: "The Solow growth model",
    _build_solow,
))

# --------------------------------------------------------------------------------------------------------------
# Concept maps, trees and timelines (the library's mind-map layouts)
# --------------------------------------------------------------------------------------------------------------


def _branches_ok(v: list) -> str | None:
    if not 2 <= len(v) <= 6:
        return "needs 2 to 6 branches"
    for head, subs in split_labelled(v):
        if not plain_text_ok(head, 24) or len(subs) > 4 or any(not plain_text_ok(s, 24) for s in subs):
            return "each branch is 'Label: item, item' with at most 4 short items"
    return None


def _node(text: str, size: int = 26) -> dict[str, Any]:
    return {"node": Text(text, font_size=size, color=theme.FG)}


def map_data(center: str, branches: list[str]) -> dict[str, Any]:
    """The nested dict the mind-map classes take."""
    kids = []
    for head, subs in split_labelled(branches):
        entry = _node(head)
        if subs:
            entry["child"] = [_node(s, 22) for s in subs]
        kids.append(entry)
    root = _node(center, 30)
    root["child"] = kids
    return root


def _map_beats(m: Any, branches: list[str]) -> list[Beat]:
    names = [h for h, _ in split_labelled(branches)]
    return [Beat(lambda i=i: Indicate(m.get_node((0, i)), color=theme.YELLOW), f"{n}") for i, n in enumerate(names[:4])]


def _map_spec(name: str, cls: str, description: str, title: str, center_desc: str) -> None:
    def build(p: dict[str, Any]) -> Built:
        m = getattr(lib("mindmap"), cls)(map_data(p["center"], list(p["branches"])))
        return Built(m, _map_beats(m, list(p["branches"])), max_scale=1.6)

    register(ExtSpec(
        name, description, "general",
        {"center": P("string", center_desc, required=True, validator=lambda v: None if plain_text_ok(v) else "must be 1 to 28 plain characters"),
         "branches": P("string[]", "2-6 items written 'Branch: sub, sub' (up to 4 subs each)", required=True, validator=_branches_ok)},
        {"center": "Renewable energy", "branches": ["Solar: panels, farms", "Wind: onshore, offshore", "Hydro: dams", "Storage: batteries"]},
        lambda p, t=title: p.get("title") or t,
        build,
    ))


_map_spec("concept_map", "StandardMap", "Concept map: a central idea with branches and sub-ideas spreading both ways.",
          "Concept map", "the central idea")
_map_spec("hierarchy_tree", "CatalogMap", "Hierarchy tree (org chart or taxonomy): a root with parents and children below it.",
          "Hierarchy", "the root of the tree")


def _events_ok(v: list) -> str | None:
    if not 2 <= len(v) <= 8:
        return "needs 2 to 8 events"
    for e in v:
        when, sep, what = str(e).partition(":")
        if not sep or not plain_text_ok(when, 10) or not plain_text_ok(what, 30):
            return "each event is 'year: short description'"
    return None


def _build_timeline(p: dict[str, Any]) -> Built:
    events = [(w.strip(), t.strip()) for w, _, t in (str(e).partition(":") for e in p["events"])]
    kids = [{"node": label(f"{w}\n{t}", 22, theme.FG, width=16)} for w, t in events]
    m = lib("mindmap").TimeLine({"node": Text(p.get("title") or "Timeline", font_size=30, color=theme.ACCENT), "child": kids})
    return Built(m, [Beat(lambda i=i: Indicate(m.get_node((0, i)), color=theme.YELLOW), events[i][0]) for i in range(min(len(events), 5))],
                 max_scale=1.6)


register(ExtSpec(
    "timeline",
    "Timeline: events in order, each a year and a short description, highlighted one by one.",
    "general",
    {"events": P("string[]", "2-8 items written 'year: description'", required=True, validator=_events_ok),
     "title": P("string", "title in the middle of the line", default="Timeline")},
    {"events": ["1879: Light bulb", "1888: AC motor", "1954: Solar cell", "1991: Lithium-ion"], "title": "Electrical milestones"},
    lambda p: p.get("title") or "Timeline",
    _build_timeline,
))

# --------------------------------------------------------------------------------------------------------------
# File tree
# --------------------------------------------------------------------------------------------------------------
_PATH = re.compile(r"^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+){0,3}$")


def _paths_ok(v: list) -> str | None:
    if not 2 <= len(v) <= 8:
        return "needs 2 to 8 paths"
    if any(not _PATH.match(x) or len(x) > 40 or any(not seg.strip(".") for seg in x.split("/")) for x in v):
        return "paths look like 'src/main.py' (letters, digits . _ -, up to 4 levels)"
    return None


def build_tree(paths: list[str]) -> dict[str, Any]:
    tree: dict[str, Any] = {}
    for path in paths:
        node, parts = tree, path.split("/")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if node is None:
                break
        else:
            node.setdefault(parts[-1], None)
    return tree


def tree_lines(tree: dict[str, Any], prefix: str = "") -> list[str]:
    """Paths in the order the tree is drawn (parents before children)."""
    out: list[str] = []
    for name, child in tree.items():
        out.append(prefix + name)
        if isinstance(child, dict):
            out += tree_lines(child, prefix + name + "/")
    return out


def _build_file_tree(p: dict[str, Any]) -> Built:
    tree = build_tree(list(p["paths"]))
    order = tree_lines(tree)
    ft = lib().FileTree(tree, font_size=30)
    picks = [order.index(h) for h in p.get("highlight", []) if h in order][:4]
    names = [order[i] for i in picks]
    return Built(ft, [Beat(lambda i=i: ft.highlight(i, theme.YELLOW), n) for i, n in zip(picks, names)], max_scale=1.5)


register(ExtSpec(
    "file_tree",
    "Folder and file tree of a project, with chosen files highlighted in turn.",
    "computer-science",
    {"paths": P("string[]", "2-8 file paths such as 'src/main.py'", required=True, validator=_paths_ok),
     "highlight": P("string[]", "up to 4 of those paths (folders or files) to point out")},
    {"paths": ["src/main.py", "src/utils/helpers.py", "README.md", "tests/test_main.py"], "highlight": ["src/main.py", "tests"]},
    lambda p: "Project structure",
    _build_file_tree,
))

# --------------------------------------------------------------------------------------------------------------
# Gears
# --------------------------------------------------------------------------------------------------------------


def gear_ratio(driver: int, driven: int) -> float:
    return driven / driver


def _build_gears(p: dict[str, Any]) -> Built:
    gb = lib("gearbox")
    n1, n2 = int(clamp(p.get("driver_teeth", 15), 8, 40, 15)), int(clamp(p.get("driven_teeth", 25), 8, 60, 25))
    g1 = gb.Gear(n1, stroke_opacity=0, fill_color=theme.ACCENT, fill_opacity=1)
    g2 = gb.Gear(n2, stroke_opacity=0, fill_color=theme.WARN, fill_opacity=1)
    g1.shift(-g1.rp * 1.5 * RIGHT)
    g2.mesh_to(g1)
    teeth = max(3, n1 // 2)  # whole teeth, so the pair stays meshed
    ratio = gear_ratio(n1, n2)
    word = "slower, with more torque" if ratio > 1 else ("faster, with less torque" if ratio < 1 else "at the same speed")
    note = f"{n1} teeth drive {n2} teeth: ratio {ratio:.2f} to 1, so the output turns {word}"
    spin = lambda: [Rotate(g1, g1.pitch_angle * teeth, rate_func=linear), Rotate(g2, -g2.pitch_angle * teeth, rate_func=linear)]  # noqa: E731
    return Built(VGroup(g1, g2), [Beat(spin, note)], note="Blue gear drives the orange gear", max_scale=2.0)


register(ExtSpec(
    "gear_train",
    "Two meshing gears; the tooth counts set the speed and torque ratio.",
    "physics",
    {"driver_teeth": P("integer", "teeth on the driving gear, 8 to 40", default=15),
     "driven_teeth": P("integer", "teeth on the driven gear, 8 to 60", default=25)},
    {"driver_teeth": 15, "driven_teeth": 25},
    lambda p: "Gears: speed and torque",
    _build_gears,
))

# --------------------------------------------------------------------------------------------------------------
# Queue
# --------------------------------------------------------------------------------------------------------------


def parse_queue_ops(ops: list[str], start: int, capacity: int) -> list[tuple[str, str | None]]:
    out: list[tuple[str, str | None]] = []
    size = start
    for raw in ops:
        words = str(raw).split()
        if not words or words[0].lower() not in ("enqueue", "dequeue"):
            raise ValueError(f"unknown operation {raw!r}")
        kind = words[0].lower()
        if kind == "enqueue":
            if len(words) != 2 or not plain_text_ok(words[1], 4):
                raise ValueError("enqueue needs one short value")
            if size >= capacity:
                raise ValueError("the queue is full")
            size += 1
            out.append((kind, words[1]))
        else:
            if size == 0:
                raise ValueError("cannot dequeue from an empty queue")
            size -= 1
            out.append((kind, None))
    return out


def _queue_ops_ok(v: list) -> str | None:
    try:
        parse_queue_ops(v, 3, 5)
    except ValueError as e:
        return f"{e}; use 'enqueue 4' or 'dequeue', and keep the queue within capacity (starts with 3 items, holds 5)"
    return None


def _build_queue(p: dict[str, Any]) -> Built:
    alg = lib("algorithm")
    from manim_extensions.algorithm.node import Node  # noqa: PLC0415  (submodule of the already-required library)

    ops = parse_queue_ops(list(p["operations"]), 3, 5)
    q = alg.Queue(capacity=5, init_data=[1, 2, 3], total_width=8)

    def make(op: str, val: str | None):
        def play():
            if op == "enqueue":
                return alg.Queue.Enqueue(q, Node(val, width=q.item_width).next_to(q, RIGHT, buff=1.5))  # same cell size as the queue, or it overlaps its neighbours
            return alg.Queue.Dequeue(q)
        return Beat(play, f"enqueue {val}: joins the back" if op == "enqueue" else "dequeue: leaves from the front")

    return Built(q, [make(o, v) for o, v in ops], note="A queue: first in, first out. Capacity 5, starting with 1, 2, 3")


register(ExtSpec(
    "queue_operations",
    "A queue of capacity 5 starting with 1, 2, 3: enqueue at the back, dequeue from the front (first in, first out).",
    "computer-science",
    {"operations": P("string[]", "up to 8 steps: 'enqueue 4' or 'dequeue' (never more than 5 items, never empty)", required=True,
                     validator=_queue_ops_ok)},
    {"operations": ["enqueue 4", "dequeue", "dequeue", "enqueue 5"]},
    lambda p: "Queue: first in, first out",
    _build_queue,
))
