"""Circuit schematics, built on the community plugin manim-eng (https://github.com/overegneered/manim-eng).

manim-eng is MIT-licensed with one extra clause from its author: the repository's contents may not be used to train AI
models without his written approval. This project only renders diagrams with it; nothing from the library (source,
docs or examples) is used as training or fine-tuning data, and it must stay that way (see ``docs/THIRD_PARTY.md``).

Complements the hand-written ``circuit`` component (animated current in a loop of R, C, L, lamp and switch) with
standard schematic symbols: diodes, Zener diodes, variable resistors and thermistors, wires routed at right angles,
and a current arrow.
"""
from __future__ import annotations

import math
import re

import numpy as np

from manim import FadeIn, Create, Write

from ..ctx import Ctx
from ..registry import Param as P, component
from ._plugin import require

_PIP = "manim-eng"
_CX, _CY = 0.0, 0.45
_W, _H = 3.4, 1.7  # half-width and half-height of the wire loop

# type name -> manim-eng class name. "battery" is accepted for "cell".
_CLASSES = {
    "cell": "Cell", "resistor": "Resistor", "variable_resistor": "VariableResistor", "capacitor": "Capacitor",
    "inductor": "Inductor", "switch": "Switch", "closed_switch": "Switch", "diode": "Diode", "zener": "ZenerDiode",
    "thermistor": "Thermistor",
}
_ALIASES = {"battery": "cell", "source": "cell", "variable": "variable_resistor", "zener_diode": "zener"}

# Labels are typeset by LaTeX, so only a small safe alphabet is accepted (no backslashes, braces or other TeX
# commands). The Greek letters people type for units are converted.
_LABEL = re.compile(r"^[A-Za-z0-9 .,=+\-_^/()ΩµμΦ]{1,14}$")
_ELEMENT = re.compile(r"^\s*([A-Za-z_]+)\s*(?::\s*(.*?))?\s*$")


def parse_elements(elements: list[str]) -> list[tuple[str, str]]:
    """'cell:12 V' strings to (canonical type, label); raises ValueError on anything unsupported."""
    out: list[tuple[str, str]] = []
    for raw in elements:
        m = _ELEMENT.match(raw)
        if not m:
            raise ValueError(f"cannot read {raw!r}")
        kind = m.group(1).lower()
        kind = _ALIASES.get(kind, kind)
        if kind not in _CLASSES:
            raise ValueError(f"unknown element type {m.group(1)!r}")
        text = (m.group(2) or "").strip()
        if text and not _LABEL.match(text):
            raise ValueError(f"label {text!r} may only use letters, digits, spaces and . , = + - _ ^ / ( ) Ω µ")
        out.append((kind, text))
    return out


def _tex(label: str) -> str:
    return label.replace("Ω", r"\Omega ").replace("µ", r"\mu ").replace("μ", r"\mu ").replace("Φ", r"\Phi ")


def _elements_ok(v: list) -> str | None:
    if not 1 <= len(v) <= 6:
        return "needs 1 to 6 elements"
    try:
        parse_elements(v)
    except ValueError as e:
        return str(e)
    return None


def _current_ok(v: str) -> str | None:
    return None if v == "" or _LABEL.match(v) else "label may only use letters, digits and . , = + - _ ^ / ( )"


# Where each element goes around the loop, clockwise from the left side: (side, index along that side)
_ORDER = {
    1: ["top"], 2: ["left", "right"], 3: ["left", "top", "right"], 4: ["left", "top", "right", "bottom"],
    5: ["left", "top", "top", "right", "bottom"], 6: ["left", "top", "top", "right", "bottom", "bottom"],
}


def _slots(n: int) -> list[tuple[float, float, float]]:
    """(x, y, rotation) per element. Every element is wired from its right terminal to the next one's left terminal, so
    the rotation always makes 'right' the clockwise exit."""
    sides = _ORDER[n]
    out, seen = [], {"top": 0, "bottom": 0}
    for side in sides:
        if side == "left":
            out.append((_CX - _W, _CY, math.pi / 2))
        elif side == "right":
            out.append((_CX + _W, _CY, -math.pi / 2))
        else:
            count = sides.count(side)
            k = seen[side]
            seen[side] += 1
            xs = [0.0] if count == 1 else [-1.6, 1.6]
            x = xs[k] if side == "top" else xs[-1 - k]  # the bottom row runs right to left
            out.append((_CX + x, _CY + (_H if side == "top" else -_H), 0.0 if side == "top" else math.pi))
    return out


@component(
    "circuit_diagram",
    "Circuit schematic of 1-6 standard symbols. Types: cell, resistor, variable_resistor, capacitor, inductor, "
    "switch, closed_switch, diode, zener, thermistor.",
    {"elements": P("string[]", "'type:label' round the loop, e.g. ['cell:12 V', 'resistor:R_1']",
                   validator=_elements_ok),
     "current": P("string", "current arrow label, e.g. 'I'", default="",
                  validator=_current_ok)},
    {"elements": ["cell:12 V", "resistor:R_1", "switch:S", "diode:D"], "current": "I"},
    domain="electrical",
    plugin="manim-eng",
)
def circuit_diagram(ctx: Ctx, elements: list[str], current: str = "") -> None:
    eng = require("manim_eng", _PIP)
    specs = parse_elements(elements)
    parts = []
    for kind, _ in specs:
        part = getattr(eng, _CLASSES[kind])()
        if kind == "closed_switch":
            part.close()
        parts.append(part)
    for part, (x, y, rot) in zip(parts, _slots(len(parts))):
        part.rotate(rot)
        # place by the middle of the two terminals, not the bounding box: a switch blade or a thermistor slash would
        # otherwise pull the terminals off the wire line
        mid = (part.left.get_center() + part.right.get_center()) / 2
        part.shift(np.array([x, y, 0.0]) - mid)
    for part, (_, text) in zip(parts, specs):
        if not text:
            continue
        try:
            part.set_label(_tex(text))
        except Exception:  # noqa: BLE001 - e.g. a stray "_" that LaTeX rejects: retry without sub/superscripts
            part.set_label(_tex(re.sub(r"[_^]", "", text)))
    circuit = eng.Circuit(*parts)
    for a, b in zip(parts, parts[1:] + parts[:1]):
        circuit.connect(a.right, b.left)
    wires = list(circuit.wires)

    head = ctx.heading("Circuit diagram")
    ctx.play(Write(head), run_time=0.7)
    ctx.play(*[FadeIn(p) for p in parts], run_time=0.9)
    ctx.play(*[Create(w) for w in wires], run_time=min(2.0, max(1.0, ctx.remaining * 0.25)))
    switches = [p for p, (kind, _) in zip(parts, specs) if kind == "switch"]
    if switches:
        ctx.play(*[s.animate.close() for s in switches], run_time=0.6)
    if current.strip():
        pick = next((p for p, (k, _) in zip(parts, specs) if k in ("resistor", "variable_resistor", "thermistor")), None)
        pick = pick or next((p for p, (k, _) in zip(parts, specs) if k != "cell"), parts[0])
        try:
            ctx.play(pick.animate.set_current(_tex(current.strip())), run_time=0.8)
        except Exception:  # noqa: BLE001 - keep the diagram if the arrow label cannot be typeset
            pass
    ctx.wait(ctx.remaining)
