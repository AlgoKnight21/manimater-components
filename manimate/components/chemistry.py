"""Chemistry pack, built on the community plugin manim-chemistry (https://github.com/UnMolDeQuimica/manim-chemistry).

Behaviour of manim-chemistry 0.5.1 that shaped this pack (checked, not assumed):

* ``BohrAtom`` fills shells 2, 8, then everything left in the third, so it is only right up to argon (element 18).
* ``ChemicalFormula`` merges repeated elements and drops parentheses (``Ca(OH)2`` becomes "CaO"), so it is used only
  for formulas where that cannot happen; the rest fall back to plain markup.
* Molecules with explicit hydrogen atoms crash inside the plugin, and single-atom molecules fail, so structures are
  drawn as skeletal formulas (implicit hydrogens), which is also how a chemistry class draws them.
* Nothing here calls the plugin's PubChem download helpers: renders never touch the network.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from manim import DOWN, LEFT, RIGHT, Arrow, Create, FadeIn, MarkupText, Mobject, VGroup, Write

from .. import theme
from ..ctx import Ctx, label
from ..registry import Param as P, component
from ._plugin import fit, require

_PIP = "manim-chemistry"
_CY = 0.55


# --------------------------------------------------------------------------------------------------------------
# Elements
# --------------------------------------------------------------------------------------------------------------
# (symbol, name, atomic mass) for hydrogen to argon
ELEMENTS = [
    ("H", "Hydrogen", 1.008), ("He", "Helium", 4.003), ("Li", "Lithium", 6.94), ("Be", "Beryllium", 9.012),
    ("B", "Boron", 10.81), ("C", "Carbon", 12.011), ("N", "Nitrogen", 14.007), ("O", "Oxygen", 15.999),
    ("F", "Fluorine", 18.998), ("Ne", "Neon", 20.180), ("Na", "Sodium", 22.990), ("Mg", "Magnesium", 24.305),
    ("Al", "Aluminium", 26.982), ("Si", "Silicon", 28.085), ("P", "Phosphorus", 30.974), ("S", "Sulfur", 32.06),
    ("Cl", "Chlorine", 35.45), ("Ar", "Argon", 39.948),
]
BY_SYMBOL = {sym: (z, name, mass) for z, (sym, name, mass) in enumerate(ELEMENTS, start=1)}

# Every element symbol, for validating equations (not only the 18 that can be drawn as atoms).
ALL_SYMBOLS = set(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb "
    "Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au "
    "Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts "
    "Og".split()
)


def shell_counts(electrons: int) -> list[int]:
    """Electrons per shell for elements 1-18 (2, 8, then the rest)."""
    shells = []
    for cap in (2, 8, 8):
        take = min(cap, electrons)
        if take:
            shells.append(take)
        electrons -= take
    return shells


@component(
    "bohr_atom",
    "Bohr model of an atom (H to Ar): nucleus and orbiting electron shells, counts labelled.",
    {"element": P("string", "symbol, H to Ar", default="Na",
                  validator=lambda v: None if v in BY_SYMBOL else "must be an element symbol from H to Ar")},
    {"element": "Na"},
    domain="chemistry",
    plugin="manim-chemistry",
)
def bohr_atom(ctx: Ctx, element: str = "Na") -> None:
    mc = require("manim_chemistry", _PIP)
    z, name, mass = BY_SYMBOL.get(element, BY_SYMBOL["Na"])
    neutrons = round(mass) - z
    atom = mc.BohrAtom(e=z, p=z, n=neutrons)
    fit(atom, max_w=5.0, max_h=4.2, center=(-3.4, _CY - 0.1, 0), max_scale=1.6)
    shells = list(atom.get_electrons())
    info = VGroup(
        label(name, 44, theme.FG),
        label(f"{element}   Z = {z}", 28, theme.ACCENT),
        label(f"protons {z}   neutrons {neutrons}   electrons {z}", 24, theme.MUTED),
        label("shells: " + ", ".join(str(c) for c in shell_counts(z)), 26, theme.FG),
        label("red = protons, white = neutrons, blue = electrons", 20, theme.MUTED),
    ).arrange(DOWN, buff=0.3, aligned_edge=LEFT).move_to((3.2, _CY, 0))
    head = ctx.heading("Bohr model")
    ctx.play(Write(head), Create(atom), run_time=min(2.0, max(1.0, ctx.remaining * 0.25)))
    ctx.play(*[FadeIn(row) for row in info], run_time=0.8)
    centre = atom.get_center()
    for k, shell in enumerate(shells):  # outer shells turn slower, alternating direction
        speed = (1.0 if k % 2 == 0 else -1.0) * 0.9 / (k + 1)
        shell.add_updater(lambda m, dt, s=speed: m.rotate(dt * s, about_point=centre))
    ctx.wait(ctx.remaining)
    for shell in shells:
        shell.clear_updaters()


# --------------------------------------------------------------------------------------------------------------
# Formulas
# --------------------------------------------------------------------------------------------------------------
_SIMPLE = re.compile(r"^(?:[A-Z][a-z]?\d*)+$")


def formula_counts(formula: str) -> Counter:
    """Atom counts of a formula such as ``Ca(OH)2`` or ``C6H12O6``. Raises ValueError if it is not a formula."""
    stack: list[Counter] = [Counter()]
    i = 0
    while i < len(formula):
        ch = formula[i]
        if ch == "(":
            stack.append(Counter())
            i += 1
        elif ch == ")":
            if len(stack) < 2:
                raise ValueError(f"unbalanced parentheses in {formula!r}")
            group = stack.pop()
            i += 1
            m = re.match(r"\d+", formula[i:])
            mult = int(m.group()) if m else 1
            i += len(m.group()) if m else 0
            for sym, n in group.items():
                stack[-1][sym] += n * mult
        else:
            m = re.match(r"([A-Z][a-z]?)(\d*)", formula[i:])
            if not m or m.group(1) not in ALL_SYMBOLS:
                raise ValueError(f"unknown element in {formula!r}")
            stack[-1][m.group(1)] += int(m.group(2) or 1)
            i += len(m.group(0))
    if len(stack) != 1:
        raise ValueError(f"unbalanced parentheses in {formula!r}")
    if not stack[0]:
        raise ValueError("empty formula")
    return stack[0]


def _plugin_safe(formula: str) -> bool:
    """True when manim-chemistry's ChemicalFormula renders this formula faithfully (no parentheses, no repeats)."""
    if not _SIMPLE.match(formula):
        return False
    symbols = re.findall(r"[A-Z][a-z]?", formula)
    return len(symbols) == len(set(symbols))


def formula_mobject(formula: str, font_size: int = 40) -> Mobject:
    if _plugin_safe(formula):
        try:
            mc = require("manim_chemistry", _PIP)
            return mc.ChemicalFormula(formula).scale(font_size / 48)
        except Exception:  # noqa: BLE001 - any plugin trouble falls back to plain markup below
            pass
    return MarkupText(re.sub(r"(\d+)", r"<sub>\1</sub>", formula), font_size=font_size, color=theme.FG)


# --------------------------------------------------------------------------------------------------------------
# Skeletal molecules
# --------------------------------------------------------------------------------------------------------------
def _mol_lines(atoms: list[tuple[str, float, float]], bonds: list[tuple[int, int, int]], name: str) -> list[str]:
    """The V2000 mol-block lines that the plugin's ``from_mol_string`` reads (it expects a list of lines)."""
    out = [name, "  manimate", "", f"{len(atoms):3d}{len(bonds):3d}  0  0  0  0  0  0  0  0999 V2000"]
    for sym, x, y in atoms:
        out.append(f"{x:10.4f}{y:10.4f}{0.0:10.4f} {sym:<3} 0  0  0  0  0  0  0  0  0  0  0  0")
    out += [f"{a:3d}{b:3d}{order:3d}  0  0  0  0" for a, b, order in bonds]
    out.append("M  END")
    return out


def _ring(n: int, r: float = 1.2) -> list[tuple[str, float, float]]:
    return [("C", r * math.cos(math.pi / 2 + math.tau * k / n), r * math.sin(math.pi / 2 + math.tau * k / n))
            for k in range(n)]


# name -> (atoms, bonds, molecular formula)
MOLECULES: dict[str, tuple[list, list, str]] = {
    "methanol": ([("C", -0.6, 0), ("O", 0.6, 0)], [(1, 2, 1)], "CH4O"),
    "ethanol": ([("C", -1.2, 0), ("C", 0, .7), ("O", 1.2, 0)], [(1, 2, 1), (2, 3, 1)], "C2H6O"),
    "acetic acid": ([("C", -1.2, 0), ("C", 0, .7), ("O", 0, 2.0), ("O", 1.2, 0)], [(1, 2, 1), (2, 3, 2), (2, 4, 1)], "C2H4O2"),
    "acetone": ([("C", -1.2, 0), ("C", 0, .7), ("O", 0, 2.0), ("C", 1.2, 0)], [(1, 2, 1), (2, 3, 2), (2, 4, 1)], "C3H6O"),
    "ethene": ([("C", -.7, 0), ("C", .7, 0)], [(1, 2, 2)], "C2H4"),
    "ethyne": ([("C", -.8, 0), ("C", .8, 0)], [(1, 2, 3)], "C2H2"),
    "benzene": (_ring(6), [(1, 2, 2), (2, 3, 1), (3, 4, 2), (4, 5, 1), (5, 6, 2), (6, 1, 1)], "C6H6"),
    "cyclohexane": (_ring(6), [(i, i % 6 + 1, 1) for i in range(1, 7)], "C6H12"),
    "glycine": ([("N", -1.8, 0), ("C", -.6, .7), ("C", .6, 0), ("O", .6, 1.4), ("O", 1.8, .7)],
                [(1, 2, 1), (2, 3, 1), (3, 4, 2), (3, 5, 1)], "C2H5NO2"),
}


@component(
    "molecule_structure",
    "Skeletal structure of a small organic molecule.",
    {"molecule": P("string", "molecule", default="ethanol", choices=list(MOLECULES))},
    {"molecule": "ethanol"},
    domain="chemistry",
    plugin="manim-chemistry",
)
def molecule_structure(ctx: Ctx, molecule: str = "ethanol") -> None:
    mc = require("manim_chemistry", _PIP)
    atoms, bonds, formula = MOLECULES.get(molecule, MOLECULES["ethanol"])
    mol = mc.MMoleculeObject.from_mol_string(_mol_lines(atoms, bonds, molecule))
    fit(mol, max_w=8.0, max_h=3.3, center=(0, _CY + 0.3, 0), max_scale=1.8)
    name = label(molecule.capitalize(), 40, theme.FG)
    form = formula_mobject(formula, 36)
    caption = label("each corner or line end is a carbon atom; hydrogens are implied", 20, theme.MUTED)
    foot = VGroup(name, form).arrange(RIGHT, buff=0.5).move_to((0, -1.9, 0))
    caption.next_to(foot, DOWN, buff=0.2)
    head = ctx.heading("Molecular structure")
    ctx.play(Write(head), Create(mol), run_time=min(2.5, max(1.2, ctx.remaining * 0.3)))
    ctx.play(FadeIn(foot), FadeIn(caption), run_time=0.8)
    ctx.wait(ctx.remaining)


# --------------------------------------------------------------------------------------------------------------
# Chemical equations
# --------------------------------------------------------------------------------------------------------------
Term = tuple[int, str, Counter]


def parse_equation(text: str) -> tuple[list[Term], list[Term]]:
    """'2H2 + O2 -> 2H2O' into (reactants, products) as (coefficient, formula, atom counts)."""
    sides = re.split(r"\s*(?:->|=>|→|=)\s*", text.strip())
    if len(sides) != 2:
        raise ValueError("needs one arrow, like 'H2 + O2 -> H2O'")
    parsed: list[list[Term]] = []
    for side in sides:
        terms: list[Term] = []
        for raw in side.split("+"):
            m = re.match(r"^\s*(\d{1,2})?\s*([A-Z(][A-Za-z0-9()]*)\s*$", raw)
            if not m:
                raise ValueError(f"cannot read {raw.strip()!r}")
            formula = m.group(2)
            terms.append((int(m.group(1) or 1), formula, formula_counts(formula)))
        if not 1 <= len(terms) <= 4:
            raise ValueError("each side needs 1 to 4 substances")
        parsed.append(terms)
    return parsed[0], parsed[1]


def side_totals(terms: list[Term]) -> Counter:
    total: Counter = Counter()
    for coef, _, counts in terms:
        for sym, n in counts.items():
            total[sym] += coef * n
    return total


def _equation_ok(v: str) -> str | None:
    try:
        parse_equation(v)
    except ValueError as e:
        return str(e)
    return None


@component(
    "chemical_equation",
    "A chemical equation with atom counts per element, showing whether it is balanced.",
    {"equation": P("string", "e.g. '2H2 + O2 -> 2H2O'", validator=_equation_ok)},
    {"equation": "2H2 + O2 -> 2H2O"},
    domain="chemistry",
    plugin="manim-chemistry",
)
def chemical_equation(ctx: Ctx, equation: str = "2H2 + O2 -> 2H2O") -> None:
    left, right = parse_equation(equation)
    pieces: list[Mobject] = []

    def add_side(terms: list[Term]) -> None:
        for k, (coef, formula, _) in enumerate(terms):
            if k:
                pieces.append(label("+", 40, theme.MUTED))
            term = VGroup(*([label(str(coef), 40, theme.YELLOW)] if coef > 1 else []), formula_mobject(formula))
            pieces.append(term.arrange(RIGHT, buff=0.08))

    add_side(left)
    pieces.append(Arrow(LEFT * 0.4, RIGHT * 0.4, buff=0, color=theme.FG, stroke_width=5, max_tip_length_to_length_ratio=0.5))
    add_side(right)
    row = VGroup(*pieces).arrange(RIGHT, buff=0.3)
    fit(row, max_w=11.5, max_h=1.2, center=(0, 1.7, 0), max_scale=1.0)

    totals_l, totals_r = side_totals(left), side_totals(right)
    symbols = list(dict.fromkeys([*totals_l, *totals_r]))
    balanced = all(totals_l[s] == totals_r[s] for s in symbols)
    rows = VGroup(*[
        label(f"{s}:   {totals_l[s]} on the left,  {totals_r[s]} on the right", 26,
              theme.GOOD if totals_l[s] == totals_r[s] else theme.DANGER)
        for s in symbols
    ]).arrange(DOWN, buff=0.18)
    verdict = label("Balanced: the same atoms on both sides" if balanced else "Not balanced: atom counts differ", 30,
                    theme.GOOD if balanced else theme.DANGER)
    below = VGroup(rows, verdict).arrange(DOWN, buff=0.35).move_to((0, -0.7, 0))
    head = ctx.heading("Balancing a chemical equation")
    ctx.play(Write(head), FadeIn(row), run_time=1.2)
    step = max(0.3, min(0.8, (ctx.remaining - 1.5) / max(1, len(rows))))
    for r in rows:
        ctx.play(FadeIn(r, shift=DOWN * 0.1), run_time=step)
    ctx.play(FadeIn(verdict), run_time=0.6)
    ctx.wait(ctx.remaining)
