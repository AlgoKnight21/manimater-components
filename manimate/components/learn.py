"""Study components: worked numerical problems, line-by-line code walkthroughs and algorithm dry runs."""
from __future__ import annotations

import re
from typing import Any

from manim import (
    DOWN, LEFT, RIGHT, UP, Create, Dot, FadeIn, FadeOut, Line, MathTex, Rectangle, RoundedRectangle, Text, VGroup, Write,
)

from .. import theme
from ..ctx import HAS_LATEX, STAGE_W, Ctx, label
from ..tex import normalize_tex, tex_to_plain
from ..registry import Param as P, component

DOMAIN = "learn"
MONO = "Monospace"


def _text(item: Any, key: str, limit: int) -> str:
    """A cleaned string field from an object[] item (the LLM may leave it out or send a number)."""
    v = item.get(key, "") if isinstance(item, dict) else ""
    return str(v).strip()[:limit] if v is not None else ""


def _math(latex: str, size: int, color: str = theme.FG, max_w: float = 7.0):
    """A formula as MathTex, or plain text when LaTeX is missing or the formula does not compile."""
    latex = normalize_tex(latex or "")
    m = None
    if HAS_LATEX and latex:
        try:
            m = MathTex(latex, font_size=size, color=color)
        except Exception:  # noqa: BLE001 - a bad formula must not cost the whole scene
            m = None
    if m is None:
        m = Text(tex_to_plain(latex) or " ", font_size=int(size * 0.7), color=color)
    if m.width > max_w:
        m.scale_to_fit_width(max_w)
    return m


# ------------------------------------------------------------------------------------------------ problem_statement
@component(
    "problem_statement",
    "A numerical problem laid out as what is Given and what to Find. Use right after the title in a worked-problem video.",
    {"given": P("string[]", "2-6 given values, short, e.g. 'Supply frequency f = 50 Hz'", required=True),
     "find": P("string", "What must be found, e.g. 'Synchronous speed N_s'", required=True),
     "title": P("string", "Heading", default="Problem")},
    {"given": ["Supply frequency f = 50 Hz", "Number of poles P = 4"], "find": "Synchronous speed N_s", "title": "Problem"},
    domain=DOMAIN,
)
def problem_statement(ctx: Ctx, given: list[str], find: str, title: str = "Problem") -> None:
    head = ctx.heading(title)
    given_head = label("Given", 32, theme.ACCENT, weight="BOLD")
    rows = VGroup(*[
        VGroup(Dot(color=theme.ACCENT, radius=0.07), label(g, 30, theme.FG, width=58, align="left")).arrange(RIGHT, buff=0.3)
        for g in given[:6]
    ]).arrange(DOWN, aligned_edge=LEFT, buff=0.32)
    given_block = VGroup(given_head, rows).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
    find_block = VGroup(label("Find", 32, theme.WARN, weight="BOLD"), label(find, 32, theme.FG, width=56, align="left")
                        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
    body = VGroup(given_block, find_block).arrange(DOWN, aligned_edge=LEFT, buff=0.7)
    ctx.fit(body, max_w=11.5, max_h=4.4, center=(0, 0.1, 0))
    ctx.play(Write(head), run_time=0.8)
    ctx.play(FadeIn(given_head, shift=RIGHT * 0.2), run_time=0.4)
    step = ctx.beat(ctx.remaining - 1.5, len(rows), cap=1.2)
    for r in rows:
        ctx.play(FadeIn(r, shift=RIGHT * 0.3), run_time=min(0.6, step))
        ctx.wait(max(0.0, step - 0.6))
    ctx.play(FadeIn(find_block, shift=UP * 0.2), run_time=0.8)


# ---------------------------------------------------------------------------------------------------- solution_steps
@component(
    "solution_steps",
    "A worked solution: 2-5 steps (6 at most), each with a name, one formula line and a short note, then a boxed final answer. "
    "Use after problem_statement. Put one calculation per step (formula, substitution, result).",
    {"title": P("string", "Heading", default="Solution"),
     "steps": P("object[]", "2-5 steps in order", required=True,
                item_hint="{label: step name under 25 chars, math: ONE LaTeX line without $, note: under 60 chars of plain words}"),
     "answer": P("string", "Final answer with its unit, e.g. 'N_s = 1500 rpm'. Leave empty to skip the answer box.", default="")},
    {"title": "Solution",
     "steps": [{"label": "Write the formula", "math": "N_s = \\frac{120 f}{P}", "note": "Speed set by frequency and poles"},
               {"label": "Substitute", "math": "N_s = \\frac{120 \\times 50}{4}", "note": "f = 50 Hz, P = 4"},
               {"label": "Calculate", "math": "N_s = 1500", "note": "Units are rpm"}],
     "answer": "N_s = 1500 rpm"},
    domain=DOMAIN,
)
def solution_steps(ctx: Ctx, steps: list[dict], title: str = "Solution", answer: str = "") -> None:
    steps = steps[:6]
    head = ctx.heading(title)
    lefts, rights = [], []
    for i, item in enumerate(steps):
        name, note = _text(item, "label", 40), _text(item, "note", 90)
        parts = [label(f"{i + 1}. {name}" if name else f"{i + 1}.", 26, theme.ACCENT, width=24, align="left", weight="BOLD")]
        if note:
            parts.append(label(note, 21, theme.MUTED, width=34, align="left"))
        lefts.append(VGroup(*parts).arrange(DOWN, aligned_edge=LEFT, buff=0.08))
        rights.append(_math(_text(item, "math", 160), 40))
    left_col = VGroup(*lefts).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
    col_x = left_col.get_right()[0] + 0.7
    for lf, rt in zip(lefts, rights):
        rt.move_to([col_x + rt.width / 2, lf.get_center()[1], 0])
    table = VGroup(left_col, *rights)
    parts = [table]
    box = None
    if answer.strip():
        txt = _math(answer.strip(), 44, theme.GOOD, max_w=9.5) if re.search(r"[\\^_=]", answer) else label(answer.strip(), 36, theme.GOOD, width=45, weight="BOLD")
        frame = RoundedRectangle(corner_radius=0.15, width=txt.width + 0.9, height=txt.height + 0.5, color=theme.GOOD, stroke_width=3)
        box = VGroup(frame, txt)
        txt.move_to(frame)
        parts.append(box)
    body = VGroup(*parts).arrange(DOWN, buff=0.4)
    ctx.fit(body, max_w=12.2, max_h=4.7 if box is not None else 5.0, center=(0, 0.0, 0))
    ctx.play(Write(head), run_time=0.7)
    step = ctx.beat(ctx.remaining - 1.0, len(steps) + (1 if box is not None else 0), cap=3.0)
    shown: list = []
    for lf, rt in zip(lefts, rights):
        dim = [m.animate.set_opacity(0.55) for m in shown]
        ctx.play(FadeIn(lf, shift=RIGHT * 0.25), Write(rt), *dim, run_time=min(1.2, max(0.5, step * 0.7)))
        ctx.wait(max(0.0, step - 1.2))
        shown += [lf, rt]
    if box is not None:
        ctx.play(Create(box[0]), Write(box[1]), run_time=min(1.2, max(0.6, step)))


# ------------------------------------------------------------------------------------------------ shared text helper
# Hidden glyphs added to every cell of text so all of them (code lines, line numbers, table cells) have the same vertical
# extent, which keeps their baselines level. Only the visible glyphs count for width and alignment.
_STRUT = "|([_gT"


def _padded(body: str, size: int, colour: str, *, font: str | None = None, weight: str = "NORMAL", t2c: dict[str, str] | None = None):
    """Text whose bounding box has a fixed height. ``.vis`` is the group of visible glyphs (use it for width and x-centring);
    ``.shown`` is the same glyphs, to be used when changing opacity (touching the hidden padding would reveal it)."""
    body = body if body.strip() else " "
    kw: dict[str, Any] = {"font_size": size, "color": colour, "weight": weight}
    if font:
        kw["font"] = font
    try:
        t = Text(f"{body} {_STRUT}", t2c=t2c or {}, **kw)
        visible = sum(1 for ch in body if not ch.isspace())
        if len(t) == visible + len(_STRUT):  # one glyph per character, so the padding is safe to hide
            VGroup(*t[visible:]).set_opacity(0)
            t.shown = VGroup(*t[:visible])
            t.vis = t.shown if visible else t
            return t
    except Exception:  # noqa: BLE001 - fall back to plain text below
        pass
    t = Text(body, **kw)
    t.vis = t.shown = t
    return t


def _put(m, x: float, y: float, anchor: str = "centre") -> None:
    """Move ``m`` so its visible glyphs are centred (or left-aligned) at x, and its box is centred at y."""
    vis = m.vis
    dx = x - (vis.get_left()[0] if anchor == "left" else vis.get_center()[0])
    m.shift([dx, y - m.get_center()[1], 0])


# -------------------------------------------------------------------------------------------------- code_walkthrough
_KEYWORDS = {
    "def", "return", "if", "elif", "else", "for", "while", "in", "not", "and", "or", "import", "from", "class", "break",
    "continue", "pass", "None", "True", "False", "lambda", "with", "as", "try", "except", "raise", "yield", "is",
    "function", "const", "let", "var", "new", "this", "null", "true", "false", "int", "float", "double", "void", "char",
    "bool", "auto", "struct", "switch", "case", "do", "public", "private", "static", "string", "long", "unsigned",
}
_C_LIKE = {"c", "cpp", "c++", "java", "javascript", "js", "typescript", "ts", "go", "rust", "csharp", "c#", "kotlin", "swift", "php", "dart"}
_REST = r"(?P<string>\"[^\"]*\"|'[^']*')|(?P<number>\b\d+(?:\.\d+)?\b)|(?P<word>[A-Za-z_]\w*)"
_TOKEN_HASH = re.compile(r"(?P<comment>#.*)|" + _REST)    # Python, shell, Ruby, ...: '#' starts a comment, '//' is an operator
_TOKEN_SLASH = re.compile(r"(?P<comment>//.*)|" + _REST)  # C, Java, JavaScript, ...: '//' starts a comment


def split_indent(line: str) -> tuple[int, str]:
    """(indent in columns, text without the leading whitespace). A tab counts as four columns."""
    line = line.replace("\t", "    ").rstrip()
    body = line.lstrip(" ")
    return len(line) - len(body), body


def token_colours(body: str, language: str = "python") -> dict[str, str]:
    """Manim ``t2c`` slice keys ('[3:7]') for keywords, strings, numbers and comments in one line of code."""
    out: dict[str, str] = {}
    pattern = _TOKEN_SLASH if language.strip().lower() in _C_LIKE else _TOKEN_HASH
    for m in pattern.finditer(body):
        kind = m.lastgroup
        colour = (theme.MUTED if kind == "comment" else theme.GOOD if kind == "string" else theme.WARN if kind == "number"
                  else theme.PURPLE if m.group() in _KEYWORDS else None)
        if colour:
            out[f"[{m.start()}:{m.end()}]"] = colour
    return out


@component(
    "code_walkthrough",
    "Shows up to 8 lines of code and highlights them one at a time with a short note per line. Use to explain code or an "
    "algorithm line by line. For longer code use several scenes with first_line set so numbering continues; narration "
    "should mention the lines in order.",
    {"lines": P("string[]", "1-8 lines of code, indentation with spaces, each under 60 characters", required=True),
     "notes": P("string[]", "One short plain-words note per line, same order and same count as lines (under 70 chars each)", required=True),
     "language": P("string", "Language name shown as a label", default="python"),
     "title": P("string", "Heading, e.g. 'Binary search'", default=""),
     "first_line": P("integer", "Number of the first line shown, so long code can be split across scenes", default=1)},
    {"lines": ["def total(xs):", "    s = 0", "    for x in xs:", "        s += x", "    return s"],
     "notes": ["Define a function taking a list", "Start the running sum at zero", "Visit each item in turn",
               "Add the item to the sum", "Give the sum back"],
     "language": "python", "title": "Summing a list", "first_line": 1},
    domain=DOMAIN,
)
def code_walkthrough(ctx: Ctx, lines: list[str], notes: list[str], language: str = "python", title: str = "",
                     first_line: int = 1) -> None:
    lines = lines[:8]
    notes = (list(notes) + [""] * len(lines))[: len(lines)]
    size = 28 if len(lines) <= 6 else 25
    parsed = [split_indent(ln) for ln in lines]
    # Width of one monospace character, measured, so indentation lines up whatever font the host picked.
    cw = Text("0" * 20, font=MONO, font_size=size).width / 20
    code_rows = [_padded(body, size, theme.FG, font=MONO, t2c=token_colours(body, language)) for _, body in parsed]
    num_rows = [_padded(str(first_line + i), size, theme.MUTED, font=MONO) for i in range(len(lines))]
    pitch = max(r.height for r in code_rows + num_rows) + 0.12
    num_w = max(r.vis.width for r in num_rows)
    code_x = num_w + 0.55                       # where code starts, measured from the left edge of the line numbers
    content_w = max(code_x + ind * cw + r.vis.width for (ind, _), r in zip(parsed, code_rows))
    top = (len(lines) - 1) * pitch / 2
    for i, ((ind, _), cr, nr) in enumerate(zip(parsed, code_rows, num_rows)):
        y = top - i * pitch
        _put(nr, 0.0, y, "left")
        _put(cr, code_x + ind * cw, y, "left")
    pad = 0.45
    gutter = Line([num_w + 0.27, top + pitch / 2, 0], [num_w + 0.27, -top - pitch / 2, 0], color=theme.MUTED, stroke_width=1.5)
    panel = RoundedRectangle(corner_radius=0.18, color=theme.MUTED, stroke_width=1.5, fill_color=theme.PANEL, fill_opacity=1,
                             width=content_w + 2 * pad, height=len(lines) * pitch + 2 * pad - 0.1)
    panel.move_to([content_w / 2, 0, 0])
    tag = label(language, 20, theme.MUTED)
    tag.next_to(panel, UP, buff=0.08).align_to(panel, RIGHT)
    mark = Rectangle(width=panel.width - 0.2, height=pitch * 0.94, fill_color=theme.ACCENT, fill_opacity=0.22, stroke_width=0)
    mark.move_to([content_w / 2, top, 0])
    code_group = VGroup(*num_rows, *code_rows)
    stage = VGroup(panel, gutter, mark, code_group, tag)
    head = ctx.heading(title) if title else None
    note_h = 0.9
    stage_top, stage_bottom = (2.55 if head else 3.25), -1.0   # code sits high, the note goes under it
    factor = min(1.0, STAGE_W / max(panel.width, 1e-6), (stage_top - stage_bottom) / max(stage.height, 1e-6))
    if factor < 1.0:
        stage.scale(factor)
    stage.move_to([0, (stage_top + stage_bottom) / 2, 0])
    stage.shift(LEFT * panel.get_center()[0])
    note_y = stage.get_bottom()[1] - 0.2 - note_h / 2
    if head:
        ctx.play(Write(head), run_time=0.6)
    ctx.play(FadeIn(VGroup(panel, gutter, code_group, tag)), run_time=0.8)
    step = ctx.beat(ctx.remaining - 0.3, len(lines), cap=4.0, floor=0.8)
    note = None
    for i, (cr, text) in enumerate(zip(code_rows, notes)):
        target = [mark.get_center()[0], cr.get_center()[1], 0]
        new_note = None
        if text.strip():
            new_note = label(text.strip(), 26, theme.ACCENT, width=64)
            if new_note.width > STAGE_W:
                new_note.scale_to_fit_width(STAGE_W)
            new_note.move_to([0, note_y, 0])
        t = min(0.6, step * 0.5)
        anims = [mark.animate.move_to(target)] if i else [FadeIn(mark)]
        if note is not None:
            anims.append(FadeOut(note))
        if new_note is not None:
            anims.append(FadeIn(new_note, shift=UP * 0.12))
        ctx.play(*anims, run_time=t)
        ctx.wait(max(0.0, step - t))
        note = new_note


# ------------------------------------------------------------------------------------------------------ trace_table
def split_cells(row: str, n: int) -> list[str]:
    """'a | b | c' to exactly ``n`` cells (padded with blanks or cut)."""
    cells = [c.strip()[:30] for c in str(row).split("|")]
    return (cells + [""] * n)[:n]


@component(
    "trace_table",
    "A dry-run table that fills in one row per step, to follow how variables change while an algorithm or process runs.",
    {"columns": P("string[]", "2-6 column headings, e.g. variable names", required=True),
     "rows": P("string[]", "Up to 8 rows, cells separated by '|', e.g. 'lo=0 | hi=6 | mid=3'. Same cell count as columns.", required=True),
     "title": P("string", "Heading", default="Dry run")},
    {"columns": ["step", "lo", "hi", "mid"], "rows": ["1 | 0 | 6 | 3", "2 | 4 | 6 | 5", "3 | 4 | 4 | 4"], "title": "Tracing the search"},
    domain=DOMAIN,
)
def trace_table(ctx: Ctx, columns: list[str], rows: list[str], title: str = "Dry run") -> None:
    columns = [c.strip()[:30] for c in columns[:6]]
    n = len(columns)
    body_rows = [split_cells(r, n) for r in rows[:8]]
    head = ctx.heading(title)
    size = 30 if len(body_rows) <= 5 else 26
    header = [_padded(c, size, theme.ACCENT, weight="BOLD") for c in columns]
    cells = [[_padded(t, size, theme.FG) for t in r] for r in body_rows]
    widths = [max([header[j].vis.width] + [r[j].vis.width for r in cells]) + 0.8 for j in range(n)]
    total = sum(widths)
    xs, x = [], -total / 2
    for w in widths:
        xs.append(x + w / 2)
        x += w
    pitch = max(0.62, header[0].height + 0.1)
    top = (len(body_rows) * pitch) / 2 + pitch / 2
    for j, h in enumerate(header):
        _put(h, xs[j], top)
    rule = Line([-total / 2, top - pitch / 2, 0], [total / 2, top - pitch / 2, 0], color=theme.MUTED, stroke_width=2)
    row_groups = []
    for i, r in enumerate(cells):
        y = top - pitch * (i + 1)
        for j, c in enumerate(r):
            _put(c, xs[j], y)
        row_groups.append(VGroup(*r))
    row_shown = [VGroup(*[c.shown for c in r if len(c.shown)]) for r in cells]
    hl = Rectangle(width=total, height=pitch * 0.92, fill_color=theme.YELLOW, fill_opacity=0.16, stroke_width=0)
    table = VGroup(*header, rule, *row_groups, hl)
    ctx.fit(table, max_w=12.0, max_h=4.6, center=(0, 0.0, 0))
    ctx.play(Write(head), run_time=0.7)
    ctx.play(FadeIn(VGroup(*header)), Create(rule), run_time=0.7)
    step = ctx.beat(ctx.remaining - 0.3, max(1, len(row_groups)), cap=2.5, floor=0.6)
    prev = None
    cx = rule.get_center()[0]
    for i, rg in enumerate(row_groups):
        target = [cx, rg.get_center()[1], 0]
        if i == 0:
            hl.move_to(target)
        anims = [FadeIn(rg, shift=UP * 0.15), FadeIn(hl) if i == 0 else hl.animate.move_to(target)]
        if prev is not None and len(prev):
            anims.append(prev.animate.set_opacity(0.6))
        ctx.play(*anims, run_time=min(0.7, step * 0.6))
        ctx.wait(max(0.0, step - 0.7))
        prev = row_shown[i]
