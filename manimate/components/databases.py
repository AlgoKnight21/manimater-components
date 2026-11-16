"""Databases pack, built on the community plugin manim-databases (https://github.com/pg-pilot/manim-databases, MIT).

manim-databases is early-alpha and is not on PyPI, so ``scripts/install_plugins.sh`` installs it from a pinned commit.
Only the two mobjects that work today (``MTable`` and ``MBTree``) are wrapped; its others are stubs upstream.
The SQL text shown under the table is generated here from the same operations that drive the animation.
"""
from __future__ import annotations

import re
from typing import Any

from manim import Create, FadeIn, Transform, Write

from .. import theme
from ..ctx import Ctx, label
from ..registry import Param as P, component
from ._plugin import require

_PIP = "manim-databases"
_CY = 0.55

# --------------------------------------------------------------------------------------------------------------
# Table
# --------------------------------------------------------------------------------------------------------------
_INSERT = re.compile(r"^\s*insert\s+(.+)$", re.IGNORECASE)
_UPDATE = re.compile(r"^\s*update\s+row\s+(\d{1,2})\s+(\w+)\s+(.+?)\s*$", re.IGNORECASE)
_DELETE = re.compile(r"^\s*delete\s+row\s+(\d{1,2})\s*$", re.IGNORECASE)
_SELECT = re.compile(r"^\s*select\s+row\s+(\d{1,2})\s*$", re.IGNORECASE)


def _cell(text: str) -> Any:
    text = text.strip()
    return int(text) if re.fullmatch(r"-?\d{1,9}", text) else text


def _split_row(row: str) -> list[Any]:
    return [_cell(c) for c in row.split(",")]


def _sql(v: Any) -> str:
    return str(v) if isinstance(v, int) else "'" + str(v).replace("'", "''") + "'"


def _rows_ok(v: list) -> str | None:
    if not 1 <= len(v) <= 6:
        return "needs 1 to 6 rows"
    return None


def _ops_ok(v: list) -> str | None:
    if len(v) > 5:
        return "at most 5 operations"
    for op in v:
        if not any(p.match(op) for p in (_INSERT, _UPDATE, _DELETE, _SELECT)):
            return f"cannot read {op!r}; use 'insert a, b', 'update row 2 col value', 'delete row 1' or 'select row 2'"
    return None


def _columns_ok(v: list) -> str | None:
    if not 2 <= len(v) <= 5:
        return "needs 2 to 5 columns"
    if any(not re.fullmatch(r"\w{1,12}", c) for c in v):
        return "column names must be single words of up to 12 letters, digits or underscores"
    return None


@component(
    "database_table",
    "A table as rows are inserted, updated, deleted or selected, with the SQL shown.",
    {"columns": P("string[]", "2-5 column names; the first is the key", validator=_columns_ok),
     "rows": P("string[]", "1-6 rows like '1, alice, shipped'", validator=_rows_ok),
     "operations": P("string[]", "'insert a, b' | 'update row 2 col value' | 'delete row 1' | 'select row 2'",
                     default=[], validator=_ops_ok)},
    {"columns": ["id", "customer", "status"], "rows": ["1, alice, shipped", "2, bob, pending"],
     "operations": ["insert 3, carol, shipped", "update row 2 status shipped", "delete row 1"]},
    domain="databases",
    plugin="manim-databases",
)
def database_table(ctx: Ctx, columns: list[str] | None = None, rows: list[str] | None = None,
                   operations: list[str] | None = None) -> None:
    db = require("manim_databases", _PIP)
    cols = columns or ["id", "customer", "status"]
    width = len(cols)
    model: list[list[Any]] = []
    for raw in rows or ["1, alice, shipped"]:
        cells = _split_row(raw)[:width]
        model.append(cells + [""] * (width - len(cells)))  # a short row is padded, a long one trimmed
    table = db.MTable(columns=cols, rows=[list(r) for r in model], primary_key=cols[0])
    table.move_to((0, _CY + 0.6, 0))
    key = cols[0]
    head = ctx.heading("A database table")
    sql_line = label("", 26, theme.YELLOW).move_to((0, -2.1, 0))
    ctx.play(Write(head), Create(table), run_time=1.2)
    ctx.add(sql_line)
    ops = list(operations or [])
    t = max(0.6, min(1.5, (ctx.remaining - 1.0) / max(1, len(ops))))
    for op in ops:
        sql, anim = None, None
        if m := _INSERT.match(op):
            vals = _split_row(m.group(1))[:width]
            vals += [""] * (width - len(vals))
            sql = f"INSERT INTO t VALUES ({', '.join(_sql(v) for v in vals)});"
            anim = table.animate.insert_row(vals)
            model.append(vals)
        elif m := _UPDATE.match(op):
            row, col, value = int(m.group(1)) - 1, m.group(2), _cell(m.group(3))
            match = next((c for c in cols if c.lower() == col.lower()), None)
            if match and 0 <= row < len(model):
                sql = f"UPDATE t SET {match} = {_sql(value)} WHERE {key} = {_sql(model[row][0])};"
                anim = table.animate.update_cell(row, match, value)
                model[row][cols.index(match)] = value
        elif m := _DELETE.match(op):
            row = int(m.group(1)) - 1
            if 0 <= row < len(model):
                sql = f"DELETE FROM t WHERE {key} = {_sql(model[row][0])};"
                anim = table.animate.delete_row(row)
                del model[row]
        elif m := _SELECT.match(op):
            row = int(m.group(1)) - 1
            if 0 <= row < len(model):
                sql = f"SELECT * FROM t WHERE {key} = {_sql(model[row][0])};"
                anim = table.animate.highlight_row(row)
        if anim is None:  # an operation that does not fit the table (row out of range, unknown column) is skipped
            continue
        new_line = label(sql, 26, theme.YELLOW).move_to((0, -2.1, 0))
        ctx.play(anim, Transform(sql_line, new_line), run_time=t)
    ctx.wait(ctx.remaining)


# --------------------------------------------------------------------------------------------------------------
# B-tree index
# --------------------------------------------------------------------------------------------------------------
def _keys_ok(v: list) -> str | None:
    if not 3 <= len(v) <= 12:
        return "needs 3 to 12 keys"
    if any(not float(x).is_integer() or not 0 <= x <= 99 for x in v):
        return "keys must be whole numbers from 0 to 99"
    if len(set(v)) != len(v):
        return "keys must all be different"
    return None


@component(
    "btree_index",
    "B-tree index built from keys, then one key inserted or searched.",
    {"keys": P("number[]", "3-12 different numbers 0-99", validator=_keys_ok),
     "then": P("string", "after building", default="search", choices=["none", "insert", "search"]),
     "key": P("number", "key to insert or search, 0-99", default=12,
              validator=lambda v: None if float(v).is_integer() and 0 <= v <= 99 else "must be a whole number from 0 to 99")},
    {"keys": [10, 20, 5, 6, 12, 30, 7, 17], "then": "search", "key": 12},
    domain="databases",
    plugin="manim-databases",
)
def btree_index(ctx: Ctx, keys: list[float] | None = None, then: str = "search", key: float = 12) -> None:
    db = require("manim_databases", _PIP)
    base = [int(k) for k in (keys or [10, 20, 5, 6, 12, 30, 7, 17])]
    key = int(key)
    tree = db.MBTree(order=3, keys=base, max_width=11.0)
    tree.move_to((0, _CY, 0))
    head = ctx.heading("B-tree index")
    ctx.play(Write(head), Create(tree), run_time=min(2.5, max(1.2, ctx.remaining * 0.3)))
    if then == "insert":
        note = label(f"Insert {key}" + (" (already there)" if key in base else ""), 30, theme.YELLOW).move_to((0, -2.3, 0))
        ctx.play(FadeIn(note), run_time=0.5)
        if key not in base:
            ctx.play(tree.animate.insert(key), run_time=max(1.0, min(3.0, ctx.remaining - 1.0)))
    elif then == "search":
        verdict = "found" if key in base else "not in the tree"
        note = label(f"Search for {key}: {verdict}", 30, theme.YELLOW if key in base else theme.DANGER).move_to((0, -2.3, 0))
        ctx.play(FadeIn(note), run_time=0.5)
        ctx.play(tree.animate.search(key), run_time=max(1.0, min(3.0, ctx.remaining - 1.0)))
    ctx.wait(ctx.remaining)
