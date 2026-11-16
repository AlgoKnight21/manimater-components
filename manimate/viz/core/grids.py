"""Grids of numbers: a coloured heatmap and a labelled matrix (README section 5.3)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, AnimationGroup, Group, ImageMobject, Indicate, Square, Succession, Text, VGroup)

from .palette import PALETTE, colormap, hex_to_rgb

MAX_CELL_GRID = 16  # above this many rows or columns the grid becomes one image (performance budget)
SHOW_VALUES_UP_TO = 6


def _as_matrix(values) -> np.ndarray:
    a = np.asarray(values, dtype=float)
    if a.ndim != 2 or a.size == 0:
        raise ValueError("values must be a non-empty 2-D array")
    if not np.all(np.isfinite(a)):
        raise ValueError("values must be finite numbers")
    return a


@dataclass(frozen=True)
class HeatmapData:
    values: np.ndarray
    vmin: float
    vmax: float


class HeatmapGrid(Group):  # Group, not VGroup: the big-grid mode holds an ImageMobject
    """Intuition: numbers as terrain colour: bright means big. Row and column labels say what each axis is.

    It is a ``Group`` (not a ``VGroup``) because grids above 16 x 16 hold an ``ImageMobject``; put it in a ``Group``.
    """

    def __init__(self, values, *, cmap: str = "viridis", vmin: float | None = None, vmax: float | None = None,
                 cell: float = 0.4, row_labels: list[str] | None = None, col_labels: list[str] | None = None,
                 show_values: bool | None = None):
        super().__init__()
        a = _as_matrix(values)
        self.cmap, self.cell = cmap, cell
        self.vmin = float(a.min()) if vmin is None else vmin
        self.vmax = float(a.max()) if vmax is None else vmax
        if self.vmax == self.vmin:
            self.vmax = self.vmin + 1.0
        self.values = a
        rows, cols = a.shape
        self.as_image = rows > MAX_CELL_GRID or cols > MAX_CELL_GRID
        self.show_values = (rows <= SHOW_VALUES_UP_TO and cols <= SHOW_VALUES_UP_TO) if show_values is None else show_values
        self.cells: list[list[Square]] = []
        if self.as_image:
            self.image = self._image(a)
            self.add(self.image)
        else:
            for i in range(rows):
                row = []
                for j in range(cols):
                    sq = Square(side_length=cell, stroke_width=0.5, stroke_color=PALETTE.bg, fill_opacity=1.0,
                                fill_color=colormap(cmap, a[i, j], self.vmin, self.vmax))
                    sq.move_to((j * cell, -i * cell, 0))
                    row.append(sq)
                    self.add(sq)
                self.cells.append(row)
            if self.show_values:
                for i in range(rows):
                    for j in range(cols):
                        t = Text(f"{a[i, j]:.1f}", font_size=max(10, int(cell * 40)), color=PALETTE.bg if
                                 a[i, j] > (self.vmin + self.vmax) / 2 else PALETTE.fg)
                        self.add(t.move_to(self.cells[i][j]))
        self.center()
        self._add_labels(row_labels, col_labels)

    def _image(self, a: np.ndarray) -> ImageMobject:
        px = np.array([[hex_to_rgb(colormap(self.cmap, v, self.vmin, self.vmax)) for v in row] for row in a])
        img = ImageMobject((px * 255).astype(np.uint8))
        img.set_resampling_algorithm(0)
        return img.scale_to_fit_width(a.shape[1] * self.cell)

    def _add_labels(self, row_labels, col_labels) -> None:
        rows, cols = self.values.shape
        if self.as_image:
            return
        if row_labels:
            for i, name in enumerate(row_labels[:rows]):
                self.add(Text(name, font_size=18, color=PALETTE.muted).next_to(self.cells[i][0], LEFT, buff=0.15))
        if col_labels:
            for j, name in enumerate(col_labels[:cols]):
                self.add(Text(name, font_size=18, color=PALETTE.muted).next_to(self.cells[0][j], UP, buff=0.15))

    @property
    def data(self) -> HeatmapData:
        return HeatmapData(self.values, self.vmin, self.vmax)

    def animate_values(self, new, run_time: float = 1.0):
        """Return an animation that recolours every cell to the new values (same shape)."""
        b = _as_matrix(new)
        if b.shape != self.values.shape:
            raise ValueError("new values must have the same shape as the old ones")
        self.values = b
        if self.as_image:
            raise ValueError("grids above 16 x 16 are one image; build a new HeatmapGrid and cross-fade instead")
        return AnimationGroup(*(self.cells[i][j].animate.set_fill(colormap(self.cmap, b[i, j], self.vmin, self.vmax))
                                for i in range(b.shape[0]) for j in range(b.shape[1])), run_time=run_time)

    def highlight_cell(self, i: int, j: int):
        if self.as_image:
            raise ValueError("cannot highlight a single cell in image mode")
        return Indicate(self.cells[i][j], color=PALETTE.yellow)

    def highlight_row(self, i: int):
        if self.as_image:
            raise ValueError("cannot highlight a row in image mode")
        return Indicate(VGroup(*self.cells[i]), color=PALETTE.yellow)


class MatrixGrid(VGroup):
    """Intuition: a matrix as a table you can sweep with your finger: rows against columns."""

    def __init__(self, values, *, name: str | None = None, row_labels: list[str] | None = None,
                 col_labels: list[str] | None = None, decimals: int = 2, color: str = PALETTE.fg):
        super().__init__()
        a = _as_matrix(values)
        self.values, self.rows, self.cols = a, a.shape[0], a.shape[1]
        self.entries = [[Text(f"{a[i, j]:.{decimals}f}", font_size=24, color=color).move_to((j * 1.0, -i * 0.6, 0))
                         for j in range(self.cols)] for i in range(self.rows)]
        body = VGroup(*(e for row in self.entries for e in row)).center()
        left = Text("[", font_size=int(60 + 20 * self.rows), color=PALETTE.muted).next_to(body, LEFT, buff=0.2)
        right = Text("]", font_size=int(60 + 20 * self.rows), color=PALETTE.muted).next_to(body, RIGHT, buff=0.2)
        self.add(body, left, right)
        if name:
            self.add(Text(f"{name}  ({self.rows}x{self.cols})", font_size=20, color=PALETTE.muted).next_to(body, UP, buff=0.35))
        for i, nm in enumerate((row_labels or [])[: self.rows]):
            self.add(Text(nm, font_size=18, color=PALETTE.muted).next_to(left, LEFT, buff=0.15).set_y(self.entries[i][0].get_y()))
        for j, nm in enumerate((col_labels or [])[: self.cols]):
            self.add(Text(nm, font_size=18, color=PALETTE.muted).next_to(body, DOWN, buff=0.25).set_x(self.entries[0][j].get_x()))

    def row(self, i: int) -> VGroup:
        return VGroup(*self.entries[i])

    def col(self, j: int) -> VGroup:
        return VGroup(*(r[j] for r in self.entries))

    def animate_multiply(self, other: "MatrixGrid", run_time: float = 3.0):
        """Sweep each row of ``self`` against the matching column of ``other`` (inner sizes must agree)."""
        if self.cols != other.rows:
            raise ValueError(f"cannot multiply {self.rows}x{self.cols} by {other.rows}x{other.cols}")
        steps = [AnimationGroup(Indicate(self.row(i), color=PALETTE.yellow), Indicate(other.col(i % other.cols), color=PALETTE.yellow))
                 for i in range(self.rows)]
        return Succession(*steps, run_time=run_time)
