"""Positional encoding as waves, as a sum onto embeddings, and as a heatmap (README section 6.2)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from manim import (DOWN, LEFT, PI, RIGHT, UP, AnimationGroup, Dot, Group, Line, ParametricFunction, Rectangle, Square, Succession, Text, UpdateFromAlphaFunc,
                   ValueTracker, VGroup)

from ..core.grids import HeatmapGrid
from ..core.palette import PALETTE, colormap, lerp_color
from ._math import MAX_TOKENS, pe_frequencies, seeded_embeddings, sinusoidal_pe

MAX_WAVE_ROWS = 4
WAVE_X0, WAVE_W = -4.3, 6.2   # where the curves start and how wide they are
ROW_H = 1.15
AMP = 0.38


@dataclass(frozen=True)
class PEWavesData:
    position: float
    freqs: np.ndarray
    wavelengths: np.ndarray
    vector: np.ndarray  # sin0, cos0, sin1, cos1, ... at the current position


class PositionalEncodingWaves(VGroup):
    """Intuition: a clock with several hands. The fast hand tells neighbouring positions apart; the slow hands tell far-apart
    positions apart. Read all the hands at one position and you get that position's address, written as numbers.
    """

    def __init__(self, d_model: int = 8, n_positions: int = 12, n_freqs: int = 4, *, base: float = 10000.0, position: float = 0.0):
        super().__init__()
        if d_model < 2 or d_model % 2:
            raise ValueError("d_model must be an even number, at least 2")
        if not 2 <= n_positions <= MAX_TOKENS:
            raise ValueError(f"n_positions must be between 2 and {MAX_TOKENS}")
        if not 1 <= n_freqs <= min(MAX_WAVE_ROWS, d_model // 2):
            raise ValueError(f"n_freqs must be between 1 and {min(MAX_WAVE_ROWS, d_model // 2)} for d_model={d_model}")
        if base <= 1:
            raise ValueError("base must be greater than 1")
        self.d_model, self.n_positions, self.n_freqs, self.base = d_model, n_positions, n_freqs, float(base)
        self.pos = ValueTracker(0.0)
        freqs, wavelengths = pe_frequencies(d_model, base)
        self.freqs, self.wavelengths = freqs[:n_freqs], wavelengths[:n_freqs]
        sx = WAVE_W / n_positions
        top = (n_freqs - 1) * ROW_H / 2
        self._sx, self._row_y = sx, [top - k * ROW_H for k in range(n_freqs)]
        self.curves: list[tuple[ParametricFunction, ParametricFunction]] = []
        static = VGroup()
        self._ref: Line | None = None
        for k in range(n_freqs):
            y0, w = self._row_y[k], float(self.freqs[k])
            base_line = Line((WAVE_X0, y0, 0), (WAVE_X0 + WAVE_W, y0, 0), stroke_width=1, color=PALETTE.muted)
            self._ref = self._ref or base_line
            static.add(base_line)
            s = ParametricFunction(lambda t, y0=y0, w=w: (WAVE_X0 + t * sx, y0 + AMP * np.sin(w * t), 0), t_range=[0, n_positions, 0.05],
                                   color=PALETTE.blue, stroke_width=4)
            c = ParametricFunction(lambda t, y0=y0, w=w: (WAVE_X0 + t * sx, y0 + AMP * np.cos(w * t), 0), t_range=[0, n_positions, 0.05],
                                   color=PALETTE.teal, stroke_width=4)
            self.curves.append((s, c))
            static.add(s, c)
            static.add(Text(f"wavelength {self.wavelengths[k]:.1f}", font_size=18, color=PALETTE.muted).next_to((WAVE_X0, y0, 0), LEFT, buff=0.15))
        for p in range(0, n_positions + 1, 1 if n_positions <= 12 else 2):
            x = WAVE_X0 + p * sx
            static.add(Line((x, self._row_y[-1] - 0.62, 0), (x, self._row_y[-1] - 0.52, 0), stroke_width=2, color=PALETTE.muted))
            static.add(Text(str(p), font_size=16, color=PALETTE.muted).move_to((x, self._row_y[-1] - 0.8, 0)))
        static.add(Text("position", font_size=18, color=PALETTE.muted).move_to((WAVE_X0 + WAVE_W / 2, self._row_y[-1] - 1.15, 0)))
        # the readout: 2 * n_freqs cells, the numbers the waves are "read off" into
        self.cells = VGroup(*(Square(side_length=0.42, stroke_width=1, stroke_color=PALETTE.bg, fill_opacity=1.0) for _ in range(2 * n_freqs)))
        self.cells.arrange(DOWN, buff=0.04).move_to((3.7, 0, 0))
        head = Text("PE(position)", font_size=22, color=PALETTE.fg).next_to(self.cells, UP, buff=0.3)
        self.pos_text = Text("", font_size=22, color=PALETTE.yellow)
        self.cursor = Line(UP * 0, UP * 0, color=PALETTE.yellow, stroke_width=3)
        self.dots = VGroup(*(Dot(radius=0.06, color=PALETTE.yellow) for _ in range(2 * n_freqs)))
        self.add(static, self.cells, head, self.pos_text, self.cursor, self.dots)
        self.set_position(position)
        self.add_updater(lambda m: m._sync())
        self._last = None

    def _vector(self, p: float) -> np.ndarray:
        v = np.empty(2 * self.n_freqs)
        v[0::2], v[1::2] = np.sin(p * self.freqs), np.cos(p * self.freqs)
        return v

    def _sync(self) -> None:
        p = float(self.pos.get_value())
        if p != getattr(self, "_last", None):
            self._apply(p)

    def _world(self, x: float, y: float):
        """Local layout coordinates -> where they are now, so the group can be moved or scaled by ``ctx.fit``."""
        start, end = self._ref.get_start(), self._ref.get_end()
        k = float(np.linalg.norm(end - start)) / WAVE_W
        return start + k * np.array([x - WAVE_X0, y - self._row_y[0], 0.0])

    def _apply(self, p: float) -> None:
        self._last = p
        x = WAVE_X0 + p * self._sx
        self.cursor.put_start_and_end_on(self._world(x, self._row_y[0] + 0.6), self._world(x, self._row_y[-1] - 0.6))
        vec = self._vector(p)
        for k in range(self.n_freqs):
            for h, f in ((0, np.sin), (1, np.cos)):
                self.dots[2 * k + h].move_to(self._world(x, self._row_y[k] + AMP * f(self.freqs[k] * p)))
        for cell, v in zip(self.cells, vec):
            cell.set_fill(colormap("coolwarm", v, -1.0, 1.0))
        self.pos_text.become(Text(f"position {p:.1f}", font_size=22 * self._text_scale(), color=PALETTE.yellow).next_to(self.cells, DOWN, buff=0.3 * self._text_scale()))

    def _text_scale(self) -> float:
        return float(np.linalg.norm(self._ref.get_end() - self._ref.get_start())) / WAVE_W

    def set_position(self, p: float) -> "PositionalEncodingWaves":
        """Put the cursor at ``p`` (between 0 and ``n_positions``) right now."""
        if not 0 <= p <= self.n_positions:
            raise ValueError(f"position must be between 0 and {self.n_positions}, got {p}")
        self.pos.set_value(float(p))
        self._apply(float(p))
        return self

    def animate_scan(self, position: float, *, run_time: float = 2.0):
        """Slide the cursor to ``position``; the readout column recolours as it moves."""
        if not 0 <= position <= self.n_positions:
            raise ValueError(f"position must be between 0 and {self.n_positions}, got {position}")
        anim = self.pos.animate.set_value(float(position)).build()
        anim.run_time = run_time
        return anim

    def bind(self, tracker: ValueTracker, param: str = "position") -> "PositionalEncodingWaves":
        """Follow an external tracker (the slider behaviour)."""
        if param != "position":
            raise ValueError("PositionalEncodingWaves can only bind 'position'")
        self.add_updater(lambda m: m.pos.set_value(min(max(tracker.get_value(), 0.0), m.n_positions)))
        return self

    @property
    def data(self) -> PEWavesData:
        p = float(self.pos.get_value())
        return PEWavesData(p, self.freqs, self.wavelengths, self._vector(p))


@dataclass(frozen=True)
class PESumData:
    tokens: tuple[str, ...]
    embeddings: np.ndarray
    pe: np.ndarray
    total: np.ndarray


class PositionalEncodingSum(Group):
    """Intuition: stamping an address onto each word. The embedding says what the word is, the wave pattern says where it
    sits, and the model only ever sees their sum, so the same word in two places arrives as two slightly different arrows.
    """

    MAX_TOKENS = 8
    MAX_D = 8

    def __init__(self, tokens: list[str], d_model: int = 8, *, embeddings=None, seed: int = 7, base: float = 10000.0):
        super().__init__()
        n = len(tokens)
        if not 1 <= n <= self.MAX_TOKENS:
            raise ValueError(f"tokens must hold between 1 and {self.MAX_TOKENS} items")
        if d_model < 2 or d_model % 2 or d_model > self.MAX_D:
            raise ValueError(f"d_model must be an even number between 2 and {self.MAX_D}")
        emb = seeded_embeddings(n, d_model, seed) if embeddings is None else np.asarray(embeddings, dtype=float)
        if emb.shape != (n, d_model):
            raise ValueError(f"embeddings must have shape ({n}, {d_model}), got {emb.shape}")
        pe = sinusoidal_pe(n, d_model, base)
        total = emb + pe
        self.tokens = tuple(str(t) for t in tokens)
        self._data = PESumData(self.tokens, emb, pe, total)
        hi = float(np.max(np.abs(total))) or 1.0
        mk = lambda a: HeatmapGrid(a, cmap="coolwarm", vmin=-hi, vmax=hi, cell=0.38, show_values=False)  # noqa: E731
        self.emb_grid, self.pe_grid, self.sum_grid = mk(emb), mk(pe), mk(total)
        w = d_model * 0.38
        xs = (-w - 0.9, 0.0, w + 0.9)
        parts = Group()
        for grid, x, title, color in ((self.emb_grid, xs[0], "embedding: what", PALETTE.blue), (self.pe_grid, xs[1], "position: where", PALETTE.teal),
                                      (self.sum_grid, xs[2], "sum: what the model sees", PALETTE.yellow)):
            grid.move_to((x, 0, 0))
            parts.add(grid, Text(title, font_size=20, color=color).next_to(grid, UP, buff=0.3))
        parts.add(Text("+", font_size=34, color=PALETTE.muted).move_to(((xs[0] + xs[1]) / 2, 0, 0)),
                  Text("=", font_size=34, color=PALETTE.muted).move_to(((xs[1] + xs[2]) / 2, 0, 0)))
        for i, t in enumerate(self.tokens):
            parts.add(Text(t, font_size=20, color=PALETTE.muted).next_to(self.emb_grid.cells[i][0], LEFT, buff=0.15))
        self.add(parts)
        self.n = n

    def animate_add(self, *, run_time: float = 3.0):
        """Token by token: light its embedding row and position row, then let the sum row fill in."""
        steps = []
        per = run_time / self.n
        for i in range(self.n):
            fills = []
            for j, cell in enumerate(self.sum_grid.cells[i]):
                target = cell.get_fill_color().to_hex()
                fills.append(UpdateFromAlphaFunc(cell, lambda m, a, t=target: m.set_fill(lerp_color(PALETTE.panel, t, a))))
            steps.append(AnimationGroup(self.emb_grid.highlight_row(i), self.pe_grid.highlight_row(i), AnimationGroup(*fills), run_time=per, lag_ratio=0.0))
        return Succession(*steps)

    def bind(self, tracker: ValueTracker, param: str = "token") -> "PositionalEncodingSum":
        raise ValueError("PositionalEncodingSum has no continuous parameter to bind; use PositionalEncodingWaves for a slider")

    @property
    def data(self) -> PESumData:
        return self._data


@dataclass(frozen=True)
class PEHeatmapData:
    pe: np.ndarray
    position: int | None
    dim: int | None


class PEHeatmap(Group):
    """Intuition: every wave seen from above. Each column is one wave (left columns wiggle fast, right columns barely
    move) and each row is one position's address. Pick a column and you can see which wave it is.
    """

    MAX_POSITIONS = 64
    MAX_DIMS = 64

    def __init__(self, n_positions: int, d_model: int, *, base: float = 10000.0):
        super().__init__()
        if not 2 <= n_positions <= self.MAX_POSITIONS:
            raise ValueError(f"n_positions must be between 2 and {self.MAX_POSITIONS}")
        if d_model < 2 or d_model % 2 or d_model > self.MAX_DIMS:
            raise ValueError(f"d_model must be an even number between 2 and {self.MAX_DIMS}")
        self.n_positions, self.d_model = n_positions, d_model
        self._pe = sinusoidal_pe(n_positions, d_model, base)
        self.cell = min(0.3, 4.4 / n_positions, 9.0 / d_model)
        self.grid = HeatmapGrid(self._pe, cmap="coolwarm", vmin=-1.0, vmax=1.0, cell=self.cell, show_values=False)
        self.add(self.grid)
        self.add(Text("dimension", font_size=20, color=PALETTE.muted).next_to(self.grid, UP, buff=0.25))
        self.add(Text("position", font_size=20, color=PALETTE.muted).rotate(PI / 2).next_to(self.grid, LEFT, buff=0.25))
        self.col_marker = Rectangle(width=self.cell, height=n_positions * self.cell, stroke_color=PALETTE.yellow, stroke_width=4, fill_opacity=0)
        self.row_marker = Rectangle(width=d_model * self.cell, height=self.cell, stroke_color=PALETTE.teal, stroke_width=4, fill_opacity=0)
        for mk in (self.col_marker, self.row_marker):
            mk.set_stroke(opacity=0.0)
            self.add(mk)
        self._pos: int | None = None
        self._dim: int | None = None

    def _col_target(self, j: int):
        return np.array([self.grid.get_left()[0] + (j + 0.5) * self.cell, self.grid.get_center()[1], 0.0])

    def _row_target(self, p: int):
        return np.array([self.grid.get_center()[0], self.grid.get_top()[1] - (p + 0.5) * self.cell, 0.0])

    def _check_dim(self, j: int) -> int:
        if not 0 <= j < self.d_model:
            raise ValueError(f"dimension {j} is outside 0..{self.d_model - 1}")
        return j

    def _check_pos(self, p: int) -> int:
        if not 0 <= p < self.n_positions:
            raise ValueError(f"position {p} is outside 0..{self.n_positions - 1}")
        return p

    @staticmethod
    def wave_of(dim: int) -> tuple[int, str]:
        """Which wave a column belongs to: ``(pair index, "sin" or "cos")``; pair k matches row k of PositionalEncodingWaves."""
        return dim // 2, "sin" if dim % 2 == 0 else "cos"

    def set_dim(self, j: int) -> "PEHeatmap":
        self.col_marker.move_to(self._col_target(self._check_dim(j))).set_stroke(opacity=1.0)
        self._dim = j
        return self

    def set_position(self, p: int) -> "PEHeatmap":
        self.row_marker.move_to(self._row_target(self._check_pos(p))).set_stroke(opacity=1.0)
        self._pos = p
        return self

    def animate_dim(self, j: int, *, run_time: float = 0.8):
        """Slide the yellow column marker to dimension ``j``."""
        self._check_dim(j)
        self._dim = j
        return self.col_marker.animate(run_time=run_time).move_to(self._col_target(j)).set_stroke(opacity=1.0)

    def animate_position(self, p: int, *, run_time: float = 0.8):
        """Slide the teal row marker to position ``p``."""
        self._check_pos(p)
        self._pos = p
        return self.row_marker.animate(run_time=run_time).move_to(self._row_target(p)).set_stroke(opacity=1.0)

    def bind(self, tracker: ValueTracker, param: str = "position") -> "PEHeatmap":
        """Keep the row marker on the (rounded) position held by the tracker."""
        if param != "position":
            raise ValueError("PEHeatmap can only bind 'position'")
        self.add_updater(lambda m: m.set_position(int(round(min(max(tracker.get_value(), 0), m.n_positions - 1)))))
        return self

    @property
    def data(self) -> PEHeatmapData:
        return PEHeatmapData(self._pe, self._pos, self._dim)
