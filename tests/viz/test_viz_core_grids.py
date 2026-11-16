import numpy as np
import pytest

from manimate.viz.core.grids import MAX_CELL_GRID, HeatmapGrid, MatrixGrid
from manimate.viz.core.palette import colormap


def test_heatmap_colours_cells_from_values_and_shows_numbers_when_small():
    h = HeatmapGrid([[0.0, 1.0], [0.5, 0.25]])
    assert h.cells[0][1].get_fill_color().to_hex().lower() == colormap("viridis", 1.0, 0, 1).lower()
    assert h.show_values and h.data.vmax == 1.0 and len(h.cells) == 2


def test_big_heatmap_becomes_one_image():
    n = MAX_CELL_GRID + 1
    h = HeatmapGrid(np.random.default_rng(0).random((n, n)))
    assert h.as_image and h.cells == [] and len(h) == 1
    with pytest.raises(ValueError, match="image"):
        h.highlight_row(0)


def test_heatmap_rejects_bad_input_and_shape_changes():
    with pytest.raises(ValueError):
        HeatmapGrid([1, 2, 3])
    with pytest.raises(ValueError):
        HeatmapGrid([[np.nan]])
    with pytest.raises(ValueError, match="same shape"):
        HeatmapGrid([[1, 2]]).animate_values([[1]])


def test_matrix_rows_cols_and_multiplication_shape_check():
    a, b = MatrixGrid([[1, 2, 3], [4, 5, 6]], name="A"), MatrixGrid([[1], [2], [3]])
    assert len(a.row(0)) == 3 and len(a.col(1)) == 2
    assert a.animate_multiply(b) is not None
    with pytest.raises(ValueError, match="cannot multiply"):
        b.animate_multiply(a)
