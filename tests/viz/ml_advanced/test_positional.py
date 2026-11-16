import numpy as np
import pytest
from manim import Animation, ValueTracker

from manimate.ctx import STAGE_H, STAGE_W
from manimate.viz.ml_advanced import _math as m
from manimate.viz.ml_advanced.positional import PositionalEncodingWaves


def test_readout_vector_equals_the_pe_matrix_row():
    w = PositionalEncodingWaves(d_model=8, n_positions=12, n_freqs=4).set_position(5)
    assert np.allclose(w.data.vector, m.sinusoidal_pe(6, 8)[5])


def test_wavelengths_grow_geometrically_and_start_at_two_pi():
    w = PositionalEncodingWaves(d_model=8, n_freqs=4)
    assert np.isclose(w.data.wavelengths[0], 2 * np.pi)
    r = w.data.wavelengths[1:] / w.data.wavelengths[:-1]
    assert np.allclose(r, r[0]) and r[0] > 1


def test_cursor_tracks_the_position_and_cells_recolour():
    w = PositionalEncodingWaves(base=100.0)
    before = [c.get_fill_color().to_hex() for c in w.cells]
    w.set_position(6)
    after = [c.get_fill_color().to_hex() for c in w.cells]
    assert w.cursor.get_center()[0] > -4.3 and before != after
    assert len(w.dots) == len(w.cells) == 8


def test_animate_scan_is_an_animation_and_updater_follows_the_tracker():
    w = PositionalEncodingWaves()
    a = w.animate_scan(7, run_time=1.5)
    assert isinstance(a, Animation) and a.run_time == 1.5
    w.pos.set_value(7.0)
    w.update(0)
    assert w.data.position == 7.0 and np.allclose(w.data.vector, m.sinusoidal_pe(8, 8)[7])


def test_bind_follows_an_external_tracker_and_clamps():
    t = ValueTracker(3)
    w = PositionalEncodingWaves(n_positions=10).bind(t)
    w.update(0)
    assert w.data.position == 3.0
    t.set_value(99)
    w.update(0)
    assert w.data.position == 10.0


def test_fits_the_stage():
    w = PositionalEncodingWaves(d_model=16, n_positions=16, n_freqs=4)
    assert w.width <= STAGE_W and w.height <= STAGE_H


@pytest.mark.parametrize("kw, msg", [
    (dict(d_model=7), "even"),
    (dict(n_positions=1), "between 2"),
    (dict(n_freqs=5), "n_freqs"),
    (dict(d_model=4, n_freqs=3), "n_freqs"),
    (dict(base=1.0), "base"),
])
def test_bad_input(kw, msg):
    with pytest.raises(ValueError, match=msg):
        PositionalEncodingWaves(**kw)


def test_position_out_of_range():
    with pytest.raises(ValueError, match="between 0 and"):
        PositionalEncodingWaves(n_positions=8).set_position(9)
    with pytest.raises(ValueError, match="between 0 and"):
        PositionalEncodingWaves(n_positions=8).animate_scan(-1)


def test_scan_moves_the_cursor_when_played():
    from manimate.viz.core import dry_run
    box = {}

    def script(ctx):
        w = box["w"] = PositionalEncodingWaves(base=60.0)
        ctx.fit(w)
        ctx.add(w)
        ctx.play(w.animate_scan(8), run_time=1.0)

    ctx = dry_run(script, 4.0)
    w = box["w"]
    assert w.data.position == pytest.approx(8.0)
    assert np.allclose(w.data.vector, m.sinusoidal_pe(9, 8, base=60.0)[8])
    from manimate.ctx import STAGE_CENTER, STAGE_H, STAGE_W
    lo, hi = w.get_corner(np.array([-1, -1, 0])), w.get_corner(np.array([1, 1, 0]))  # the picture, not the ValueTracker
    assert abs(lo[0] - STAGE_CENTER[0]) <= STAGE_W / 2 and hi[0] <= STAGE_W / 2 and lo[1] >= STAGE_CENTER[1] - STAGE_H / 2 and hi[1] <= STAGE_CENTER[1] + STAGE_H / 2


# ---- PositionalEncodingSum -------------------------------------------------------------------------------------------
from manimate.viz.ml_advanced.positional import PositionalEncodingSum


def test_sum_is_embedding_plus_pe_and_grids_have_matching_shapes():
    s = PositionalEncodingSum(["a", "b", "c"], d_model=8)
    d = s.data
    assert np.allclose(d.total, d.embeddings + m.sinusoidal_pe(3, 8)) and d.total.shape == (3, 8)
    assert s.emb_grid.values.shape == s.pe_grid.values.shape == s.sum_grid.values.shape == (3, 8)


def test_same_word_at_two_positions_gets_two_different_inputs():
    E = np.tile(np.arange(8.0), (2, 1))  # identical embeddings
    d = PositionalEncodingSum(["x", "x"], 8, embeddings=E).data
    assert not np.allclose(d.total[0], d.total[1])


def test_animate_add_has_one_step_per_token_and_plays():
    from manim import Group
    from manimate.viz.core import dry_run, fade_in_whole
    s = PositionalEncodingSum(["a", "b", "c"], 8)
    assert len(s.animate_add().animations) == 3

    def script(ctx):
        g = PositionalEncodingSum(["a", "b", "c"], 8)
        ctx.fit(g)
        fade_in_whole(ctx, g)
        ctx.play(g.animate_add(), run_time=3.0)

    assert dry_run(script, 6.0).on_stage()


@pytest.mark.parametrize("kw, msg", [
    (dict(tokens=[]), "between 1"),
    (dict(tokens=["a"], d_model=7), "even"),
    (dict(tokens=["a"], d_model=10), "even"),
    (dict(tokens=["a", "b"], embeddings=np.ones((2, 3))), "shape"),
])
def test_sum_bad_input(kw, msg):
    with pytest.raises(ValueError, match=msg):
        PositionalEncodingSum(**kw)


# ---- PEHeatmap -------------------------------------------------------------------------------------------------------
from manimate.viz.ml_advanced.positional import PEHeatmap


def test_heatmap_data_is_the_pe_matrix_and_cells_are_coloured_from_it():
    h = PEHeatmap(10, 8)
    assert np.allclose(h.data.pe, m.sinusoidal_pe(10, 8)) and h.grid.values.shape == (10, 8)
    assert h.grid.cells[0][0].get_fill_color().to_hex() != h.grid.cells[0][1].get_fill_color().to_hex()  # sin(0)=0 vs cos(0)=1


def test_markers_hidden_until_set_then_land_on_the_right_cell():
    h = PEHeatmap(10, 8)
    assert h.col_marker.get_stroke_opacity() == 0.0
    h.set_dim(3).set_position(4)
    cell = h.grid.cells[4][3]
    assert np.allclose(h.col_marker.get_center()[0], cell.get_center()[0], atol=1e-6)
    assert np.allclose(h.row_marker.get_center()[1], cell.get_center()[1], atol=1e-6)
    assert h.data.dim == 3 and h.data.position == 4


def test_wave_of_maps_columns_to_wave_rows():
    assert PEHeatmap.wave_of(0) == (0, "sin") and PEHeatmap.wave_of(5) == (2, "cos")


def test_big_heatmap_becomes_an_image_but_markers_still_work():
    h = PEHeatmap(40, 32)
    assert h.grid.as_image and h.width <= 12.4 and h.height <= 5.8
    h.set_dim(31).set_position(39)
    assert h.col_marker.get_stroke_opacity() == 1.0


def test_marker_animations_play_and_bind_follows_tracker():
    from manim import Animation
    from manimate.viz.core import dry_run
    h = PEHeatmap(8, 8)
    t = ValueTracker(2)
    h.bind(t)
    h.update(0)
    assert h.data.position == 2

    def script(ctx):
        g = PEHeatmap(8, 8)
        ctx.fit(g)
        ctx.add(g)
        ctx.play(g.animate_dim(5), run_time=1.0)
        ctx.play(g.animate_position(3), run_time=1.0)

    dry_run(script, 4.0)


@pytest.mark.parametrize("kw, msg", [(dict(n_positions=1, d_model=8), "between 2"), (dict(n_positions=8, d_model=7), "even"), (dict(n_positions=8, d_model=66), "even")])
def test_heatmap_bad_input(kw, msg):
    with pytest.raises(ValueError, match=msg):
        PEHeatmap(**kw)


def test_marker_index_errors():
    with pytest.raises(ValueError, match="outside"):
        PEHeatmap(8, 8).set_dim(8)
    with pytest.raises(ValueError, match="outside"):
        PEHeatmap(8, 8).animate_position(-1)
