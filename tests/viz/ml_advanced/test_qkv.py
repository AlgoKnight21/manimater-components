import numpy as np
import pytest
from manim import Animation, ValueTracker

from manimate.viz.ml_advanced import _math as m
from manimate.viz.ml_advanced.qkv import MAX_QKV_D_MODEL, MAX_QKV_TOKENS, QKVProjection


def _x(n=4, d=6):
    return m.seeded_embeddings(n, d)


def test_data_matches_the_pure_math():
    q = QKVProjection(_x(), d_head=3)
    d = q.data
    Q, K, V = m.qkv_project(d.X, d.Wq, d.Wk, d.Wv)
    assert np.allclose(d.Q, Q) and np.allclose(d.K, K) and np.allclose(d.V, V) and d.Q.shape == (4, 3)


def test_same_seed_gives_same_weights():
    assert np.allclose(QKVProjection(_x(), 3, seed=5).data.Wq, QKVProjection(_x(), 3, seed=5).data.Wq)


def test_grids_have_the_shapes_of_the_matrices():
    q = QKVProjection(_x(4, 6), d_head=3)
    assert q.x_grid.values.shape == (4, 6)
    assert q.w_grids["k"].values.shape == (6, 3) and q.out_grids["v"].values.shape == (4, 3)


def test_custom_weights_are_used():
    W = np.eye(6)[:, :2]
    q = QKVProjection(_x(), d_head=2, weights=(W, W, W))
    assert np.allclose(q.data.Q, _x()[:, :2])


def test_animate_project_returns_one_step_per_token_and_does_not_play():
    a = QKVProjection(_x(3), 2).animate_project(run_time=3.0)
    assert isinstance(a, Animation) and len(a.animations) == 3


def test_cursor_is_hidden_until_set_then_follows_the_tracker():
    q = QKVProjection(_x(), 2)
    assert all(r.get_stroke_opacity() == 0.0 for r in q.cursor)
    t = ValueTracker(2)
    q.bind(t)
    q.update(0)
    assert all(r.get_stroke_opacity() > 0 for r in q.cursor)
    y_before = q.cursor[0].get_center()[1]
    t.set_value(0)
    q.update(0)
    assert q.cursor[0].get_center()[1] > y_before  # token 0 is the top row


@pytest.mark.parametrize("X, dh, msg", [
    (np.ones((MAX_QKV_TOKENS + 1, 4)), 2, "at most"),
    (np.ones((3, MAX_QKV_D_MODEL + 1)), 2, "at most"),
    (np.ones(5), 2, "2-D"),
    (np.ones((3, 4)), 0, "d_head"),
])
def test_bad_input(X, dh, msg):
    with pytest.raises(ValueError, match=msg):
        QKVProjection(X, dh)


def test_set_token_out_of_range():
    with pytest.raises(ValueError, match="outside"):
        QKVProjection(_x(3), 2).set_token(3)


def test_largest_projection_fits_the_stage():
    from manimate.ctx import STAGE_H, STAGE_W
    q = QKVProjection(_x(MAX_QKV_TOKENS, MAX_QKV_D_MODEL), d_head=4)
    assert q.width <= STAGE_W and q.height <= STAGE_H
