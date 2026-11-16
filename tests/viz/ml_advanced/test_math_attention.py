import numpy as np
import pytest

from manimate.viz.ml_advanced._math import (attention_output, attention_scores, attention_weights, kv_cache_steps,
                                            pe_components, qkv_project, sinusoidal_pe)

RNG = np.random.default_rng(0)


def test_qkv_shapes_and_weight_check():
    X, W = RNG.normal(size=(4, 8)), RNG.normal(size=(8, 3))
    Q, K, V = qkv_project(X, W, W, W)
    assert Q.shape == K.shape == V.shape == (4, 3)
    with pytest.raises(ValueError, match="d_model"):
        qkv_project(X, W[:5], W, W)


def test_softmax_rows_sum_to_one_and_causal_mask_zeroes_the_future():
    Q, K = RNG.normal(size=(5, 4)), RNG.normal(size=(5, 4))
    assert np.allclose(attention_weights(Q, K).sum(axis=1), 1.0)
    W = attention_weights(Q, K, causal=True)
    assert np.allclose(W.sum(axis=1), 1.0) and np.all(np.triu(W, k=1) == 0)
    assert np.isinf(attention_scores(Q, K, causal=True)[0, 1])


def test_scaling_divides_by_sqrt_dk_and_output_is_a_weighted_sum_of_values():
    Q = K = np.eye(4)
    assert attention_scores(Q, K, scale=True)[0, 0] == pytest.approx(0.5)
    V = RNG.normal(size=(4, 2))
    assert np.allclose(attention_output(Q, K, V), attention_weights(Q, K) @ V)


def test_positional_encoding_first_row_values_and_bounds():
    pe = sinusoidal_pe(10, 8)
    assert np.allclose(pe[0], [0, 1, 0, 1, 0, 1, 0, 1]) and np.all(np.abs(pe) <= 1)
    with pytest.raises(ValueError):
        sinusoidal_pe(0, 8)


def test_pe_components_rebuild_the_matrix_row_and_frequencies_fall():
    comps = pe_components(5, 8)
    assert np.allclose([c.value for c in comps], sinusoidal_pe(6, 8)[5])
    omegas = [c.omega for c in comps[::2]]
    assert omegas == sorted(omegas, reverse=True) and omegas[0] == 1.0


def test_kv_cache_work_with_and_without_a_cache():
    steps = kv_cache_steps(6)
    assert sum(s.work_with_cache for s in steps) == 6
    assert sum(s.work_without_cache for s in steps) == 6 * 7 // 2
    assert steps[3].attended == (0, 1, 2, 3) and steps[3].cached == (0, 1, 2)


def test_sliding_window_evicts_the_oldest_token():
    steps = kv_cache_steps(6, window=3)
    assert steps[1].evicted is None and steps[3].evicted == 0 and steps[5].attended == (3, 4, 5)
    assert all(len(s.attended) <= 3 for s in steps)
    with pytest.raises(ValueError):
        kv_cache_steps(3, window=0)
