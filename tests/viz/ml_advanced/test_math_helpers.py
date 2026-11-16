"""Tests for the helpers added for the QKV / positional / KV-cache / head pictures (the core maths is tested in test_math_attention.py)."""
import numpy as np
import pytest

from manimate.viz.ml_advanced import _math as m


def _head(n=5, d=6, dh=4, causal=False):
    X = m.seeded_embeddings(n, d)
    return m.attention_head(X, *(m.init_weights((d, dh), seed=s) for s in (1, 2, 3)), causal=causal)


def test_seeded_helpers_are_deterministic_and_validated():
    assert np.allclose(m.seeded_embeddings(4, 6, seed=3), m.seeded_embeddings(4, 6, seed=3))
    assert np.allclose(m.init_weights((6, 4), seed=3), m.init_weights((6, 4), seed=3))
    with pytest.raises(ValueError, match="between 1 and"):
        m.seeded_embeddings(m.MAX_TOKENS + 1, 4)
    with pytest.raises(ValueError, match="at least 1"):
        m.seeded_embeddings(3, 0)


def test_attention_head_keeps_every_intermediate_consistent():
    d = _head(causal=True)
    assert np.allclose(d.weights.sum(axis=1), 1.0) and np.all(d.weights[np.triu_indices(5, k=1)] == 0.0)
    assert np.allclose(d.output[2], sum(d.weights[2, j] * d.V[j] for j in range(5)))
    assert np.allclose(d.scores[np.tril_indices(5)], (d.Q @ d.K.T / 2.0)[np.tril_indices(5)])


def test_pe_frequencies_match_the_matrix_and_wavelengths_grow_geometrically():
    freqs, wl = m.pe_frequencies(8)
    assert np.isclose(wl[0], 2 * np.pi)
    pe = m.sinusoidal_pe(6, 8)
    assert np.allclose(pe[5, 0::2], np.sin(5 * freqs)) and np.allclose(pe[5, 1::2], np.cos(5 * freqs))
    r = wl[1:] / wl[:-1]
    assert np.allclose(r, r[0]) and r[0] > 1


def test_pe_frequencies_reject_odd_widths_and_bad_base():
    with pytest.raises(ValueError, match="even"):
        m.pe_frequencies(7)
    with pytest.raises(ValueError, match="base"):
        m.pe_frequencies(8, base=1.0)


def test_kv_cache_counts_the_saving_the_timeline_draws():
    steps = m.kv_cache_steps(6)
    assert sum(s.work_with_cache for s in steps) == 6 and sum(s.work_without_cache for s in steps) == 21
    w = m.kv_cache_steps(6, window=3)
    assert [s.evicted for s in w] == [None, None, None, 0, 1, 2] and w[-1].attended == (3, 4, 5)
