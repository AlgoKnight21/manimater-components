import numpy as np
import pytest

from manimate.viz.core import numeric as nm


def test_rng_is_deterministic_per_seed():
    assert nm.rng(3).random() == nm.rng(3).random() != nm.rng(4).random()


def test_softmax_rows_sum_to_one_and_survive_huge_values():
    p = nm.softmax([[1000.0, 1001.0, 999.0], [0, 0, 0]])
    assert np.allclose(p.sum(axis=1), 1) and np.all(np.isfinite(p))
    assert np.allclose(p[1], 1 / 3)


def test_softmax_masked_positions_get_exactly_zero():
    p = nm.softmax([1.0, 2.0, -np.inf])
    assert p[2] == 0 and p.sum() == pytest.approx(1)


def test_softmax_temperature_sharpens_and_flattens():
    x = [1.0, 2.0, 3.0]
    assert nm.softmax(x, temperature=0.1).max() > nm.softmax(x).max() > nm.softmax(x, temperature=10).max()
    with pytest.raises(ValueError):
        nm.softmax(x, temperature=0)


def test_layernorm_zero_mean_unit_variance():
    y = nm.layernorm(np.random.default_rng(0).normal(5, 3, size=(4, 16)))
    assert np.allclose(y.mean(axis=1), 0, atol=1e-9) and np.allclose(y.std(axis=1), 1, atol=1e-3)


def test_gaussian_integrates_to_one_and_peaks_at_mean():
    x = np.linspace(-8, 8, 4001)
    g = nm.gaussian(x, 1.0, 0.7)
    assert np.trapezoid(g, x) == pytest.approx(1.0, abs=1e-4) and x[g.argmax()] == pytest.approx(1.0, abs=0.01)
    with pytest.raises(ValueError):
        nm.gaussian(0, sigma=0)


def test_normalize_range_and_constant_input():
    assert nm.normalize([2, 4, 6]).tolist() == [0.0, 0.5, 1.0]
    assert nm.normalize([3, 3, 3], 0, 10).tolist() == [5.0, 5.0, 5.0]
    assert nm.normalize([]).size == 0


def test_safe_div_handles_zero_and_nonfinite():
    assert nm.safe_div(1, 0, default=-1) == -1
    assert nm.safe_div([1, 2, 3], [1, 0, 3]).tolist() == [1.0, 0.0, 1.0]
    assert nm.safe_div(np.inf, 2, default=7) == 7


def test_moving_average_trailing_window():
    assert nm.moving_average([1, 2, 3, 4], 2).tolist() == [1.0, 1.5, 2.5, 3.5]
    assert nm.moving_average([1, 2, 3], 1).tolist() == [1, 2, 3] and nm.moving_average([5, 5, 5], 99).tolist() == [5, 5, 5]
    assert nm.moving_average([], 3).size == 0


def test_rk4_matches_exponential_decay_much_better_than_euler():
    f = lambda t, y: -y
    _, r = nm.rk4(f, [1.0], 2.0, 0.1)
    _, e = nm.euler(f, [1.0], 2.0, 0.1)
    exact = np.exp(-2.0)
    assert abs(r[-1, 0] - exact) < 1e-5 < abs(e[-1, 0] - exact)


def test_rk4_harmonic_oscillator_conserves_energy():
    _, y = nm.rk4(lambda t, s: np.array([s[1], -s[0]]), [1.0, 0.0], 20.0, 0.01)
    energy = y[:, 0] ** 2 + y[:, 1] ** 2
    assert np.allclose(energy, 1.0, atol=1e-6)


def test_integrators_guard_against_bad_step_sizes():
    with pytest.raises(ValueError, match="dt"):
        nm.rk4(lambda t, y: y, [1.0], 1.0, 0.0)
    with pytest.raises(ValueError, match="too many steps"):
        nm.rk4(lambda t, y: y, [1.0], 1e6, 0.01)
