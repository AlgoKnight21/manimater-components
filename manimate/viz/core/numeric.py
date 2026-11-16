"""Small NumPy kernels shared by the domains. No Manim imports: everything here is testable on its own."""
from __future__ import annotations

from typing import Callable

import numpy as np

MAX_STEPS = 20_000  # integrators stop here instead of running away on a bad step size


def rng(seed: int = 7) -> np.random.Generator:
    """The one way viz makes random numbers, so two renders of the same storyboard are identical."""
    return np.random.default_rng(int(seed))


def softmax(x, axis: int = -1, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax. ``-inf`` entries (masked positions) get exactly 0 weight."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    z = np.asarray(x, dtype=float) / temperature
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def layernorm(x, axis: int = -1, eps: float = 1e-5) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    mu = a.mean(axis=axis, keepdims=True)
    var = a.var(axis=axis, keepdims=True)
    return (a - mu) / np.sqrt(var + eps)


def gaussian(x, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    """Normal probability density."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    z = (np.asarray(x, dtype=float) - mu) / sigma
    return np.exp(-0.5 * z * z) / (sigma * np.sqrt(2 * np.pi))


def normalize(x, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Rescale to [lo, hi]. A constant input maps to the midpoint instead of dividing by zero."""
    a = np.asarray(x, dtype=float)
    span = a.max() - a.min() if a.size else 0.0
    if span == 0:
        return np.full_like(a, (lo + hi) / 2)
    return lo + (a - a.min()) / span * (hi - lo)


def safe_div(a, b, default: float = 0.0):
    """Elementwise ``a / b`` that returns ``default`` where ``b`` is zero or the result is not finite."""
    a_, b_ = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(b_ == 0, default, a_ / np.where(b_ == 0, 1.0, b_))
    out = np.where(np.isfinite(out), out, default)
    return float(out) if out.ndim == 0 else out


def moving_average(x, window: int) -> np.ndarray:
    """Trailing average: element ``i`` is the mean of the last ``window`` samples (fewer at the start). Same length as ``x``."""
    a = np.asarray(x, dtype=float)
    w = max(1, int(window))
    if a.size == 0:
        return a
    csum = np.cumsum(np.insert(a, 0, 0.0))
    idx = np.arange(1, a.size + 1)
    lo = np.maximum(idx - w, 0)
    return (csum[idx] - csum[lo]) / (idx - lo)


def _integrate(step: Callable, f: Callable, y0, t_end: float, dt: float, t0: float):
    if dt <= 0:
        raise ValueError("dt must be positive")
    n = int(np.ceil((t_end - t0) / dt))
    if n > MAX_STEPS:
        raise ValueError(f"too many steps ({n}); raise dt or shorten the interval (max {MAX_STEPS})")
    y = np.atleast_1d(np.asarray(y0, dtype=float))
    ts = t0 + dt * np.arange(n + 1)
    ys = np.empty((n + 1, y.size))
    ys[0] = y
    for i in range(n):
        y = step(f, ts[i], y, dt)
        ys[i + 1] = y
    return ts, ys


def _euler_step(f, t, y, dt):
    return y + dt * np.asarray(f(t, y), dtype=float)


def _rk4_step(f, t, y, dt):
    k1 = np.asarray(f(t, y), dtype=float)
    k2 = np.asarray(f(t + dt / 2, y + dt / 2 * k1), dtype=float)
    k3 = np.asarray(f(t + dt / 2, y + dt / 2 * k2), dtype=float)
    k4 = np.asarray(f(t + dt, y + dt * k3), dtype=float)
    return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def euler(f: Callable, y0, t_end: float, dt: float = 0.01, t0: float = 0.0):
    """Integrate ``dy/dt = f(t, y)``. Returns ``(times, states)`` with states of shape ``(steps + 1, dim)``."""
    return _integrate(_euler_step, f, y0, t_end, dt, t0)


def rk4(f: Callable, y0, t_end: float, dt: float = 0.01, t0: float = 0.0):
    """Fourth-order Runge-Kutta for ``dy/dt = f(t, y)``. Returns ``(times, states)``."""
    return _integrate(_rk4_step, f, y0, t_end, dt, t0)
