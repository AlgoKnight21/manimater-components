"""Pure maths for ``viz.signals`` control pictures (README section 8.1). No Manim imports: everything here is plain Python,
NumPy and SciPy, so the numbers can be tested exactly and read by the pictures through their ``.data`` properties.

Conventions: polynomials are coefficient arrays with the highest power first (``[1, 3, 2]`` is ``s^2 + 3s + 2``).
Poles and zeros are complex numbers; a transfer function built from them must have real coefficients, so complex roots
come in conjugate pairs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy import signal as _signal
from scipy.optimize import linear_sum_assignment

ROOT_TOL = 1e-9  # a root with |real part| below this counts as "on the imaginary axis"
SETTLE_BAND = 0.02  # the usual 2 % settling band


# --------------------------------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------------------------------
def complex_list(values) -> list[complex]:
    """Roots as complex numbers. Accepts complex numbers, real numbers or ``(re, im)`` pairs."""
    out: list[complex] = []
    for v in values:
        z = complex(v[0], v[1]) if isinstance(v, (tuple, list, np.ndarray)) and len(v) == 2 else complex(v)
        if not (math.isfinite(z.real) and math.isfinite(z.imag)):
            raise ValueError("poles and zeros must be finite numbers")
        out.append(z)
    return out


def flat_to_complex(flat: Sequence[float]) -> list[complex]:
    """``[re, im, re, im, ...]`` (how the registry passes complex numbers in a ``number[]``) to a list of complex."""
    vals = list(flat)
    if len(vals) % 2:
        raise ValueError("give poles and zeros as pairs: re, im, re, im, ...")
    return complex_list(list(zip(vals[0::2], vals[1::2])))


def ceil_to(x: float, step: float) -> float:
    """Round ``x`` up to a multiple of ``step`` (used to pick tidy axis limits)."""
    return math.ceil(x / step - 1e-9) * step


def crossing(x, y, level: float) -> float | None:
    """First ``x`` where ``y`` crosses ``level`` (linear interpolation between samples), or ``None``."""
    x_, s = np.asarray(x, float), np.asarray(y, float) - level
    for i in np.nonzero(s[:-1] * s[1:] <= 0)[0]:
        if s[i] == s[i + 1]:
            continue
        return float(x_[i] + (x_[i + 1] - x_[i]) * s[i] / (s[i] - s[i + 1]))
    return None


# --------------------------------------------------------------------------------------------------------------------
# Transfer functions
# --------------------------------------------------------------------------------------------------------------------
def _trim(c) -> np.ndarray:
    a = np.atleast_1d(np.asarray(c, dtype=float))
    if a.ndim != 1 or a.size == 0 or not np.all(np.isfinite(a)):
        raise ValueError("polynomial coefficients must be a non-empty 1-D list of finite numbers")
    nz = np.nonzero(a)[0]
    return a[nz[0]:] if len(nz) else np.array([0.0])


@dataclass(frozen=True)
class TF:
    """A transfer function ``num(s) / den(s)`` with real coefficients."""
    num: np.ndarray
    den: np.ndarray

    def __post_init__(self) -> None:
        if self.den[0] == 0:
            raise ValueError("the denominator cannot be zero")

    def __call__(self, s):
        z = np.asarray(s, dtype=complex)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.polyval(self.num, z) / np.polyval(self.den, z)

    @property
    def poles(self) -> np.ndarray:
        return np.roots(self.den)

    @property
    def zeros(self) -> np.ndarray:
        return np.roots(self.num) if len(self.num) > 1 else np.array([], dtype=complex)

    @property
    def stable(self) -> bool:
        return bool(np.all(self.poles.real < -ROOT_TOL))

    @property
    def proper(self) -> bool:
        return len(self.num) <= len(self.den)


def tf_from_coeffs(num, den) -> TF:
    return TF(_trim(num), _trim(den))


def tf_from_pz(poles=(), zeros=(), gain: float = 1.0) -> TF:
    """``gain * prod(s - z) / prod(s - p)``. Complex roots must come in conjugate pairs."""
    p, z = complex_list(poles), complex_list(zeros)
    num = gain * (np.poly(z) if z else np.array([1.0]))
    den = np.poly(p) if p else np.array([1.0])
    for name, c in (("numerator", num), ("denominator", den)):
        if np.any(np.abs(np.imag(c)) > 1e-9 * max(1.0, float(np.max(np.abs(c))))):
            raise ValueError(f"complex poles and zeros must come in conjugate pairs (the {name} is not real)")
    return TF(_trim(np.real(num)), _trim(np.real(den)))


def unity_feedback(tf: TF, k: float = 1.0) -> TF:
    """Closed loop ``k L / (1 + k L)`` of the open loop ``tf = L``."""
    return TF(_trim(k * tf.num), _trim(np.polyadd(tf.den, k * tf.num)))


def auto_w_range(tf: TF) -> tuple[float, float]:
    """A frequency window (rad/s) that is one decade wider than the poles and zeros on each side, between 3 and 8 decades."""
    mags = [abs(r) for r in np.concatenate([tf.poles, tf.zeros]) if abs(r) > ROOT_TOL]
    if not mags:
        return 0.1, 100.0
    lo, hi = math.floor(math.log10(min(mags))) - 1, math.ceil(math.log10(max(mags))) + 1
    while hi - lo < 3:
        lo, hi = lo - 1, hi + 1
    while hi - lo > 8:
        lo, hi = lo + 1, hi - 1
    return 10.0 ** lo, 10.0 ** hi


def _fmt(x: float) -> str:
    return f"{round(float(x), 3):g}"


_SUP = {1: "", 2: "²", 3: "³"}


def poly_str(coeffs, var: str = "s") -> str:
    """Readable polynomial text, e.g. ``[1, 0.6, 1]`` -> ``s² + 0.6s + 1``."""
    c = _trim(coeffs)
    deg, terms = len(c) - 1, []
    for i, a in enumerate(c):
        p = deg - i
        if abs(a) < 1e-12:
            continue
        mag = abs(a)
        if p == 0:
            body = _fmt(mag)
        else:
            body = ("" if abs(mag - 1) < 1e-12 else _fmt(mag)) + var + _SUP.get(p, f"^{p}")
        terms.append(("-" if a < 0 else "+", body))
    if not terms:
        return "0"
    s = ("-" if terms[0][0] == "-" else "") + terms[0][1]
    for sign, body in terms[1:]:
        s += f" {sign} {body}"
    return s


# --------------------------------------------------------------------------------------------------------------------
# Frequency response: Bode and Nyquist
# --------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class BodeData:
    w: np.ndarray
    mag_db: np.ndarray
    phase_deg: np.ndarray
    w_gc: float | None  # gain crossover: |L| = 0 dB
    phase_margin_deg: float | None  # 180 + phase at the gain crossover
    w_pc: float | None  # phase crossover: phase = -180 degrees
    gain_margin_db: float | None  # -|L| in dB at the phase crossover


def bode_data(tf: TF, w=None, *, w_range: tuple[float, float] | None = None, n: int = 400) -> BodeData:
    """Magnitude (dB), unwrapped phase (degrees) and the gain and phase margins of ``L(jw)``."""
    if w is None:
        lo, hi = w_range or auto_w_range(tf)
        if not 0 < lo < hi:
            raise ValueError("w_range must be (low, high) with 0 < low < high")
        w = np.logspace(math.log10(lo), math.log10(hi), n)
    w = np.asarray(w, dtype=float)
    h = tf(1j * w)
    mag = 20 * np.log10(np.maximum(np.abs(h), 1e-12))
    phase = np.degrees(np.unwrap(np.angle(h)))
    lw = np.log10(w)
    gc, pc = crossing(lw, mag, 0.0), crossing(lw, phase, -180.0)
    pm = None if gc is None else float(180.0 + np.interp(gc, lw, phase))
    gm = None if pc is None else float(-np.interp(pc, lw, mag))
    return BodeData(w, mag, phase, None if gc is None else 10.0 ** gc, pm, None if pc is None else 10.0 ** pc, gm)


def bode_asymptotes(poles, zeros, gain: float, w) -> np.ndarray:
    """Straight-line magnitude approximation (dB): each root away from the origin bends the line by 20 dB/decade at its corner."""
    lw = np.log10(np.asarray(w, dtype=float))
    db, k0, origin = np.zeros_like(lw), abs(gain), 0
    for roots, sign in ((complex_list(zeros), 1), (complex_list(poles), -1)):
        for r in roots:
            m = abs(r)
            if m < ROOT_TOL:
                origin += sign
            else:
                k0 *= m ** sign
                db += sign * 20 * np.maximum(lw - math.log10(m), 0.0)
    return db + 20 * math.log10(max(k0, 1e-300)) + 20 * origin * lw


@dataclass(frozen=True)
class NyquistData:
    w: np.ndarray
    points: np.ndarray  # L(jw) for w > 0 (the mirror image for w < 0 is the complex conjugate)
    open_loop_rhp: int  # P: right-half-plane poles of L
    closed_loop_rhp: int  # Z: right-half-plane poles of 1 + L
    clockwise_encirclements: int  # N = Z - P: how many times the contour winds clockwise round -1
    stable: bool  # the closed loop is stable when Z == 0
    min_distance: float  # closest approach to the critical point -1


def nyquist_data(tf: TF, w=None, *, w_range: tuple[float, float] | None = None, n: int = 600) -> NyquistData:
    """The Nyquist contour of ``L(jw)`` and the encirclement count about -1.

    The count comes from the pole locations (``N = Z - P``), which stays correct for integrators where the drawn
    contour has to be indented round the origin.
    """
    if w is None:
        lo, hi = w_range or auto_w_range(tf)
        w = np.logspace(math.log10(lo), math.log10(hi), n)
    w = np.asarray(w, dtype=float)
    pts = tf(1j * w)
    p = int(np.sum(tf.poles.real > ROOT_TOL))
    z = int(np.sum(np.roots(np.polyadd(tf.den, tf.num)).real > ROOT_TOL))
    return NyquistData(w, pts, p, z, z - p, z == 0, float(np.min(np.abs(pts + 1.0))))


# --------------------------------------------------------------------------------------------------------------------
# Root locus
# --------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LocusData:
    k: np.ndarray
    roots: np.ndarray  # shape (len(k), n_branches): the closed-loop poles, ordered so each column is one continuous branch
    critical_gain: float | None  # smallest K that puts a closed-loop pole in the right half plane


def locus_k_values(k_max: float, n: int = 240) -> np.ndarray:
    """Gains from 0 to ``k_max``, packed tightly near 0 where the branches leave the open-loop poles."""
    if k_max <= 0:
        raise ValueError("k_max must be positive")
    return np.concatenate([[0.0], np.geomspace(k_max * 1e-4, k_max, n - 1)])


def root_locus_data(num, den, k_values=None, *, k_max: float = 10.0) -> LocusData:
    """Closed-loop poles of ``1 + K num(s)/den(s) = 0`` as the gain K grows."""
    num_, den_ = _trim(num), _trim(den)
    if len(num_) >= len(den_):
        raise ValueError("the open loop must be strictly proper (fewer zeros than poles)")
    k = locus_k_values(k_max) if k_values is None else np.asarray(k_values, dtype=float)
    rows = [np.roots(np.polyadd(den_, k[0] * num_))]
    for kk in k[1:]:
        new = np.roots(np.polyadd(den_, kk * num_))
        _, col = linear_sum_assignment(np.abs(rows[-1][:, None] - new[None, :]))
        rows.append(new[col])
    roots = np.array(rows)
    worst = roots.real.max(axis=1)
    crit = None
    bad = np.nonzero(worst > ROOT_TOL)[0]
    if len(bad) and bad[0] > 0:
        i = bad[0]
        a, b = worst[i - 1], worst[i]
        crit = float(k[i - 1] + (k[i] - k[i - 1]) * (0.0 - a) / (b - a)) if b != a else float(k[i])
        crit = max(crit, float(k[i - 1]))
    return LocusData(k, roots, crit)


# --------------------------------------------------------------------------------------------------------------------
# Time response
# --------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ResponseData:
    t: np.ndarray
    y: np.ndarray
    kind: str
    stable: bool
    final_value: float | None
    rise_time: float | None  # 10 % to 90 % of the final value
    overshoot_pct: float | None
    settling_time: float | None  # last time outside the 2 % band; None if it never settles in the window
    peak_time: float | None


def auto_horizon(tf: TF) -> float:
    """A time window long enough to see the response settle (or clearly run away), between 1 s and 400 s."""
    re = -tf.poles.real
    decaying = re[re > ROOT_TOL]
    t = 5.0 / decaying.min() if tf.stable and len(decaying) else (8.0 / max(abs(re.min()), 0.2) if len(re) else 10.0)
    return float(min(max(t, 1.0), 400.0))


def _step_metrics(t: np.ndarray, y: np.ndarray, target: float):
    """``(rise, overshoot %, settling, peak time)`` of a step response heading for ``target``; ``None`` where undefined."""
    if not np.all(np.isfinite(y)) or not np.isfinite(target) or abs(target) < 1e-12:
        return None, None, None, None
    yn = y / target
    peak = int(np.argmax(yn))
    t10, t90 = crossing(t, yn, 0.1), crossing(t, yn, 0.9)
    outside = np.nonzero(np.abs(yn - 1.0) > SETTLE_BAND)[0]
    if len(outside) == 0:
        settle: float | None = 0.0
    else:
        settle = float(t[outside[-1] + 1]) if outside[-1] + 1 < len(t) else None
    return (None if t10 is None or t90 is None else t90 - t10), max(0.0, float(yn[peak] - 1.0)) * 100.0, settle, float(t[peak])


def step_response(tf: TF, t=None, kind: str = "step", *, n: int = 400) -> ResponseData:
    """Step or impulse response with rise time, overshoot and settling time (step only)."""
    if kind not in ("step", "impulse"):
        raise ValueError("kind must be 'step' or 'impulse'")
    if not tf.proper:
        raise ValueError("the transfer function must be proper (no more zeros than poles)")
    t = np.linspace(0.0, auto_horizon(tf), n) if t is None else np.asarray(t, dtype=float)
    system = _signal.lti(tf.num, tf.den)
    t, y = system.step(T=t) if kind == "step" else system.impulse(T=t)
    y = np.nan_to_num(np.asarray(y, dtype=float), nan=0.0, posinf=1e12, neginf=-1e12)
    stable = tf.stable
    if kind == "impulse" or not stable:
        return ResponseData(t, y, kind, stable, None, None, None, None, None)
    final = float(np.real(tf(0.0)))
    rise, over, settle, peak = _step_metrics(t, y, final)
    return ResponseData(t, y, kind, stable, final, rise, over, settle, peak)


# --------------------------------------------------------------------------------------------------------------------
# PID control
# --------------------------------------------------------------------------------------------------------------------
PLANTS: dict[str, TF] = {
    "first_order": tf_from_coeffs([1.0], [1.0, 1.0]),
    "second_order": tf_from_coeffs([1.0], [1.0, 0.6, 1.0]),
    "integrator": tf_from_coeffs([1.0], [1.0, 1.0, 0.0]),
}


def pid_closed_loop_poles(kp: float, ki: float, kd: float, plant: TF) -> np.ndarray:
    """Poles of the loop ``C(s) G(s)`` closed with unity feedback, ``C = kp + ki/s + kd s``."""
    if ki == 0:  # C = kp + kd s: the integrator's s cancels, so do not leave a spurious root at the origin
        return np.roots(np.polyadd(plant.den, np.polymul([kd, kp], plant.num)))
    return np.roots(np.polyadd(np.polymul([1.0, 0.0], plant.den), np.polymul([kd, kp, ki], plant.num)))


@dataclass(frozen=True)
class PIDData:
    t: np.ndarray
    y: np.ndarray
    u: np.ndarray
    p: np.ndarray  # the three contributions to u
    i: np.ndarray
    d: np.ndarray
    error: np.ndarray
    setpoint: float
    stable: bool
    overshoot_pct: float | None
    settling_time: float | None
    steady_state_error: float


def pid_simulate(kp: float, ki: float, kd: float, plant: TF | str = "second_order", setpoint: float = 1.0, t=None, *,
                 t_end: float = 12.0, n: int = 240) -> PIDData:
    """Unity-feedback loop with a PID controller on a strictly proper plant, stepped with an exact (zero-order-hold) plant model.

    The derivative acts on the measurement (not the error), so a setpoint step does not kick the output. The integral is a
    running sum with no anti-windup. States are clipped at 1e9 so an unstable loop stays plottable.
    """
    g = PLANTS[plant] if isinstance(plant, str) else plant
    if min(kp, ki, kd) < 0:
        raise ValueError("kp, ki and kd must not be negative")
    if len(g.num) >= len(g.den):
        raise ValueError("the plant must be strictly proper and have at least one pole")
    t = np.linspace(0.0, t_end, n) if t is None else np.asarray(t, dtype=float)
    dt = float(t[1] - t[0])
    a, b, c, d0 = _signal.tf2ss(g.num, g.den)
    ad, bd, cd, _, _ = _signal.cont2discrete((a, b, c, d0), dt, method="zoh")
    x, integ, y_prev = np.zeros(ad.shape[0]), 0.0, 0.0
    ys, us, ps, is_, ds, es = (np.zeros(len(t)) for _ in range(6))
    for k in range(len(t)):
        y = float((cd @ x)[0])
        e = setpoint - y
        integ += e * dt
        ps[k], is_[k] = kp * e, ki * integ
        ds[k] = 0.0 if k == 0 else -kd * (y - y_prev) / dt
        us[k], ys[k], es[k], y_prev = ps[k] + is_[k] + ds[k], y, e, y
        x = np.clip(ad @ x + bd[:, 0] * us[k], -1e9, 1e9)
    stable = bool(np.all(pid_closed_loop_poles(kp, ki, kd, g).real < -ROOT_TOL))
    _, over, settle, _ = _step_metrics(t, ys, setpoint)
    return PIDData(t, ys, us, ps, is_, ds, es, setpoint, stable, over, settle, float(setpoint - ys[-1]))


def pid_terms(error_fn: Callable[[float], float], kp: float, ki: float, kd: float, t_end: float, *, n: int = 600):
    """``(t, p, i, d)``: the three PID contributions to the control signal for a given error signal ``e(t)`` (0 before t = 0)."""
    if t_end <= 0:
        raise ValueError("t_end must be positive")
    t = np.linspace(0.0, t_end, n)
    e = np.array([float(error_fn(x)) for x in t])
    integral = np.concatenate([[0.0], np.cumsum(0.5 * (e[1:] + e[:-1]) * np.diff(t))])
    return t, kp * e, ki * integral, kd * np.gradient(e, t)


# --------------------------------------------------------------------------------------------------------------------
# State space
# --------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class StateSpace:
    """``x' = A x + B u``, ``y = C x + D u``."""
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray

    @property
    def n(self) -> int:
        return self.A.shape[0]

    @property
    def inputs(self) -> int:
        return self.B.shape[1]

    @property
    def outputs(self) -> int:
        return self.C.shape[0]

    @property
    def eigenvalues(self) -> np.ndarray:
        return np.linalg.eigvals(self.A)

    @property
    def stable(self) -> bool:
        return bool(np.all(self.eigenvalues.real < -ROOT_TOL))


def state_space(A, B, C, D=None) -> StateSpace:
    """Validate shapes and build a ``StateSpace`` (``D`` defaults to zero)."""
    a, b, c = (np.atleast_2d(np.asarray(m, dtype=float)) for m in (A, B, C))
    n = a.shape[0]
    if a.shape != (n, n):
        raise ValueError("A must be square")
    if b.shape[0] != n or c.shape[1] != n:
        raise ValueError(f"B needs {n} rows and C needs {n} columns to match A")
    d = np.zeros((c.shape[0], b.shape[1])) if D is None else np.atleast_2d(np.asarray(D, dtype=float))
    if d.shape != (c.shape[0], b.shape[1]):
        raise ValueError(f"D must be {c.shape[0]} x {b.shape[1]}")
    if not all(np.all(np.isfinite(m)) for m in (a, b, c, d)):
        raise ValueError("matrix entries must be finite numbers")
    return StateSpace(a, b, c, d)
