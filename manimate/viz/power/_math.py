"""Pure maths behind viz.power. No Manim import. Per-unit / amplitude-invariant conventions are stated per function."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TOPOLOGIES = ("buck", "boost", "buck_boost")


# ---------------------------------------------------------------- DC-DC converters
def conversion_ratio(topology: str, duty: float) -> float:
    """Ideal Vout/Vin in continuous conduction. ``buck_boost`` is negative (inverting)."""
    if topology not in TOPOLOGIES:
        raise ValueError(f"topology must be one of {TOPOLOGIES}")
    if not 0.0 <= duty <= 1.0 or (topology != "buck" and duty >= 1.0):
        raise ValueError("duty must be in [0, 1) (buck also allows 1)")
    return {"buck": duty, "boost": 1 / (1 - duty) if duty < 1 else np.inf, "buck_boost": -duty / (1 - duty) if duty < 1 else -np.inf}[topology]


@dataclass(frozen=True)
class ConverterWaveforms:
    t: np.ndarray
    gate: np.ndarray       # 1 while the switch is on
    v_sw: np.ndarray       # voltage across the inductor's switched side (see docstring)
    i_l: np.ndarray
    v_out: np.ndarray
    mode: str              # "CCM" or "DCM"
    di_on: float           # inductor slope while the switch is on (A/s)
    di_off: float          # inductor slope while it is off (A/s)


def converter_waveforms(topology: str, vin: float, duty: float, L: float, C: float, R: float, fsw: float,
                        periods: int = 3, points_per_period: int = 200) -> ConverterWaveforms:
    """Time-domain simulation of an ideal buck, boost or buck-boost, started at its ideal steady state.

    ``v_sw`` is the switch-node voltage for buck/boost and the inductor drive for buck-boost. For buck-boost the
    simulation tracks |Vout| and reports ``v_out`` as negative. DCM appears automatically when the current reaches zero.
    """
    ratio = conversion_ratio(topology, duty)
    if min(L, C, R, fsw, vin) <= 0 or periods < 1 or points_per_period < 20:
        raise ValueError("vin, L, C, R, fsw must be positive; periods >= 1; points_per_period >= 20")
    T = 1.0 / fsw
    dt = T / points_per_period
    n = periods * points_per_period
    vo = abs(ratio) * vin
    il = vo / R if topology == "buck" else vo / (R * (1 - duty))
    ripple = ((vin - vo) if topology == "buck" else vin) * duty * T / L
    il = max(il - ripple / 2, 0.0)  # start at the ripple valley so the mean stays on the ideal value
    t = np.arange(n) * dt
    gate = ((t % T) / T < duty).astype(float)
    v_sw, i_l, v_o = np.empty(n), np.empty(n), np.empty(n)
    dcm = False
    for k in range(n):
        on = gate[k] > 0.5
        conducting = on or il > 0.0
        if topology == "buck":
            vs = vin if on else (0.0 if il > 0 else vo)
            dil, dvo = (vs - vo) / L, (il - vo / R) / C
        elif topology == "boost":
            vs = 0.0 if on else (vo if il > 0 else vin)
            dil, dvo = (vin - vs) / L, ((0.0 if on or il <= 0 else il) - vo / R) / C
        else:
            vs = vin if on else (-vo if il > 0 else 0.0)
            dil = vs / L if conducting else 0.0
            dvo = ((0.0 if on or il <= 0 else il) - vo / R) / C
        il_new = il + dil * dt
        if not on and il_new <= 0.0:
            il_new, dcm = 0.0, True
        v_sw[k], i_l[k], v_o[k] = vs, il, vo
        il, vo = il_new, vo + dvo * dt
    mean_vo = float(np.mean(v_o))
    slopes = {"buck": ((vin - mean_vo) / L, -mean_vo / L), "boost": (vin / L, (vin - mean_vo) / L), "buck_boost": (vin / L, -mean_vo / L)}[topology]
    return ConverterWaveforms(t, gate, v_sw, i_l, -v_o if topology == "buck_boost" else v_o, "DCM" if dcm else "CCM", *slopes)


# ---------------------------------------------------------------- PWM
def triangle(t, cycles: float = 1.0) -> np.ndarray:
    """Unit triangle carrier in [-1, 1], ``cycles`` per unit of t."""
    x = (np.asarray(t, dtype=float) * cycles) % 1.0
    return 4 * np.abs(x - 0.5) - 1


def pwm_compare(reference, carrier) -> np.ndarray:
    """Comparator: 1 while the reference is above the carrier, else 0."""
    r, c = np.asarray(reference, dtype=float), np.asarray(carrier, dtype=float)
    if r.shape != c.shape:
        raise ValueError("reference and carrier must have the same shape")
    return (r > c).astype(int)


@dataclass(frozen=True)
class HBridgeStates:
    t: np.ndarray         # one fundamental period, 0..1
    ref: np.ndarray
    carrier: np.ndarray
    q1: np.ndarray
    q2: np.ndarray
    q3: np.ndarray
    q4: np.ndarray
    v_out: np.ndarray     # in units of Vdc
    i_dir: np.ndarray     # +1 / -1: direction of the load current


def hbridge_states(modulation: str = "bipolar", ma: float = 0.8, fsw_ratio: int = 15, n_per_carrier: int = 64,
                   load_angle: float = 0.4) -> HBridgeStates:
    """Switch states for one fundamental period of a single-phase full bridge (Q1,Q2 left leg; Q3,Q4 right leg)."""
    if modulation not in ("bipolar", "unipolar") or not 0 < ma <= 1.2 or fsw_ratio < 3:
        raise ValueError("modulation is 'bipolar' or 'unipolar'; 0 < ma <= 1.2; fsw_ratio >= 3")
    n = fsw_ratio * n_per_carrier
    t = np.arange(n) / n
    ref, car = ma * np.sin(2 * np.pi * t), triangle(t, fsw_ratio)
    q1 = pwm_compare(ref, car)
    q3 = pwm_compare(-ref, car) if modulation == "unipolar" else 1 - q1
    v = q1 - q3
    return HBridgeStates(t, ref, car, q1, 1 - q1, q3, 1 - q3, v.astype(float), np.sign(np.sin(2 * np.pi * t - load_angle) + 1e-12))


@dataclass(frozen=True)
class ThreePhasePWM:
    t: np.ndarray
    refs: np.ndarray       # (3, n)
    carrier: np.ndarray
    poles: np.ndarray      # (3, n) pole voltages, +-vdc/2
    v_ab: np.ndarray       # line-to-line voltage
    filtered: np.ndarray   # (3, n) moving average over one carrier period: the fundamental


def spwm_three_phase(vdc: float, ma: float, fsw_ratio: int, periods: int = 1, n_per_carrier: int = 64) -> ThreePhasePWM:
    """Sinusoidal PWM for a three-phase bridge, with the filtered fundamental (moving average over one carrier period)."""
    if vdc <= 0 or not 0 < ma <= 1.15 or fsw_ratio < 3 or periods < 1:
        raise ValueError("vdc > 0, 0 < ma <= 1.15, fsw_ratio >= 3, periods >= 1")
    n = fsw_ratio * n_per_carrier * periods
    t = np.arange(n) / (fsw_ratio * n_per_carrier)
    refs = np.stack([ma * np.sin(2 * np.pi * t - k * 2 * np.pi / 3) for k in range(3)])
    car = triangle(t, fsw_ratio)
    poles = np.where(refs > car, 0.5 * vdc, -0.5 * vdc)
    kernel = np.ones(n_per_carrier) / n_per_carrier
    wrapped = np.concatenate([poles[:, -n_per_carrier:], poles, poles[:, :n_per_carrier]], axis=1)
    filt = np.stack([np.convolve(row, kernel, mode="same") for row in wrapped])[:, n_per_carrier:-n_per_carrier]
    return ThreePhasePWM(t, refs, car, poles, poles[0] - poles[1], filt)


# ---------------------------------------------------------------- coordinate transforms (amplitude-invariant)
def clarke(abc) -> np.ndarray:
    """abc -> (alpha, beta). Amplitude-invariant: a balanced set of peak A maps to a vector of length A."""
    x = np.asarray(abc, dtype=float)
    a, b, c = x[0], x[1], x[2]
    return np.stack([(2 * a - b - c) / 3, (b - c) / np.sqrt(3)])


def inv_clarke(ab) -> np.ndarray:
    x = np.asarray(ab, dtype=float)
    al, be = x[0], x[1]
    return np.stack([al, -al / 2 + np.sqrt(3) / 2 * be, -al / 2 - np.sqrt(3) / 2 * be])


def park(ab, theta) -> np.ndarray:
    """(alpha, beta) -> (d, q) in a frame whose d axis sits at angle ``theta``."""
    x, th = np.asarray(ab, dtype=float), np.asarray(theta, dtype=float)
    return np.stack([x[0] * np.cos(th) + x[1] * np.sin(th), -x[0] * np.sin(th) + x[1] * np.cos(th)])


def inv_park(dq, theta) -> np.ndarray:
    x, th = np.asarray(dq, dtype=float), np.asarray(theta, dtype=float)
    return np.stack([x[0] * np.cos(th) - x[1] * np.sin(th), x[0] * np.sin(th) + x[1] * np.cos(th)])


@dataclass(frozen=True)
class FOCState:
    i_abc: np.ndarray
    i_alpha_beta: np.ndarray
    flux: np.ndarray       # permanent-magnet flux vector (alpha, beta), per unit
    torque: float          # 1.5 * psi * iq for a surface PM machine (psi = 1 p.u., one pole pair)


def foc_currents(id: float, iq: float, theta: float) -> FOCState:
    """Stator currents for given d-q currents at rotor angle ``theta``. Torque is proportional to ``iq`` only."""
    ab = inv_park([id, iq], theta)
    return FOCState(inv_clarke(ab), ab, np.array([np.cos(theta), np.sin(theta)]), 1.5 * iq)


def rmf_vectors(t: float, f: float, phases: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Phase vectors (each pulsating along its own fixed axis) and their resultant, which rotates at constant length."""
    if phases < 2:
        raise ValueError("phases must be at least 2")
    k = np.arange(phases)
    axis = 2 * np.pi * k / phases
    i = np.cos(2 * np.pi * f * t - axis)
    vecs = np.stack([i * np.cos(axis), i * np.sin(axis)], axis=1)
    return vecs, vecs.sum(axis=0)


# ---------------------------------------------------------------- stability
def power_angle_curve(Pmax: float, delta) -> np.ndarray:
    return Pmax * np.sin(np.asarray(delta, dtype=float))


def swing_simulate(H: float, Pm: float, Pmax: float, delta0: float | None, D: float, disturbance: float, t,
                   f0: float = 50.0) -> tuple[np.ndarray, np.ndarray]:
    """Classical one-machine model. Returns (delta, d_omega in rad/s) sampled at ``t``.

    ``delta0`` is the equilibrium angle (``None`` -> asin(Pm/Pmax)); the run starts ``disturbance`` radians away from it.
    """
    if H <= 0 or not 0 <= Pm < Pmax:
        raise ValueError("H > 0 and 0 <= Pm < Pmax")
    d_eq = np.arcsin(Pm / Pmax) if delta0 is None else delta0
    t = np.asarray(t, dtype=float)
    k = np.pi * f0 / H

    def f(y):
        return np.array([y[1], k * (Pm - Pmax * np.sin(y[0]) - D * y[1] / (2 * np.pi * f0))])

    y, out = np.array([d_eq + disturbance, 0.0]), np.empty((len(t), 2))
    for i, ti in enumerate(t):
        out[i] = y
        h = (t[i + 1] - ti) if i + 1 < len(t) else 0.0
        k1 = f(y); k2 = f(y + h / 2 * k1); k3 = f(y + h / 2 * k2); k4 = f(y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return out[:, 0], out[:, 1]


@dataclass(frozen=True)
class EqualArea:
    delta0: float
    delta_clear: float
    delta_max: float
    a_acc: float
    a_dec: float
    stable: bool
    delta_critical: float


def equal_area(Pm: float, Pmax_pre: float, Pmax_fault: float, Pmax_post: float, delta_clear: float) -> EqualArea:
    """Equal-area criterion: areas in per-unit power x radians, and the critical clearing angle."""
    if not (0 <= Pm < Pmax_pre and Pm < Pmax_post and 0 <= Pmax_fault < Pmax_post):
        raise ValueError("need 0 <= Pm < Pmax_pre, Pm < Pmax_post and 0 <= Pmax_fault < Pmax_post")
    d0, dm = np.arcsin(Pm / Pmax_pre), np.pi - np.arcsin(Pm / Pmax_post)
    if not d0 < delta_clear < dm:
        raise ValueError("delta_clear must lie between the pre-fault angle and the post-fault maximum angle")
    acc = Pm * (delta_clear - d0) + Pmax_fault * (np.cos(delta_clear) - np.cos(d0))
    dec = Pmax_post * (np.cos(delta_clear) - np.cos(dm)) - Pm * (dm - delta_clear)
    cr = np.arccos(np.clip((Pm * (dm - d0) + Pmax_post * np.cos(dm) - Pmax_fault * np.cos(d0)) / (Pmax_post - Pmax_fault), -1, 1))
    return EqualArea(float(d0), float(delta_clear), float(dm), float(acc), float(dec), bool(dec >= acc), float(cr))


@dataclass(frozen=True)
class FaultCurrent:
    t: np.ndarray
    i: np.ndarray
    envelope: np.ndarray   # symmetrical peak envelope (three regions)
    dc: np.ndarray         # decaying DC offset


def fault_current(x_d2: float, x_d1: float, x_d: float, t_d2: float, t_d1: float, t_a: float, phase_angle: float = 0.0,
                  t=None, f: float = 50.0, E: float = 1.0) -> FaultCurrent:
    """Short-circuit current of a synchronous machine: subtransient, transient and steady-state decay plus DC offset."""
    if not 0 < x_d2 < x_d1 < x_d or min(t_d2, t_d1, t_a) <= 0:
        raise ValueError("need 0 < x_d'' < x_d' < x_d and positive time constants")
    t = np.linspace(0, 0.6, 1200) if t is None else np.asarray(t, dtype=float)
    env = np.sqrt(2) * E * ((1 / x_d2 - 1 / x_d1) * np.exp(-t / t_d2) + (1 / x_d1 - 1 / x_d) * np.exp(-t / t_d1) + 1 / x_d)
    dc = -np.sqrt(2) * E / x_d2 * np.exp(-t / t_a) * np.sin(phase_angle - np.pi / 2)
    return FaultCurrent(t, env * np.sin(2 * np.pi * f * t + phase_angle - np.pi / 2) + dc, env, dc)


def damper_torque(slip, s_crit: float = 0.1, t_max: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Damper-bar current (per unit, signed like the slip) and braking torque (opposes the slip, zero at synchronism)."""
    if s_crit <= 0:
        raise ValueError("s_crit must be positive")
    s = np.asarray(slip, dtype=float)
    return s / np.hypot(s, s_crit), -t_max * 2 * s * s_crit / (s * s + s_crit ** 2)
