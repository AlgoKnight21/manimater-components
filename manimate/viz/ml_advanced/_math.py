"""Pure maths behind the ml_advanced pictures. No Manim import, so every number here is unit-tested directly."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.numeric import softmax


def _mat(x, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 2 or a.size == 0:
        raise ValueError(f"{name} must be a non-empty 2-D array")
    return a


def qkv_project(X, Wq, Wk, Wv) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project token embeddings ``X`` (n x d_model) into queries, keys and values (each n x d_head)."""
    X, Wq, Wk, Wv = _mat(X, "X"), _mat(Wq, "Wq"), _mat(Wk, "Wk"), _mat(Wv, "Wv")
    if not (Wq.shape[0] == Wk.shape[0] == Wv.shape[0] == X.shape[1]):
        raise ValueError("each weight matrix needs d_model rows, the same as X has columns")
    return X @ Wq, X @ Wk, X @ Wv


def attention_scores(Q, K, causal: bool = False, scale: bool = True) -> np.ndarray:
    """Raw scores ``Q K^T`` (optionally divided by sqrt(d_k)). With ``causal`` the future is ``-inf``."""
    Q, K = _mat(Q, "Q"), _mat(K, "K")
    if Q.shape[1] != K.shape[1]:
        raise ValueError("Q and K must have the same width")
    S = Q @ K.T
    if scale:
        S = S / np.sqrt(Q.shape[1])
    if causal:
        if S.shape[0] != S.shape[1]:
            raise ValueError("a causal mask needs as many queries as keys")
        S = np.where(np.triu(np.ones_like(S, dtype=bool), k=1), -np.inf, S)
    return S


def attention_weights(Q, K, causal: bool = False, scale: bool = True) -> np.ndarray:
    """Softmax of the scores: each row is a budget of attention that sums to 1."""
    return softmax(attention_scores(Q, K, causal, scale), axis=-1)


def attention_output(Q, K, V, causal: bool = False, scale: bool = True) -> np.ndarray:
    """Weighted sum of the value rows."""
    V = _mat(V, "V")
    W = attention_weights(Q, K, causal, scale)
    if W.shape[1] != V.shape[0]:
        raise ValueError("V needs one row per key")
    return W @ V


@dataclass(frozen=True)
class PEComponent:
    dim: int          # column of the positional-encoding vector
    kind: str         # "sin" or "cos"
    omega: float      # angular frequency (radians per position)
    value: float      # the term's value at this position


def _omegas(d_model: int, base: float) -> np.ndarray:
    return 1.0 / base ** (2 * (np.arange(d_model) // 2) / d_model)


def sinusoidal_pe(n_pos: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """Positional encoding matrix (n_pos x d_model): sin on even columns, cos on odd, frequencies spaced geometrically."""
    if n_pos < 1 or d_model < 2:
        raise ValueError("need n_pos >= 1 and d_model >= 2")
    if base <= 1:
        raise ValueError("base must be greater than 1")
    angles = np.arange(n_pos)[:, None] * _omegas(d_model, base)[None, :]
    pe = np.empty((n_pos, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe


def pe_components(pos: int, d_model: int, base: float = 10000.0) -> list[PEComponent]:
    """The individual sin/cos terms at one position: the pieces the wave-summation picture adds up."""
    if pos < 0:
        raise ValueError("pos must be >= 0")
    row = sinusoidal_pe(pos + 1, d_model, base)[pos]
    om = _omegas(d_model, base)
    return [PEComponent(k, "sin" if k % 2 == 0 else "cos", float(om[k]), float(row[k])) for k in range(d_model)]


@dataclass(frozen=True)
class KVStep:
    step: int                    # which token is being generated (0-based)
    attended: tuple[int, ...]    # token indices this step attends to (includes itself)
    cached: tuple[int, ...]      # of those, the ones whose K and V are read from the cache
    evicted: int | None          # token that fell out of the window at this step
    work_with_cache: int         # K/V cells computed this step when a cache exists
    work_without_cache: int      # K/V cells computed this step when nothing is cached


def kv_cache_steps(n_tokens: int, window: int | None = None) -> list[KVStep]:
    """Per-step accounting of autoregressive decoding with and without a KV cache (optionally a sliding window)."""
    if n_tokens < 1:
        raise ValueError("n_tokens must be at least 1")
    if window is not None and window < 1:
        raise ValueError("window must be at least 1")
    out = []
    for t in range(n_tokens):
        lo = 0 if window is None else max(0, t - window + 1)
        attended = tuple(range(lo, t + 1))
        evicted = t - window if window is not None and t - window >= 0 else None
        out.append(KVStep(t, attended, attended[:-1], evicted, 1, len(attended)))
    return out


# ---- helpers used by the QKV / positional / KV-cache / head / multi-head pictures ------------------------------------
from ..core.numeric import rng  # noqa: E402

MAX_TOKENS = 16  # the cache strip and timeline stay legible up to here


def init_weights(shape: tuple[int, ...], seed: int = 7, scale: float = 1.0) -> np.ndarray:
    """Small seeded weights, so two renders of the same storyboard show the same numbers."""
    return rng(seed).normal(0.0, scale / np.sqrt(max(1, shape[0])), size=shape)


def seeded_embeddings(n_tokens: int, d_model: int, seed: int = 7) -> np.ndarray:
    """Stand-in token embeddings, shape ``(n_tokens, d_model)``."""
    if not 1 <= n_tokens <= MAX_TOKENS:
        raise ValueError(f"n_tokens must be between 1 and {MAX_TOKENS}")
    if d_model < 1:
        raise ValueError("d_model must be at least 1")
    return rng(seed).normal(0.0, 1.0, size=(n_tokens, d_model))


def pe_frequencies(d_model: int, base: float = 10000.0) -> tuple[np.ndarray, np.ndarray]:
    """One angular frequency and one wavelength (in positions) per sin/cos pair."""
    if d_model < 2 or d_model % 2:
        raise ValueError("d_model must be an even number, at least 2")
    if base <= 1:
        raise ValueError("base must be greater than 1")
    freqs = base ** (-np.arange(0, d_model, 2) / d_model)
    return freqs, 2 * np.pi / freqs


@dataclass(frozen=True)
class HeadData:
    Q: np.ndarray
    K: np.ndarray
    V: np.ndarray
    scores: np.ndarray
    weights: np.ndarray
    output: np.ndarray


def attention_head(X, Wq, Wk, Wv, causal: bool = False, scale: bool = True) -> HeadData:
    """Run one whole head and keep every intermediate, so visuals and tests read the same numbers."""
    Q, K, V = qkv_project(X, Wq, Wk, Wv)
    scores = attention_scores(Q, K, causal, scale)
    weights = softmax(scores, axis=-1)
    return HeadData(Q, K, V, scores, weights, weights @ V)
