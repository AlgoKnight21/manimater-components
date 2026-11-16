"""Scenes for ``viz.ml_advanced`` (README section 6): each ``play_*`` runs as a storyboard component or a standalone Scene.

Every scene follows the house timing rules: the finished static frame fades in whole, then the parts that carry the
meaning move, and the scene ends by filling whatever narration time is left (``ctx.finish``).
"""
from __future__ import annotations

from manim import Group, ValueTracker

from ...ctx import Ctx
from ..core.focus import fade_in_whole
from ._math import attention_weights, seeded_embeddings
from .attention import AttentionArcs, AttentionHeatmap
from .attention_head import AttentionHead
from .kv_cache import DecodeTimeline, KVCacheBuffer
from .multihead import MultiHeadAttention
from .positional import PEHeatmap, PositionalEncodingSum, PositionalEncodingWaves
from .qkv import QKVProjection

DEFAULT_TOKENS = ("The", "cat", "sat", "on", "mat")
MAX_TOKENS = 8
ATTENTION_VIEWS = ("heatmap", "arcs", "both")
PE_VIEWS = ("waves", "sum", "heatmap")
KV_VIEWS = ("cache", "timeline")


def _tokens(tokens, lo: int = 2) -> list[str]:
    toks = [str(t) for t in (tokens or DEFAULT_TOKENS)][:MAX_TOKENS]
    if len(toks) < lo:
        raise ValueError(f"needs at least {lo} tokens")
    return toks


def _slots(ctx: Ctx, *weights: float, floor: float = 0.5):
    """Intuition: cutting the time that is left into shares, so the picture ends when the narration does."""
    left = ctx.remaining
    total = float(sum(weights)) or 1.0
    return [max(floor, left * w / total) for w in weights]


def play_attention_head(ctx: Ctx, tokens=None, causal: bool = True, heads: int = 1) -> None:
    """Intuition: every word asks a question, every word offers a label, and the best matches decide who to listen to."""
    toks = _tokens(tokens)
    heads = max(1, min(4, int(heads)))
    if heads == 1:
        head = AttentionHead(toks, causal=causal)
        ctx.fit(head)
        fade_in_whole(ctx, head, 1.0)
        a, b = _slots(ctx, 1, 1)
        ctx.play(head.animate_to("scores", run_time=a), run_time=a)
        ctx.play(head.animate_to("output", run_time=b), run_time=b)
    else:
        mh = MultiHeadAttention(toks, n_heads=heads, causal=causal)
        ctx.fit(mh)
        fade_in_whole(ctx, mh, 1.0)
        a, b = _slots(ctx, 3, 2)
        ctx.play(mh.animate_split(run_time=a), run_time=a)
        ctx.play(mh.animate_concat(run_time=b), run_time=b)
    ctx.finish()


def play_attention_heatmap(ctx: Ctx, tokens=None, causal: bool = True, view: str = "both", query_index: int = -1) -> None:
    """Intuition: a table of who looks at whom. Scores, a mask that hides the future, then softmax turns rows into shares."""
    if view not in ATTENTION_VIEWS:
        raise ValueError(f"view must be one of {list(ATTENTION_VIEWS)}")
    toks = _tokens(tokens)
    n = len(toks)
    q = n - 1 if query_index < 0 or query_index >= n else int(query_index)
    Q, K = seeded_embeddings(n, 4, seed=5), seeded_embeddings(n, 4, seed=11)
    heat = AttentionHeatmap(Q, K, toks, causal=causal)
    arcs = AttentionArcs(toks, attention_weights(Q, K, causal=causal)[q], query_index=q)
    if view == "heatmap":
        shown = Group(heat)
    elif view == "arcs":
        shown = Group(arcs)
    else:
        heat.scale(0.8).move_to((-3.1, 0.6, 0))
        arcs.scale(0.75).move_to((3.3, 0.6, 0))
        shown = Group(heat, arcs)
    ctx.fit(shown)
    fade_in_whole(ctx, shown, 1.0)
    if view == "arcs":
        (t,) = _slots(ctx, 1)
        ctx.play(arcs.animate_in(run_time=t), run_time=t)
    else:
        steps = [heat.animate_scale]
        if causal:
            steps.append(heat.animate_mask)
        steps.append(heat.animate_softmax)
        if view == "both":
            steps.append(arcs.animate_in)
        for anim, t in zip(steps, _slots(ctx, *([1.0] * len(steps)))):
            ctx.play(anim(run_time=t), run_time=t)
    ctx.finish()


def play_positional_encoding(ctx: Ctx, view: str = "waves", tokens=None, d_model: int = 8, n_positions: int = 12) -> None:
    """Intuition: a clock face with many hands. Each position gets its own pattern of hand angles, so order is never lost."""
    if view not in PE_VIEWS:
        raise ValueError(f"view must be one of {list(PE_VIEWS)}")
    d_model = max(4, min(16, int(d_model)))
    d_model -= d_model % 2
    n_positions = max(4, min(24, int(n_positions)))
    if view == "waves":
        w = PositionalEncodingWaves(d_model=d_model, n_positions=n_positions, n_freqs=min(4, d_model // 2), base=60.0, position=0.0)
        ctx.fit(Group(w))
        fade_in_whole(ctx, Group(w), 1.0)
        (t,) = _slots(ctx, 1)
        ctx.play(w.animate_scan(float(n_positions - 1), run_time=t), run_time=t)
    elif view == "sum":
        s = PositionalEncodingSum(_tokens(tokens), d_model)
        ctx.fit(s)
        fade_in_whole(ctx, s, 1.0)
        (t,) = _slots(ctx, 1)
        ctx.play(s.animate_add(run_time=t), run_time=t)
    else:
        h = PEHeatmap(max(n_positions, 8), max(d_model, 8), base=500.0).set_dim(0).set_position(0)
        ctx.fit(Group(h))
        fade_in_whole(ctx, Group(h), 1.0)
        dims = [min(2, h.d_model - 1), min(5, h.d_model - 1)]
        slots = _slots(ctx, 1, 1, 1)
        ctx.play(h.animate_dim(dims[0], run_time=slots[0]), run_time=slots[0])
        ctx.play(h.animate_position(min(6, h.n_positions - 1), run_time=slots[1]), run_time=slots[1])
        ctx.play(h.animate_dim(dims[1], run_time=slots[2]), run_time=slots[2])
    ctx.finish()


def play_kv_cache(ctx: Ctx, view: str = "cache", steps: int = 8, window: int = 0) -> None:
    """Intuition: a notebook of what every earlier word offered, so the model reads it back instead of recomputing it."""
    if view not in KV_VIEWS:
        raise ValueError(f"view must be one of {list(KV_VIEWS)}")
    steps = max(3, min(12, int(steps)))
    win = int(window) if window and 1 <= int(window) < steps else None
    if view == "cache":
        buf = KVCacheBuffer(steps, win)
        ctx.fit(Group(buf))
        fade_in_whole(ctx, Group(buf), 1.0)
        per = max(0.3, ctx.remaining * 0.85 / steps)
        for i in range(steps):
            ctx.play(buf.animate_step(i, run_time=per), run_time=per)
    else:
        tl = DecodeTimeline(steps, win)
        ctx.fit(Group(tl))
        fade_in_whole(ctx, Group(tl), 1.0)
        (t,) = _slots(ctx, 1)
        ctx.play(tl.animate_play(run_time=max(1.0, t * 0.9)), run_time=max(1.0, t * 0.9))
    ctx.finish()


def play_qkv_projection(ctx: Ctx, tokens=None, d_head: int = 3) -> None:
    """Intuition: one word, three roles. The same row is turned into a question, a label and a payload by three matrices."""
    n = len(_tokens(tokens))
    d_head = max(2, min(4, int(d_head)))
    qkv = QKVProjection(seeded_embeddings(n, 6), d_head=d_head).set_token(0)
    ctx.fit(Group(qkv))
    fade_in_whole(ctx, Group(qkv), 1.0)
    (t,) = _slots(ctx, 1)
    ctx.play(qkv.animate_project(run_time=max(1.0, t * 0.9)), run_time=max(1.0, t * 0.9))
    ctx.finish()
