"""Deep learning adapters (viz README sections 6 and 12.1): thin wrappers over ``viz.ml_advanced`` scenes.

Every picture is computed by plain functions in ``viz/ml_advanced/_math.py``. The weights behind the pictures are seeded
and illustrative, so the scenes teach the *mechanism* (what is multiplied, masked and normalised), not a trained model.
"""
from __future__ import annotations

from ..ctx import Ctx
from ..registry import Param as P, component
from ..viz.ml_advanced import scenes as S

_D = "machine-learning"
_TOKENS_HINT = "words or sub-words, 2 to 8, each a short string"


def _in_range(lo: float, hi: float, what: str = "value"):
    return lambda v: None if lo <= v <= hi else f"{what} must be between {lo:g} and {hi:g}"


def _tokens(v: list) -> str | None:
    if len(v) < 2:
        return "needs at least two tokens"
    if len(v) > S.MAX_TOKENS:
        return f"takes at most {S.MAX_TOKENS} tokens"
    return None if all(isinstance(t, str) and 0 < len(t) <= 14 for t in v) else "each token must be a short non-empty string (14 characters at most)"


@component(
    "attention_head",
    "One self-attention head: project each word into a query, key and value, score every pair, then mix the values by the "
    "scores. With several heads it shows the split into parallel heads and the concatenation back.",
    {"tokens": P("string[]", f"the sentence as tokens ({_TOKENS_HINT})", default=list(S.DEFAULT_TOKENS), validator=_tokens),
     "causal": P("boolean", "hide future words, as a language model does while generating", default=True),
     "heads": P("integer", "1 shows a single head in three stages; 2 to 4 shows several heads side by side", default=1,
                validator=_in_range(1, 4, "heads"))},
    {"tokens": ["The", "cat", "sat", "on", "mat"], "causal": True, "heads": 1},
    domain=_D,
)
def attention_head(ctx: Ctx, tokens: list | None = None, causal: bool = True, heads: int = 1) -> None:
    S.play_attention_head(ctx, tokens=tokens, causal=causal, heads=int(heads))


@component(
    "attention_heatmap",
    "Attention as a grid of who looks at whom: raw scores, scaling, the causal mask, then softmax turning each row into "
    "shares that add to one. Optionally with arcs from one word to the words it attends to.",
    {"tokens": P("string[]", f"the sentence as tokens ({_TOKENS_HINT})", default=list(S.DEFAULT_TOKENS), validator=_tokens),
     "causal": P("boolean", "mask out future words", default=True),
     "view": P("string", "which picture", default="both", choices=list(S.ATTENTION_VIEWS)),
     "query_index": P("integer", "which word the arcs start from, counting from 0; -1 means the last word", default=-1,
                      validator=_in_range(-1, 7, "query_index"))},
    {"tokens": ["The", "cat", "sat", "down"], "causal": True, "view": "both", "query_index": -1},
    domain=_D,
)
def attention_heatmap(ctx: Ctx, tokens: list | None = None, causal: bool = True, view: str = "both", query_index: int = -1) -> None:
    S.play_attention_heatmap(ctx, tokens=tokens, causal=causal, view=view, query_index=int(query_index))


@component(
    "positional_encoding",
    "Positional encoding in a transformer: sine and cosine waves of different speeds giving every position a unique "
    "pattern, how that pattern is added to a word's embedding, or the whole table seen from above as a heatmap.",
    {"view": P("string", "which picture", default="waves", choices=list(S.PE_VIEWS)),
     "tokens": P("string[]", f"the words for the sum view ({_TOKENS_HINT})", default=list(S.DEFAULT_TOKENS), validator=_tokens),
     "d_model": P("integer", "embedding size, an even number from 4 to 16", default=8, validator=_in_range(4, 16, "d_model")),
     "n_positions": P("integer", "how many positions to draw, 4 to 24", default=12, validator=_in_range(4, 24, "n_positions"))},
    {"view": "waves", "tokens": ["The", "cat", "sat", "on", "mat"], "d_model": 8, "n_positions": 12},
    domain=_D,
)
def positional_encoding(ctx: Ctx, view: str = "waves", tokens: list | None = None, d_model: int = 8, n_positions: int = 12) -> None:
    S.play_positional_encoding(ctx, view=view, tokens=tokens, d_model=int(d_model), n_positions=int(n_positions))


@component(
    "kv_cache",
    "The KV cache in LLM decoding: each new token writes its key and value into a slot (a sliding window overwrites the "
    "oldest), or a timeline comparing the work with and without the cache.",
    {"view": P("string", "slots filling up, or the work saved", default="cache", choices=list(S.KV_VIEWS)),
     "steps": P("integer", "tokens generated, 3 to 12", default=8, validator=_in_range(3, 12, "steps")),
     "window": P("integer", "sliding-window size; 0 keeps every token", default=0, validator=_in_range(0, 12, "window"))},
    {"view": "cache", "steps": 8, "window": 4},
    domain=_D,
)
def kv_cache(ctx: Ctx, view: str = "cache", steps: int = 8, window: int = 0) -> None:
    S.play_kv_cache(ctx, view=view, steps=int(steps), window=int(window))


@component(
    "qkv_projection",
    "Query, key and value: the same word embedding multiplied by three weight matrices, lighting up each word's row in Q, K and V.",
    {"tokens": P("string[]", f"the sentence as tokens ({_TOKENS_HINT})", default=list(S.DEFAULT_TOKENS[:4]), validator=_tokens),
     "d_head": P("integer", "size of each query, key and value vector, 2 to 4", default=3, validator=_in_range(2, 4, "d_head"))},
    {"tokens": ["I", "love", "maths"], "d_head": 3},
    domain=_D,
)
def qkv_projection(ctx: Ctx, tokens: list | None = None, d_head: int = 3) -> None:
    S.play_qkv_projection(ctx, tokens=tokens, d_head=int(d_head))
