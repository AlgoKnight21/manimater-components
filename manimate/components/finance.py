"""Quantitative-finance pack: thin adapters over ``manimate.viz.finance`` (see ``viz/README.md``, section 10).

Every picture is computed by plain functions in ``viz/finance/_math.py``; these adapters only declare the params
(few, forgiving, with plain-English range errors) and call the matching ``play_*`` scene. Educational, not advice.
"""
from __future__ import annotations

from ..ctx import Ctx
from ..registry import Param as P, component
from ..viz.finance import scenes as S

_D = "finance"


def _rng(lo: float, hi: float, what: str):
    return lambda v: None if lo <= v <= hi else f"{what} must be between {lo:g} and {hi:g}"


@component(
    "option_payoff",
    "Option profit or loss at expiry as a hockey stick with break-even and max gain/loss; also straddle, bull spread, collar.",
    {"kind": P("string", "call or put", default="call", choices=["call", "put"]),
     "strategy": P("string", "position shape", default="single", choices=S.fm.STRATEGIES),
     "strike": P("number", "strike price", default=100, validator=_rng(1, 100000, "strike")),
     "premium": P("number", "price paid per share", default=5, validator=_rng(0, 50000, "premium")),
     "side": P("string", "long (buy) or short (sell)", default="long", choices=["long", "short"])},
    {"kind": "call", "strategy": "single", "strike": 100, "premium": 5, "side": "long"},
    domain=_D,
)
def option_payoff(ctx: Ctx, kind: str = "call", strategy: str = "single", strike: float = 100, premium: float = 5,
                  side: str = "long") -> None:
    S.play_option_payoff(ctx, kind, strategy, strike, premium, side)


@component(
    "option_greeks",
    "Option value curve sharpening toward its payoff as time runs out, with delta tangent and a Greek (delta, gamma, theta, vega) beside it.",
    {"greek": P("string", "which Greek to profile", default="delta", choices=["delta", "gamma", "theta", "vega"]),
     "option_type": P("string", "call or put", default="call", choices=["call", "put"]),
     "strike": P("number", "strike price", default=100, validator=_rng(1, 100000, "strike")),
     "volatility": P("number", "yearly volatility, 0.05 to 1", default=0.25, validator=_rng(0.05, 1.0, "volatility"))},
    {"greek": "gamma", "option_type": "call", "strike": 100, "volatility": 0.25},
    domain=_D,
)
def option_greeks(ctx: Ctx, greek: str = "delta", option_type: str = "call", strike: float = 100, volatility: float = 0.25) -> None:
    S.play_option_greeks(ctx, greek, option_type, strike, volatility)


@component(
    "garch_volatility",
    "Volatility clustering: returns over a band that widens after big moves (GARCH), or how volatility fades after one shock.",
    {"view": P("string", "bands over returns, or decay after one shock", default="bands", choices=["bands", "shock"]),
     "alpha": P("number", "reaction to the last move, 0.01 to 0.3", default=0.1, validator=_rng(0.01, 0.3, "alpha")),
     "beta": P("number", "memory of past volatility, 0.3 to 0.97", default=0.85, validator=_rng(0.3, 0.97, "beta")),
     "shock_at": P("integer", "day of the big move (bands view), 5 to 80", default=30, validator=_rng(5, 80, "shock_at"))},
    {"view": "bands", "alpha": 0.1, "beta": 0.85, "shock_at": 30},
    domain=_D,
)
def garch_volatility(ctx: Ctx, view: str = "bands", alpha: float = 0.1, beta: float = 0.85, shock_at: int = 30) -> None:
    S.play_garch_volatility(ctx, view, alpha, beta, shock_at)


@component(
    "yield_curve_shift",
    "Yield curve (3M to 30Y) morphing under a parallel shift, steepening, flattening or inversion, with what it means.",
    {"shift": P("string", "how the curve moves", default="steepen", choices=S.fm.SHIFTS),
     "size_bps": P("number", "size in basis points, 10 to 250", default=100, validator=_rng(10, 250, "size_bps"))},
    {"shift": "invert", "size_bps": 100},
    domain=_D,
)
def yield_curve_shift(ctx: Ctx, shift: str = "steepen", size_bps: float = 100) -> None:
    S.play_yield_curve_shift(ctx, shift, size_bps)


@component(
    "bond_price_yield",
    "Bond price against yield (inverse, convex); or duration tangent vs convexity parabola vs the real curve.",
    {"view": P("string", "price curve, or duration and convexity", default="price_yield", choices=["price_yield", "duration_convexity"]),
     "coupon": P("number", "coupon in percent, 0.5 to 12", default=5, validator=_rng(0.5, 12, "coupon")),
     "maturity": P("integer", "years to maturity, 1 to 30", default=10, validator=_rng(1, 30, "maturity")),
     "yield_pct": P("number", "starting yield in percent, 2 to 10", default=5, validator=_rng(2, 10, "yield_pct"))},
    {"view": "duration_convexity", "coupon": 5, "maturity": 10, "yield_pct": 5},
    domain=_D,
)
def bond_price_yield(ctx: Ctx, view: str = "price_yield", coupon: float = 5, maturity: int = 10, yield_pct: float = 5) -> None:
    S.play_bond_price_yield(ctx, view, coupon, maturity, yield_pct)


@component(
    "brownian_motion",
    "Random walk, Brownian motion or geometric Brownian motion (prices): many paths, and the histogram of where they end.",
    {"variant": P("string", "random_walk, brownian or gbm (price model)", default="brownian", choices=["random_walk", "brownian", "gbm"]),
     "paths": P("integer", "paths drawn, 3 to 16", default=8, validator=_rng(3, 16, "paths")),
     "volatility": P("number", "yearly volatility, 0.02 to 1", default=0.2, validator=_rng(0.02, 1.0, "volatility")),
     "drift": P("number", "yearly drift, -0.5 to 0.5", default=0.05, validator=_rng(-0.5, 0.5, "drift"))},
    {"variant": "gbm", "paths": 8, "volatility": 0.2, "drift": 0.05},
    domain=_D,
)
def brownian_motion(ctx: Ctx, variant: str = "brownian", paths: int = 8, volatility: float = 0.2, drift: float = 0.05) -> None:
    S.play_brownian_motion(ctx, variant, paths, volatility, drift)


@component(
    "binomial_tree",
    "Recombining price tree and a European call valued by working backward from the last step with risk-neutral probability.",
    {"steps": P("integer", "steps, 2 to 4", default=3, validator=_rng(2, 4, "steps")),
     "up": P("number", "up factor per step, 1.02 to 1.5", default=1.1, validator=_rng(1.02, 1.5, "up")),
     "down": P("number", "down factor per step, 0.5 to 0.98", default=0.9, validator=_rng(0.5, 0.98, "down")),
     "rate": P("number", "interest rate per step, e.g. 0.02", default=0.02, validator=_rng(0, 0.1, "rate")),
     "strike": P("number", "strike price (start price is 100), 50 to 200", default=100, validator=_rng(50, 200, "strike"))},
    {"steps": 3, "up": 1.1, "down": 0.9, "rate": 0.02, "strike": 100},
    domain=_D,
)
def binomial_tree(ctx: Ctx, steps: int = 3, up: float = 1.1, down: float = 0.9, rate: float = 0.02, strike: float = 100) -> None:
    S.play_binomial_tree(ctx, steps, up, down, rate, strike)


@component(
    "order_book_matching",
    "Limit order book with bids, asks and spread: a buy order sweeps price levels, partly fills, or joins the queue (price-time priority).",
    {"scenario": P("string", "what the incoming order does", default="market_buy", choices=["market_buy", "partial_fill", "limit_joins_queue"]),
     "depth": P("integer", "price levels per side, 3 to 5", default=4, validator=_rng(3, 5, "depth"))},
    {"scenario": "partial_fill", "depth": 4},
    domain=_D,
)
def order_book_matching(ctx: Ctx, scenario: str = "market_buy", depth: int = 4) -> None:
    S.play_order_book_matching(ctx, scenario, depth)
