"""Scene beats for the finance visuals: ``play_*(ctx, ...)``. Each is registered by ``components/finance.py``.

Every beat follows the component rules: the finished static frame fades in together, then the parts that carry the
meaning move, and the last animation uses ``ctx.remaining`` so the picture matches its narration.
Each model states its assumption on screen. Educational only, not investment advice.
"""
from __future__ import annotations

import math

import numpy as np
from manim import (
    DOWN, LEFT, RIGHT, UP, FadeIn, FadeOut, GrowFromEdge, LaggedStart, Line, Polygon, Rectangle, RoundedRectangle,
    SurroundingRectangle, Text, Transform, ValueTracker, VGroup, Create, FadeTransform, always_redraw, rate_functions,
)

from ... import theme
from ...ctx import Ctx, label
from . import _math as fm
from .mobjects import Plot, live_text, pad_range

SMOOTH = rate_functions.smooth
LINEAR = rate_functions.linear


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _fade_whole(ctx: Ctx, *mobs) -> None:
    ctx.play(FadeIn(VGroup(*mobs)), run_time=min(1.4, max(0.6, ctx.remaining * 0.2)))


def _main(ctx: Ctx, tail: float = 0.3) -> float:
    return max(1.0, ctx.remaining - tail)


def _note(text: str, x: float, y: float, width: int = 34, size: int = 20, color: str = theme.MUTED, align: str = "left"):
    t = label(text, size, color, width=width, align=align)
    t.move_to((x, y, 0), aligned_edge=LEFT if align == "left" else None) if align == "left" else t.move_to((x, y, 0))
    return t


def _left_text(text: str, x: float, y: float, size: int = 24, color: str = theme.FG, weight: str = "NORMAL"):
    t = Text(text, font_size=size, color=color, weight=weight)
    t.move_to((x, y, 0), aligned_edge=LEFT)
    return t


def _live_left(key_fn, fmt_fn, x: float, y: float, size: int = 26, color_fn=None):
    """A left-anchored readout that is rebuilt only when its key changes."""
    def build(k):
        t = Text(fmt_fn(k), font_size=size, color=color_fn(k) if color_fn else theme.FG, weight="BOLD")
        t.move_to((x, y, 0), aligned_edge=LEFT)
        return t
    return live_text(key_fn, build)


# =====================================================================================================================
# 1. Option payoff
# =====================================================================================================================
_STRAT_TITLE = {"straddle": "Straddle: a call and a put at one strike", "bull_spread": "Bull call spread: buy low strike, sell high",
                "collar": "Collar: own the stock, buy a put, sell a call"}


def play_option_payoff(ctx: Ctx, kind: str = "call", strategy: str = "single", strike: float = 100.0,
                       premium: float = 5.0, side: str = "long") -> None:
    strike = _clamp(strike, 1, 100000)
    premium = _clamp(premium, 0, strike * 0.5)
    legs = fm.strategy_legs(strategy, kind, strike, premium, side)
    stock_at = strike if strategy == "collar" else None
    lo, hi = 0.6 * strike, 1.4 * strike
    pts = fm.profit_curve(legs, lo, hi, 140, stock_at)
    ys = [y for _, y in pts]
    yr = pad_range(min(ys + [0.0]), max(ys + [0.0]), 0.15)
    profit = lambda s: fm.strategy_profit(legs, s) + (fm.stock_profit(s, stock_at) if stock_at is not None else 0.0)

    # Bounds over the full price range 0..10x strike: "unlimited" only where the curve is still rising/falling outward.
    wide = fm.profit_curve(legs, 0.0, 10 * strike, 400, stock_at)
    wy = [y for _, y in wide]
    slope_end = wide[-1][1] - wide[-2][1]
    max_gain_txt = "Max gain: unlimited" if (wy[-1] == max(wy) and slope_end > 1e-9) else f"Max gain: {max(wy):.1f}"
    max_loss_txt = "Max loss: unlimited" if (wy[-1] == min(wy) and slope_end < -1e-9) else f"Max loss: {-min(wy):.1f}"

    title = _STRAT_TITLE.get(strategy) or f"{side.title()} {kind}: profit at expiry"
    head = ctx.heading(title)
    if head.width > 12.0:
        head.scale_to_fit_width(12.0)
    plot = Plot((lo, hi), yr, width=7.8, height=3.5, x_label="Stock price at expiry", y_label="Profit or loss",
                x_fmt="{:.0f}", y_fmt="{:+.0f}").move_to((-1.4, 0.35, 0))
    curve = plot.curve(pts, theme.ACCENT, 5)
    bes = fm.break_evens(pts)
    be_marks = VGroup()
    for b in bes:
        d = plot.dot(b, 0.0, theme.YELLOW, 0.1)
        t = label(f"break-even {b:.1f}", 20, theme.YELLOW)
        t.next_to(d, UP if yr[0] < -0.2 * (yr[1] - yr[0]) else DOWN, buff=0.18)
        be_marks.add(d, t)
    strike_line = plot.vline(strike, theme.MUTED)
    strike_tag = label(f"strike {strike:.0f}", 18, theme.MUTED).next_to(plot.p(strike, plot.y1), UP, buff=0.05)

    px = 3.05
    info = VGroup(_left_text(max_gain_txt, px, 1.75, 24, theme.GOOD), _left_text(max_loss_txt, px, 1.2, 24, theme.DANGER),
                  _note("Payoff at expiry, premium included. A model for learning, not advice.", px, -1.35, width=26)
                  ).set_z_index(1)

    S = ValueTracker(lo)
    dot = always_redraw(lambda: plot.dot(S.get_value(), profit(S.get_value()), theme.YELLOW, 0.12))
    drop = always_redraw(lambda: Line(plot.p(S.get_value(), 0.0), plot.p(S.get_value(), profit(S.get_value())),
                                      color=theme.YELLOW, stroke_width=3))
    spot = _live_left(lambda: round(S.get_value()), lambda k: f"Price at expiry: {k}", px, 0.45, 24)
    pnl = _live_left(lambda: round(profit(S.get_value()), 1),
                     lambda k: ("Profit: +" if k > 0 else "Loss: " if k < 0 else "Break-even: ") + f"{abs(k):.1f}",
                     px, -0.15, 28, lambda k: theme.GOOD if k > 0 else (theme.DANGER if k < 0 else theme.YELLOW))

    _fade_whole(ctx, head, plot, strike_line, strike_tag, curve, be_marks, info, spot, pnl, dot, drop)
    ctx.add(dot, drop, spot, pnl)
    ctx.play(S.animate.set_value(hi), run_time=_main(ctx), rate_func=SMOOTH)
    ctx.finish()


# =====================================================================================================================
# 2. Option Greeks: value curve sharpening as time runs out, with delta tangent and a Greek profile beside it
# =====================================================================================================================
_GREEK_TITLE = {"delta": "Delta = slope", "gamma": "Gamma = curvature", "theta": "Theta = decay per year", "vega": "Vega = per 1.00 vol"}
_GREEK_HEAD = {"delta": "Delta: how much value moves per 1 of price", "gamma": "Gamma: how fast delta itself changes",
               "theta": "Theta: value lost as time passes", "vega": "Vega: sensitivity to volatility"}


def play_option_greeks(ctx: Ctx, greek: str = "delta", option_type: str = "call", strike: float = 100.0,
                       volatility: float = 0.25) -> None:
    K = _clamp(strike, 1, 100000)
    sigma = _clamp(volatility, 0.05, 1.0)
    r = 0.03
    lo, hi = 0.6 * K, 1.4 * K
    grid = [lo + (hi - lo) * i / 40 for i in range(41)]
    taus = [1.0, 0.6, 0.3, 0.1, 0.03]
    vmax = max(fm.bs_price(s, K, 1.0, r, sigma, option_type) for s in grid)
    vmax = max(vmax, max(max(K - s, s - K, 0) for s in (lo, hi)))
    gvals = [fm.bs_greeks(s, K, t, r, sigma, option_type)[greek] for t in taus for s in grid]
    gr = pad_range(min(gvals + [0.0]), max(gvals + [0.0]), 0.1)

    head = ctx.heading(_GREEK_HEAD[greek])
    if head.width > 12.0:
        head.scale_to_fit_width(12.0)
    left = Plot((lo, hi), (0, vmax * 1.1), width=4.6, height=2.8, x_label="Stock price", y_label="Option value",
                x_fmt="{:.0f}", y_fmt="{:.0f}").move_to((-3.0, 0.6, 0))
    right = Plot((lo, hi), gr, width=4.6, height=2.8, x_label="Stock price", y_label=_GREEK_TITLE[greek],
                 x_fmt="{:.0f}", y_fmt="{:.2g}").move_to((3.4, 0.6, 0))
    payoff = left.curve([(s, max(s - K, 0.0) if option_type == "call" else max(K - s, 0.0)) for s in grid],
                        theme.MUTED, 3, 0.9)
    payoff_tag = label("grey: payoff at expiry", 18, theme.MUTED).move_to(left.p(lo + 0.36 * (hi - lo), vmax * 0.97))
    kline_l, kline_r = left.vline(K), right.vline(K)

    tau = ValueTracker(1.0)
    value = always_redraw(lambda: left.curve([(s, fm.bs_price(s, K, tau.get_value(), r, sigma, option_type)) for s in grid],
                                             theme.ACCENT, 5))
    gcurve = always_redraw(lambda: right.curve([(s, fm.bs_greeks(s, K, tau.get_value(), r, sigma, option_type)[greek])
                                                for s in grid], theme.PURPLE, 5))

    def tangent():
        t = max(tau.get_value(), 0.01)
        V = fm.bs_price(K, K, t, r, sigma, option_type)
        d = fm.bs_greeks(K, K, t, r, sigma, option_type)["delta"]
        h = 0.12 * K
        return Line(left.p(K - h, V - d * h), left.p(K + h, V + d * h), color=theme.YELLOW, stroke_width=4)

    tan = always_redraw(tangent)
    mark = always_redraw(lambda: right.dot(K, fm.bs_greeks(K, K, max(tau.get_value(), 0.01), r, sigma, option_type)[greek],
                                           theme.YELLOW, 0.1))
    tau_tag = _live_left(lambda: round(tau.get_value(), 2), lambda k: f"Time left: {k:.2f} years", -5.6, -1.95, 24)
    g_tag = _live_left(lambda: round(fm.bs_greeks(K, K, max(tau.get_value(), 0.01), r, sigma, option_type)[greek], 3),
                       lambda k: f"{greek.title()} at the strike: {k:.3g}", 0.6, -1.95, 24, lambda k: theme.YELLOW)
    note = _note("Black-Scholes: steady volatility, 3% rate. Not advice.", -5.6, -2.3, width=80, size=18)

    _fade_whole(ctx, head, left, right, payoff, payoff_tag, kline_l, kline_r, value, gcurve, tan, mark, tau_tag, g_tag, note)
    ctx.add(value, gcurve, tan, mark, tau_tag, g_tag)
    ctx.play(tau.animate.set_value(0.03), run_time=_main(ctx), rate_func=SMOOTH)
    ctx.finish()


# =====================================================================================================================
# 3. GARCH volatility clustering
# =====================================================================================================================
_LR_VAR = 1e-4   # a 1% daily volatility


def _garch_params(alpha: float, beta: float) -> tuple[float, float, float]:
    a = _clamp(alpha, 0.01, 0.3)
    b = _clamp(beta, 0.3, 0.97)
    if a + b > 0.99:
        b = 0.99 - a
    return _LR_VAR * (1 - a - b), a, b


def play_garch_volatility(ctx: Ctx, view: str = "bands", alpha: float = 0.1, beta: float = 0.85, shock_at: int = 30) -> None:
    omega, a, b = _garch_params(alpha, beta)
    persistence = a + b
    long_run = math.sqrt(_LR_VAR) * 100
    S = ValueTracker(0.0)
    if view == "shock":
        head = ctx.heading("After a shock, volatility fades slowly")
        n = 40
        shock_ret = math.sqrt(_LR_VAR) * 4.0
        v0 = omega + a * shock_ret ** 2 + b * _LR_VAR
        series = [math.sqrt(v) * 100 for v in fm.variance_decay(v0, _LR_VAR, persistence, n)]
        yr = (0.0, max(series) * 1.2)
        plot = Plot((0, n), yr, width=10.2, height=3.3, x_label="Days after the shock", y_label="Daily volatility (%)",
                    x_fmt="{:.0f}", y_fmt="{:.1f}").move_to((0, 0.45, 0))
        curve = plot.curve([(k, v) for k, v in enumerate(series)], theme.ACCENT, 5)
        base = plot.hline(long_run, theme.YELLOW)
        base_tag = label(f"long-run level {long_run:.1f}%", 20, theme.YELLOW).next_to(plot.p(n, long_run), DOWN, buff=0.15).shift(LEFT * 1.5)
        half = math.log(0.5) / math.log(persistence)
        note = _note(f"Persistence alpha + beta = {persistence:.2f}: about {half:.0f} days to lose half the extra.", -5.6, -2.05, width=80, size=22, color=theme.FG)
        dot = always_redraw(lambda: plot.dot(S.get_value(), series[min(n, int(round(S.get_value())))], theme.YELLOW, 0.12))
        read = _live_left(lambda: min(n, int(round(S.get_value()))),
                          lambda k: f"Day {k}: volatility {series[k]:.2f}%", 1.2, 2.45, 24, lambda k: theme.YELLOW)
        _fade_whole(ctx, head, plot, curve, base, base_tag, note, dot, read)
        ctx.add(dot, read)
        ctx.play(S.animate.set_value(n), run_time=_main(ctx), rate_func=LINEAR)
        ctx.finish()
        return

    head = ctx.heading("Volatility clusters: calm, then stormy")
    n = 90
    shock_at = int(_clamp(shock_at, 5, n - 10))
    rets, vars_ = fm.garch11(omega, a, b, n, seed=7, shock_at=shock_at, shock=4.0)
    r_pct = [x * 100 for x in rets]
    band = [2 * math.sqrt(v) * 100 for v in vars_]
    ymax = max(max(abs(x) for x in r_pct), max(band)) * 1.15
    plot = Plot((0, n), (-ymax, ymax), width=11.0, height=3.2, x_label="Day", y_label="Return (%)",
                x_fmt="{:.0f}", y_fmt="{:+.0f}").move_to((0, 0.3, 0))
    bars = VGroup(*[Line(plot.p(t, 0), plot.p(t, v), color=theme.YELLOW if t == shock_at else theme.ACCENT, stroke_width=3.5)
                    for t, v in enumerate(r_pct)])
    up = [plot.p(t, band[t]) for t in range(n)]
    dn = [plot.p(t, -band[t]) for t in range(n)]
    area = Polygon(*(up + dn[::-1]), color=theme.PURPLE, fill_color=theme.PURPLE, fill_opacity=0.22, stroke_width=2)
    legend = VGroup(_left_text("bars: daily return", -5.6, 2.55, 20, theme.ACCENT),
                    _left_text("band: expected range (2 std devs)", -2.6, 2.55, 20, theme.PURPLE))
    shock_tag = label("big move", 20, theme.YELLOW).next_to(plot.p(shock_at, r_pct[shock_at]), UP if r_pct[shock_at] > 0 else DOWN, buff=0.1)
    note = _note(f"GARCH(1,1), alpha {a:.2f}, beta {b:.2f}: the band widens after a big move.", -5.6, -2.25, width=80, size=18)
    cursor = always_redraw(lambda: Line(plot.p(S.get_value(), -ymax), plot.p(S.get_value(), ymax), color=theme.FG, stroke_width=2))
    read = _live_left(lambda: min(n - 1, int(S.get_value())),
                      lambda k: f"Day {k}: volatility {math.sqrt(vars_[k]) * 100:.2f}%", 1.5, 2.1, 20, lambda k: theme.PURPLE)
    _fade_whole(ctx, head, plot, area, bars, legend, shock_tag, note, cursor, read)
    ctx.add(cursor, read)
    ctx.play(S.animate.set_value(n - 1), run_time=_main(ctx), rate_func=LINEAR)
    ctx.finish()


# =====================================================================================================================
# 4. Yield curve shifts
# =====================================================================================================================
_MATS = [0.25, 1.0, 2.0, 5.0, 10.0, 30.0]
_MAT_LABELS = ["3M", "1Y", "2Y", "5Y", "10Y", "30Y"]
_SHIFT_TEXT = {
    "parallel_up": ("Parallel shift up", "Every yield rises by the same amount, so existing bond prices fall."),
    "parallel_down": ("Parallel shift down", "Every yield falls by the same amount, so existing bond prices rise."),
    "steepen": ("Steepening", "Long yields rise more than short ones: the curve gets steeper."),
    "flatten": ("Flattening", "Short yields rise relative to long ones: the curve gets flatter."),
    "invert": ("Inversion", "Short yields end up above long yields, a shape often read as a slowdown worry."),
}


def play_yield_curve_shift(ctx: Ctx, shift: str = "steepen", size_bps: float = 100.0) -> None:
    size = _clamp(size_bps, 10, 250) / 10000.0
    base = [fm.nelson_siegel(m) for m in _MATS]
    new = fm.shift_curve(_MATS, base, shift, size)
    pct0, pct1 = [y * 100 for y in base], [y * 100 for y in new]
    yr = pad_range(min(pct0 + pct1), max(pct0 + pct1), 0.3)
    name, why = _SHIFT_TEXT[shift]
    head = ctx.heading(f"Yield curve: {name.lower()}, {size * 10000:.0f} bps")
    plot = Plot((-0.4, 5.4), yr, width=9.0, height=3.2, x_label="Time to maturity", y_label="Yield (%)",
                x_ticks=list(range(6)), x_tick_labels=_MAT_LABELS, y_fmt="{:.1f}").move_to((0, 0.5, 0))

    def build(vals, color, op=1.0, dashed=False):
        c = plot.curve([(i, v) for i, v in enumerate(vals)], color, 5, op)
        dots = [plot.dot(i, v, color, 0.09) for i, v in enumerate(vals)]
        return VGroup(c, *dots)

    ghost = build(pct0, theme.MUTED, 0.8)
    ghost_tag = label("before", 20, theme.MUTED).next_to(plot.p(5, pct0[-1]), RIGHT, buff=0.1)
    live = build(pct0, theme.ACCENT)
    target = build(pct1, theme.ACCENT)
    tag0 = label(f"Shape: {fm.curve_shape(base)}", 26, theme.FG, weight="BOLD").move_to((3.6, 2.45, 0))
    tag1 = label(f"Shape: {fm.curve_shape(new)}", 26, theme.YELLOW, weight="BOLD").move_to((3.6, 2.45, 0))
    after_tag = label(f"after: {name.lower()}", 20, theme.ACCENT).next_to(plot.p(5, pct1[-1]), RIGHT, buff=0.1)
    why_t = label(why, 22, theme.FG, width=80).move_to((0, -2.0, 0))

    _fade_whole(ctx, head, plot, ghost, ghost_tag, live, tag0)
    ctx.play(Transform(live, target), FadeTransform(tag0, tag1), FadeIn(after_tag), run_time=max(1.2, ctx.remaining * 0.45),
             rate_func=SMOOTH)
    ctx.play(FadeIn(why_t, shift=UP * 0.15), run_time=min(1.0, max(0.5, ctx.remaining * 0.3)))
    ctx.finish()


# =====================================================================================================================
# 5. Bond price against yield, duration and convexity
# =====================================================================================================================
def play_bond_price_yield(ctx: Ctx, view: str = "price_yield", coupon: float = 5.0, maturity: float = 10.0,
                          yield_pct: float = 5.0) -> None:
    cpn = _clamp(coupon, 0.5, 12.0) / 100
    yrs = _clamp(round(maturity), 1, 30)
    y0 = _clamp(yield_pct, 2.0, 10.0) / 100
    face = 100.0
    ylo, yhi = 0.01, 0.12
    ygrid = [ylo + (yhi - ylo) * i / 55 for i in range(56)]
    pr = lambda y: fm.bond_price(face, cpn, y, yrs)
    prices = [pr(y) for y in ygrid]
    yr = pad_range(min(prices), max(prices), 0.08)
    mode = view == "duration_convexity"
    head = ctx.heading("Duration and convexity: the price curve bends" if mode else "Bond price falls as yield rises")
    if head.width > 12.0:
        head.scale_to_fit_width(12.0)
    plot = Plot((ylo * 100, yhi * 100), yr, width=7.6, height=3.4, x_label="Yield (%)", y_label="Bond price",
                x_fmt="{:g}", y_fmt="{:.0f}").move_to((-1.6, 0.4, 0))
    curve = plot.curve([(y * 100, pr(y)) for y in ygrid], theme.ACCENT, 5)
    par = plot.hline(face, theme.MUTED)
    par_tag = label("par 100", 18, theme.MUTED).next_to(plot.p(ylo * 100, face), UP, buff=0.05).shift(RIGHT * 0.5)
    px = 2.55
    static = [head, plot, curve, par, par_tag]
    Y = ValueTracker(ylo if not mode else max(ylo, y0 - 0.03))
    if mode:
        D, C = fm.modified_duration(face, cpn, y0, yrs), fm.convexity(face, cpn, y0, yrs)
        p0 = pr(y0)
        lin = lambda y: p0 * (1 - D * (y - y0))
        quad = lambda y: p0 * (1 - D * (y - y0) + 0.5 * C * (y - y0) ** 2)
        xs = [y0 - 0.05 + 0.10 * i / 40 for i in range(41)]
        xs = [x for x in xs if ylo <= x <= yhi]
        tang = plot.curve([(x * 100, lin(x)) for x in xs], theme.YELLOW, 4)
        para = plot.curve([(x * 100, quad(x)) for x in xs], theme.PURPLE, 4)
        anchor = plot.dot(y0 * 100, p0, theme.FG, 0.09)
        legend = VGroup(_left_text("blue: actual price", px, 2.0, 22, theme.ACCENT),
                        _left_text("yellow: duration only", px, 1.5, 22, theme.YELLOW),
                        _left_text("purple: plus convexity", px, 1.0, 22, theme.PURPLE))
        r1 = _live_left(lambda: round(Y.get_value() * 100, 1), lambda k: f"Yield: {k:.1f}%", px, 0.3, 24)
        r2 = _live_left(lambda: round(pr(Y.get_value()), 1), lambda k: f"Actual: {k:.1f}", px, -0.2, 22, lambda k: theme.ACCENT)
        r3 = _live_left(lambda: round(lin(Y.get_value()), 1), lambda k: f"Duration only: {k:.1f}", px, -0.7, 22, lambda k: theme.YELLOW)
        r4 = _live_left(lambda: round(quad(Y.get_value()), 1), lambda k: f"With convexity: {k:.1f}", px, -1.2, 22, lambda k: theme.PURPLE)
        note = _note(f"Duration {D:.1f} years: yield up 1% cuts the price about {D:.1f}%.", -5.6, -2.3, width=80, size=20)
        static += [tang, para, anchor, legend, note]
        readouts = [r1, r2, r3, r4]
        end = min(yhi, y0 + 0.03)
    else:
        D = fm.modified_duration(face, cpn, y0, yrs)
        r1 = _live_left(lambda: round(Y.get_value() * 100, 1), lambda k: f"Yield: {k:.1f}%", px, 1.6, 26)
        r2 = _live_left(lambda: round(pr(Y.get_value()), 1), lambda k: f"Price: {k:.1f}", px, 1.0, 28, lambda k: theme.YELLOW)
        r3 = _live_left(lambda: "above" if Y.get_value() < cpn - 1e-9 else ("below" if Y.get_value() > cpn + 1e-9 else "at"),
                        lambda k: {"above": "Price above par", "below": "Price below par", "at": "Price at par"}[k], px, 0.4, 22, lambda k: theme.MUTED)
        note = _note(f"{cpn * 100:.1f}% coupon, {yrs:.0f} years, paid twice a year. Price = discounted cash flows.", -5.6, -2.3, width=80, size=20)
        static += [note]
        readouts = [r1, r2, r3]
        end = yhi
    dot = always_redraw(lambda: plot.dot(Y.get_value() * 100, pr(Y.get_value()), theme.YELLOW, 0.12))
    static += [dot] + readouts
    _fade_whole(ctx, *static)
    ctx.add(dot, *readouts)
    ctx.play(Y.animate.set_value(end), run_time=_main(ctx), rate_func=SMOOTH)
    ctx.finish()


# =====================================================================================================================
# 6. Random walk, Brownian motion, geometric Brownian motion
# =====================================================================================================================
def play_brownian_motion(ctx: Ctx, variant: str = "brownian", paths: int = 8, volatility: float = 0.2, drift: float = 0.05) -> None:
    n = int(_clamp(paths, 3, 16))
    sigma = _clamp(volatility, 0.02, 1.0)
    mu = _clamp(drift, -0.5, 0.5)
    if variant == "random_walk":
        steps, title = 40, "Random walk: a coin flip each step"
        draw = fm.random_walk_paths(steps, n, 7)
        many = [p[-1] for p in fm.random_walk_paths(steps, 300, 11)]
        xl, yl, xr, start = "Step", "Position", (0, steps), None
        note = "Each step is +1 or -1 with equal chance."
    elif variant == "gbm":
        steps, title = 120, "Geometric Brownian motion: a model for prices"
        draw = fm.gbm_paths(100.0, mu, sigma, 1.0, steps, n, 7)
        many = [p[-1] for p in fm.gbm_paths(100.0, mu, sigma, 1.0, steps, 300, 11)]
        xl, yl, xr, start = "Time (years)", "Price", (0, 1), 100.0
        note = f"Drift {mu * 100:.0f}% a year, volatility {sigma * 100:.0f}%. Prices stay positive; the spread is skewed."
    else:
        steps, title = 120, "Brownian motion: the random walk with tiny steps"
        draw = fm.brownian_paths(1.0, steps, n, mu, sigma, 7)
        many = [p[-1] for p in fm.brownian_paths(1.0, steps, 300, mu, sigma, 11)]
        xl, yl, xr, start = "Time", "Value", (0, 1), 0.0
        note = f"Drift {mu:g}, volatility {sigma:g}. The spread of outcomes grows like the square root of time."
    srt = sorted(many)
    lo_q, hi_q = srt[int(0.02 * len(srt))], srt[int(0.98 * len(srt)) - 1]
    pv = [v for p in draw for v in p]
    yr = pad_range(min(min(pv), lo_q), max(max(pv), hi_q), 0.08)
    head = ctx.heading(title)
    if head.width > 12.0:
        head.scale_to_fit_width(12.0)
    plot = Plot(xr, yr, width=7.6, height=3.3, x_label=xl, y_label=yl, x_fmt="{:g}", y_fmt="{:.0f}" if variant != "brownian" else "{:.1f}"
                ).move_to((-1.8, 0.45, 0))
    cols = [theme.ACCENT, theme.PURPLE, theme.GOOD, theme.WARN]
    lines = [plot.curve([(xr[0] + (xr[1] - xr[0]) * i / steps, v) for i, v in enumerate(p)], cols[k % 4] if k else theme.YELLOW, 3.5, 0.95)
             for k, p in enumerate(draw)]

    # terminal histogram, sharing the plot's y axis
    bins = 14
    counts = fm.histogram(many, bins, yr[0], yr[1])
    cmax = max(max(counts), 1)
    base_x = plot.p(xr[1], yr[0])[0] + 0.3
    bar_len = 2.2
    bars = VGroup()
    for k, c in enumerate(counts):
        if c == 0:
            continue
        y_lo, y_hi = yr[0] + (yr[1] - yr[0]) * k / bins, yr[0] + (yr[1] - yr[0]) * (k + 1) / bins
        h = plot.p(xr[1], y_hi)[1] - plot.p(xr[1], y_lo)[1]
        r = Rectangle(width=bar_len * c / cmax, height=h * 0.92, stroke_width=1.5, color=theme.PURPLE, fill_color=theme.PURPLE, fill_opacity=0.55)
        r.move_to((base_x + r.width / 2, (plot.p(xr[1], y_lo)[1] + plot.p(xr[1], y_hi)[1]) / 2, 0))
        bars.add(r)
    hist_tag = label("300 runs: where they end", 20, theme.PURPLE, width=14).move_to((base_x + 1.1, -1.55, 0))
    start_line = plot.hline(start, theme.MUTED) if start is not None else VGroup()
    foot = _note(note, -5.6, -2.3, width=84, size=18)

    _fade_whole(ctx, head, plot, start_line, foot)
    ctx.play(LaggedStart(*[Create(l) for l in lines], lag_ratio=0.12), run_time=max(1.5, ctx.remaining * 0.55))
    ctx.play(LaggedStart(*[GrowFromEdge(b, LEFT) for b in bars], lag_ratio=0.08), FadeIn(hist_tag),
             run_time=_main(ctx))
    ctx.finish()


# =====================================================================================================================
# 7. Binomial tree and backward induction
# =====================================================================================================================
def play_binomial_tree(ctx: Ctx, steps: int = 3, up: float = 1.1, down: float = 0.9, rate: float = 0.02,
                       strike: float = 100.0) -> None:
    n = int(_clamp(steps, 2, 4))
    u = _clamp(up, 1.02, 1.5)
    d = _clamp(down, 0.5, 0.98)
    K = _clamp(strike, 50, 200)
    # keep the up-probability inside (0.05, 0.95) so the tree has no arbitrage
    lo_r, hi_r = math.log(d + 0.05 * (u - d)), math.log(d + 0.95 * (u - d))
    r = _clamp(rate, lo_r, hi_r)
    p = fm.risk_neutral_p(u, d, r, 1.0)
    tree = fm.crr_tree(100.0, u, d, n)
    vals = fm.tree_values(tree, K, "call", p, r, 1.0)

    head = ctx.heading("Binomial tree: pricing from the end backward")
    if head.width > 12.0:
        head.scale_to_fit_width(12.0)
    dx = min(2.4, 9.0 / n)
    h = min(1.15, 3.4 / n)
    x0 = -dx * n / 2 - 0.3

    def pos(i: int, j: int):
        return (x0 + dx * i, 0.45 + (2 * j - i) * h / 2, 0)

    nodes, edges, vtxt = {}, VGroup(), {}
    for i in range(n + 1):
        for j in range(i + 1):
            box = RoundedRectangle(corner_radius=0.08, width=1.15, height=0.46, color=theme.MUTED, stroke_width=2,
                                   fill_color=theme.PANEL, fill_opacity=1).move_to(pos(i, j))
            t = label(f"{tree[i][j]:.1f}", 22, theme.FG).move_to(box)
            nodes[(i, j)] = VGroup(box, t)
            v = Text(f"{vals[i][j]:.2f}", font_size=22, color=theme.YELLOW, weight="BOLD")
            v.move_to((pos(i, j)[0], pos(i, j)[1] - 0.4, 0))
            vtxt[(i, j)] = v
    for i in range(n):
        for j in range(i + 1):
            for dj in (0, 1):
                a, b = pos(i, j), pos(i + 1, j + dj)
                edges.add(Line((a[0] + 0.58, a[1], 0), (b[0] - 0.58, b[1], 0), color=theme.MUTED, stroke_width=2))
    up_tag = label(f"up x{u:g}", 20, theme.MUTED).move_to((x0 + dx / 2 - 0.55, pos(0, 0)[1] + h / 4 + 0.3, 0))
    dn_tag = label(f"down x{d:g}", 20, theme.MUTED).move_to((x0 + dx / 2 - 0.6, pos(0, 0)[1] - h / 4 - 0.3, 0))
    prob = label(f"Chance of up (risk-neutral): p = (e^r - d)/(u - d) = {p:.2f}", 22, theme.ACCENT).move_to((0, 2.7, 0))
    if prob.width > 12:
        prob.scale_to_fit_width(12)
    cap_end = label("End: value = max(price - strike, 0)", 22, theme.FG).move_to((0, -2.05, 0))
    cap_mid = label("Each node: discounted average of the two values to its right", 22, theme.FG).move_to((0, -2.05, 0))
    cap_root = label(f"Option price today: {vals[0][0]:.2f}   (European call, strike {K:g}, rate {r * 100:.1f}% a step)", 22, theme.YELLOW).move_to((0, -2.05, 0))
    if cap_root.width > 12:
        cap_root.scale_to_fit_width(12)

    _fade_whole(ctx, head, edges, *nodes.values(), up_tag, dn_tag, prob, cap_end)
    cols = list(range(n, -1, -1))
    step = Ctx.beat(ctx.remaining - 1.5, len(cols), cap=1.6, floor=0.6)
    for k, i in enumerate(cols):
        anims = [FadeIn(vtxt[(i, j)], shift=LEFT * 0.15) for j in range(i + 1)]
        if k == 1:
            anims.append(FadeTransform(cap_end, cap_mid))
        ctx.play(*anims, run_time=min(step, 1.4))
        ctx.wait(max(0.0, step - 1.4))
    ring = SurroundingRectangle(vtxt[(0, 0)], color=theme.YELLOW, buff=0.1)
    ctx.play(FadeTransform(cap_mid, cap_root), Create(ring), run_time=min(1.2, max(0.6, ctx.remaining)))
    ctx.finish()


# =====================================================================================================================
# 8. Order book matching (price-time priority)
# =====================================================================================================================
_TICK = 0.05
_SCALE = 3.0 / 200.0
_ROW = 0.36


def _book(depth: int) -> fm.Book:
    return fm.Book([(round(100.0 - _TICK * (i + 1), 2), 60 + 30 * i) for i in range(depth)],
                   [(round(100.0 + _TICK * (i + 1), 2), 40 + 30 * i) for i in range(depth)])


def _slots(*books: fm.Book) -> list[float]:
    prices = [p for b in books for p, _ in b.bids + b.asks]
    top, bot = round(max(prices), 2), round(min(prices), 2)
    n = int(round((top - bot) / _TICK))
    return [round(top - _TICK * k, 2) for k in range(n + 1)]


def _ladder(book: fm.Book, slots: list[float], y_top: float, new_at: float | None = None, new_qty: int = 0) -> tuple[dict, VGroup]:
    rows: dict[float, VGroup] = {}
    bid_p = {p: q for p, q in book.bids}
    ask_p = {p: q for p, q in book.asks}
    for k, price in enumerate(slots):
        y = y_top - k * _ROW
        ptxt = Text(f"{price:.2f}", font_size=20, color=theme.MUTED)
        ptxt.move_to((-2.85, y, 0), aligned_edge=RIGHT)
        row = VGroup(ptxt)
        qty, color = (ask_p.get(price), theme.DANGER) if price in ask_p else (bid_p.get(price), theme.GOOD)
        if qty:
            old = qty - (new_qty if (new_at is not None and abs(price - new_at) < 1e-9) else 0)
            x = -2.55
            if old > 0:
                b = Rectangle(width=old * _SCALE, height=_ROW * 0.78, stroke_width=0, fill_color=color, fill_opacity=0.85)
                b.move_to((x + b.width / 2, y, 0))
                row.add(b)
                x += b.width
            if old < qty:
                nb = Rectangle(width=(qty - old) * _SCALE, height=_ROW * 0.78, stroke_width=2, color=theme.YELLOW,
                               fill_color=theme.YELLOW, fill_opacity=0.9)
                nb.move_to((x + nb.width / 2, y, 0))
                row.add(nb)
                x += nb.width
            q = Text(str(qty), font_size=20, color=theme.FG)
            q.move_to((x + 0.12, y, 0), aligned_edge=LEFT)
            row.add(q)
        rows[price] = row
    return rows, VGroup(*rows.values())


def _spread_tag(book: fm.Book, slots: list[float], y_top: float) -> VGroup:
    sp, bb, ba = book.spread(), book.best_bid(), book.best_ask()
    if sp is None:
        return VGroup()
    yb, ya = y_top - slots.index(bb) * _ROW, y_top - slots.index(ba) * _ROW
    t = Text(f"spread {sp:.2f}", font_size=20, color=theme.YELLOW).move_to((1.55, (yb + ya) / 2, 0), aligned_edge=LEFT)
    return VGroup(t, Line((1.45, ya, 0), (1.45, yb, 0), color=theme.YELLOW, stroke_width=2))


def play_order_book_matching(ctx: Ctx, scenario: str = "market_buy", depth: int = 4) -> None:
    depth = int(_clamp(depth, 3, 5))
    book = _book(depth)
    best_ask, best_bid = book.best_ask(), book.best_bid()
    if scenario == "partial_fill":
        order = fm.Order("buy", round(best_ask + _TICK, 2), 150)
        tag = f"Your order: BUY 150, limit {order.price:.2f}"
    elif scenario == "limit_joins_queue":
        order = fm.Order("buy", best_bid, 50)
        tag = f"Your order: BUY 50, limit {order.price:.2f}"
    else:
        order = fm.Order("buy", None, 90)
        tag = "Your order: BUY 90 at the market price"
    trades, after, left = fm.match_order(book, order)
    filled = sum(q for _, q in trades)
    slots = _slots(book, after)
    y_top = 0.45 + (len(slots) - 1) * _ROW / 2
    rests = order.price is not None and left > 0
    rows0, g0 = _ladder(book, slots, y_top)
    rows1, g1 = _ladder(after, slots, y_top, new_at=order.price if rests else None, new_qty=left if rests else 0)

    head = ctx.heading("Order book: best price first, then first in line")
    if head.width > 12.0:
        head.scale_to_fit_width(12.0)
    asks_tag = label("ASKS: sellers", 20, theme.DANGER).move_to((-5.0, y_top, 0))
    bids_tag = label("BIDS: buyers", 20, theme.GOOD).move_to((-5.0, y_top - (len(slots) - 1) * _ROW, 0))
    sp0, sp1 = _spread_tag(book, slots, y_top), _spread_tag(after, slots, y_top)
    order_box = VGroup(RoundedRectangle(corner_radius=0.1, width=2.3, height=1.7, color=theme.YELLOW, stroke_width=3,
                                        fill_color=theme.PANEL, fill_opacity=1),
                       label(tag, 20, theme.YELLOW, width=13)).move_to((-5.0, 0.5, 0))
    order_box[1].move_to(order_box[0])
    tape_head = label("Trades", 24, theme.MUTED).move_to((4.9, 1.9, 0))
    tape = VGroup(*[Text(f"{q} at {p:.2f}", font_size=22, color=theme.FG) for p, q in trades]).arrange(DOWN, buff=0.18)
    if len(trades):
        tape.move_to((4.9, 1.5 - tape.height / 2, 0))
    else:
        tape = label("No trade: the limit price was not reached", 20, theme.MUTED, width=16).move_to((4.9, 1.3, 0))
    if scenario == "partial_fill":
        status = f"Filled {filled} of {order.qty}. The other {left} wait as the new best bid at {order.price:.2f}."
    elif scenario == "limit_joins_queue":
        status = f"No match. It joins the queue at {order.price:.2f}, behind the {dict(book.bids)[order.price]} already waiting."
    else:
        status = f"Filled {filled} of {order.qty}, taking the cheapest sellers first. The best ask moves up."
    status_t = label(status, 22, theme.FG, width=78).move_to((0, -2.2, 0))

    _fade_whole(ctx, head, g0, asks_tag, bids_tag, sp0)
    ctx.play(FadeIn(order_box, shift=RIGHT * 0.3), FadeIn(tape_head), run_time=0.8)
    swap = [Transform(rows0[p], rows1[p]) for p in slots]
    ctx.play(*swap, FadeTransform(sp0, sp1), run_time=max(1.5, ctx.remaining * 0.4), rate_func=SMOOTH)
    ctx.play(FadeIn(tape, shift=DOWN * 0.15), FadeIn(status_t, shift=UP * 0.15), run_time=min(1.0, max(0.5, ctx.remaining * 0.3)))
    ctx.finish()
