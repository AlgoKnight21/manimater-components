"""Pure maths for the finance visuals. No Manim, no NumPy-only tricks: every function can be called and checked alone.

Models are the simplest textbook ones: Black-Scholes with constant volatility, a CRR binomial tree, GARCH(1,1),
Nelson-Siegel curves, and a price-time-priority limit order book.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


# ---- normal distribution --------------------------------------------------------------------------------------
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


# ---- Black-Scholes ----------------------------------------------------------------------------------------------
def _d1d2(S: float, K: float, tau: float, r: float, sigma: float) -> tuple[float, float]:
    sd = sigma * math.sqrt(tau)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * tau) / sd
    return d1, d1 - sd


def bs_price(S: float, K: float, tau: float, r: float, sigma: float, kind: str = "call") -> float:
    """Black-Scholes price. At or past expiry (or zero volatility) it is the intrinsic value."""
    if tau <= 1e-9 or sigma <= 1e-9 or S <= 0:
        return max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)
    d1, d2 = _d1d2(S, K, tau, r, sigma)
    if kind == "call":
        return S * norm_cdf(d1) - K * math.exp(-r * tau) * norm_cdf(d2)
    return K * math.exp(-r * tau) * norm_cdf(-d2) - S * norm_cdf(-d1)


def bs_greeks(S: float, K: float, tau: float, r: float, sigma: float, kind: str = "call") -> dict[str, float]:
    """delta, gamma, theta (per year), vega (per 1.00 of volatility). Limits at expiry follow the payoff."""
    if tau <= 1e-9 or sigma <= 1e-9 or S <= 0:
        itm = S > K if kind == "call" else S < K
        return {"delta": (1.0 if kind == "call" else -1.0) if itm else 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
    d1, d2 = _d1d2(S, K, tau, r, sigma)
    pdf = norm_pdf(d1)
    delta = norm_cdf(d1) if kind == "call" else norm_cdf(d1) - 1.0
    gamma = pdf / (S * sigma * math.sqrt(tau))
    vega = S * pdf * math.sqrt(tau)
    decay = -S * pdf * sigma / (2.0 * math.sqrt(tau))
    if kind == "call":
        theta = decay - r * K * math.exp(-r * tau) * norm_cdf(d2)
    else:
        theta = decay + r * K * math.exp(-r * tau) * norm_cdf(-d2)
    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}


# ---- payoffs ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Leg:
    """One option position: ``qty`` > 0 is long, < 0 is short. ``premium`` is paid (long) or received (short) per unit."""
    kind: str      # "call" or "put"
    strike: float
    qty: float = 1.0
    premium: float = 0.0


def leg_profit(leg: Leg, S: float) -> float:
    intrinsic = max(S - leg.strike, 0.0) if leg.kind == "call" else max(leg.strike - S, 0.0)
    return leg.qty * (intrinsic - leg.premium)


def strategy_profit(legs: list[Leg], S: float) -> float:
    return sum(leg_profit(l, S) for l in legs)


STRATEGIES = ["single", "straddle", "bull_spread", "collar"]


def strategy_legs(strategy: str, kind: str, strike: float, premium: float, side: str = "long") -> list[Leg]:
    """Preset positions, built around ``strike``. Wing strikes sit 10% away; their premiums are modelled as a
    fraction of the main premium so the picture stays consistent without a second pricing input."""
    sign = 1.0 if side == "long" else -1.0
    wing = 0.1 * strike
    if strategy == "straddle":
        return [Leg("call", strike, sign, premium), Leg("put", strike, sign, premium)]
    if strategy == "bull_spread":
        # buy the lower call, sell the higher call; the higher one is cheaper
        return [Leg("call", strike, sign, premium), Leg("call", strike + wing, -sign, premium * 0.45)]
    if strategy == "collar":
        # hold the stock, buy a put below, sell a call above (stock leg is shown as a profit line, see stock_profit)
        return [Leg("put", strike - wing, sign, premium * 0.6), Leg("call", strike + wing, -sign, premium * 0.6)]
    return [Leg(kind, strike, sign, premium)]


def stock_profit(S: float, S0: float, qty: float = 1.0) -> float:
    return qty * (S - S0)


def profit_curve(legs: list[Leg], lo: float, hi: float, n: int = 120, with_stock_at: float | None = None,
                 stock_qty: float = 1.0) -> list[tuple[float, float]]:
    pts = []
    for i in range(n + 1):
        S = lo + (hi - lo) * i / n
        y = strategy_profit(legs, S)
        if with_stock_at is not None:
            y += stock_profit(S, with_stock_at, stock_qty)
        pts.append((S, y))
    return pts


def break_evens(points: list[tuple[float, float]]) -> list[float]:
    """Where the sampled profit curve crosses zero (linear interpolation between samples)."""
    out: list[float] = []
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if y0 == 0.0 and (not out or abs(out[-1] - x0) > 1e-9):
            out.append(x0)
        elif y0 * y1 < 0:
            out.append(x0 + (x1 - x0) * (-y0) / (y1 - y0))
    if points and points[-1][1] == 0.0 and (not out or abs(out[-1] - points[-1][0]) > 1e-9):
        out.append(points[-1][0])
    return out


# ---- stochastic paths -------------------------------------------------------------------------------------------
def random_walk_paths(steps: int, n: int, seed: int = 7) -> list[list[float]]:
    """``n`` walks of +/-1 steps, each starting at 0 (length steps + 1)."""
    rng = random.Random(seed)
    paths = []
    for _ in range(n):
        p = [0.0]
        for _ in range(steps):
            p.append(p[-1] + (1.0 if rng.random() < 0.5 else -1.0))
        paths.append(p)
    return paths


def brownian_paths(T: float, steps: int, n: int, drift: float = 0.0, vol: float = 1.0, seed: int = 7) -> list[list[float]]:
    rng = random.Random(seed)
    dt = T / steps
    sq = math.sqrt(dt)
    paths = []
    for _ in range(n):
        p = [0.0]
        for _ in range(steps):
            p.append(p[-1] + drift * dt + vol * sq * rng.gauss(0.0, 1.0))
        paths.append(p)
    return paths


def gbm_paths(S0: float, mu: float, sigma: float, T: float, steps: int, n: int, seed: int = 7) -> list[list[float]]:
    """Geometric Brownian motion, exact discretisation: S(t+dt) = S(t) exp((mu - sigma^2/2) dt + sigma sqrt(dt) Z)."""
    rng = random.Random(seed)
    dt = T / steps
    sq = math.sqrt(dt)
    paths = []
    for _ in range(n):
        p = [S0]
        for _ in range(steps):
            p.append(p[-1] * math.exp((mu - 0.5 * sigma * sigma) * dt + sigma * sq * rng.gauss(0.0, 1.0)))
        paths.append(p)
    return paths


def histogram(values: list[float], bins: int, lo: float, hi: float) -> list[int]:
    counts = [0] * bins
    if hi <= lo:
        return counts
    for v in values:
        k = int((v - lo) / (hi - lo) * bins)
        counts[min(bins - 1, max(0, k))] += 1
    return counts


# ---- GARCH(1,1) -------------------------------------------------------------------------------------------------
def unconditional_variance(omega: float, alpha: float, beta: float) -> float:
    return omega / (1.0 - alpha - beta)


def garch11(omega: float, alpha: float, beta: float, n: int, seed: int = 7,
            shock_at: int | None = None, shock: float = 4.0) -> tuple[list[float], list[float]]:
    """Returns (returns, variances). sigma_t^2 = omega + alpha r_{t-1}^2 + beta sigma_{t-1}^2.

    With ``shock_at`` set, the standardised draw at that step is replaced by ``shock`` so the viewer sees one clear jump.
    """
    if alpha < 0 or beta < 0 or omega <= 0 or alpha + beta >= 1:
        raise ValueError("need omega > 0, alpha >= 0, beta >= 0 and alpha + beta < 1")
    rng = random.Random(seed)
    var = unconditional_variance(omega, alpha, beta)
    r_prev = 0.0
    rets: list[float] = []
    vars_: list[float] = []
    for t in range(n):
        if t > 0:
            var = omega + alpha * r_prev * r_prev + beta * vars_[-1]
        z = shock if (shock_at is not None and t == shock_at) else rng.gauss(0.0, 1.0)
        r_prev = math.sqrt(var) * z
        rets.append(r_prev)
        vars_.append(var)
    return rets, vars_


def variance_decay(var_after_shock: float, long_run: float, persistence: float, steps: int) -> list[float]:
    """Expected variance after a shock: it returns to the long-run level geometrically at rate alpha + beta."""
    return [long_run + (var_after_shock - long_run) * persistence ** k for k in range(steps + 1)]


# ---- bonds and curves -------------------------------------------------------------------------------------------
def bond_cashflows(face: float, coupon: float, years: float, freq: int = 2) -> list[tuple[float, float]]:
    """(time in years, cash) for each coupon, with the face value added to the last one."""
    n = max(1, int(round(years * freq)))
    c = face * coupon / freq
    return [((i + 1) / freq, c + (face if i == n - 1 else 0.0)) for i in range(n)]


def bond_price(face: float, coupon: float, y: float, years: float, freq: int = 2) -> float:
    return sum(cf / (1.0 + y / freq) ** (t * freq) for t, cf in bond_cashflows(face, coupon, years, freq))


def macaulay_duration(face: float, coupon: float, y: float, years: float, freq: int = 2) -> float:
    p = bond_price(face, coupon, y, years, freq)
    return sum(t * cf / (1.0 + y / freq) ** (t * freq) for t, cf in bond_cashflows(face, coupon, years, freq)) / p


def modified_duration(face: float, coupon: float, y: float, years: float, freq: int = 2) -> float:
    return macaulay_duration(face, coupon, y, years, freq) / (1.0 + y / freq)


def convexity(face: float, coupon: float, y: float, years: float, freq: int = 2) -> float:
    p = bond_price(face, coupon, y, years, freq)
    s = sum(cf * t * (t + 1.0 / freq) / (1.0 + y / freq) ** (t * freq + 2) for t, cf in bond_cashflows(face, coupon, years, freq))
    return s / p


def price_approx(face: float, coupon: float, y0: float, dy: float, years: float, freq: int = 2) -> tuple[float, float]:
    """(first-order, second-order) price estimates for a yield move of ``dy``."""
    p0 = bond_price(face, coupon, y0, years, freq)
    d = modified_duration(face, coupon, y0, years, freq)
    c = convexity(face, coupon, y0, years, freq)
    first = p0 * (1.0 - d * dy)
    second = p0 * (1.0 - d * dy + 0.5 * c * dy * dy)
    return first, second


def nelson_siegel(tau: float, level: float = 0.045, slope: float = -0.02, curvature: float = 0.01, lam: float = 1.8) -> float:
    """Yield at maturity ``tau`` years (level = long rate, level + slope = short rate)."""
    if tau <= 1e-9:
        return level + slope
    x = tau / lam
    f = (1.0 - math.exp(-x)) / x
    return level + slope * f + curvature * (f - math.exp(-x))


SHIFTS = ["parallel_up", "parallel_down", "steepen", "flatten", "invert"]


def shift_curve(maturities: list[float], yields: list[float], kind: str, size: float) -> list[float]:
    """Move a curve. ``size`` is in decimal yield (0.01 = 100 bps). Pivot for twists is the 5-year point."""
    pivot = 5.0
    out = []
    for m, y in zip(maturities, yields):
        w = (m - pivot) / (max(maturities) - pivot) if m >= pivot else (m - pivot) / (pivot - min(maturities))
        if kind == "parallel_up":
            out.append(y + size)
        elif kind == "parallel_down":
            out.append(y - size)
        elif kind == "steepen":      # long end up, short end down
            out.append(y + size * w)
        elif kind == "flatten":      # long end down, short end up
            out.append(y - size * w)
        elif kind == "invert":       # twist hard enough that the short end ends above the long end
            out.append(y - 2.0 * size * w)
        else:
            raise ValueError(f"unknown shift {kind!r}")
    return out


def curve_shape(yields: list[float]) -> str:
    if yields[0] > yields[-1] + 1e-9:
        return "inverted"
    if abs(yields[-1] - yields[0]) < 0.003:
        return "flat"
    return "normal"


# ---- binomial tree ----------------------------------------------------------------------------------------------
def risk_neutral_p(u: float, d: float, r: float, dt: float) -> float:
    p = (math.exp(r * dt) - d) / (u - d)
    return p


def crr_tree(S0: float, u: float, d: float, n: int) -> list[list[float]]:
    """tree[i][j] = S0 * u^j * d^(i-j): step i, j up-moves (recombining)."""
    return [[S0 * u ** j * d ** (i - j) for j in range(i + 1)] for i in range(n + 1)]


def tree_values(tree: list[list[float]], K: float, kind: str, p: float, r: float, dt: float,
                american: bool = False) -> list[list[float]]:
    """Backward induction. Returns the option value at every node (same shape as ``tree``)."""
    n = len(tree) - 1
    pay = (lambda s: max(s - K, 0.0)) if kind == "call" else (lambda s: max(K - s, 0.0))
    vals: list[list[float]] = [[0.0] * (i + 1) for i in range(n + 1)]
    vals[n] = [pay(s) for s in tree[n]]
    disc = math.exp(-r * dt)
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            cont = disc * (p * vals[i + 1][j + 1] + (1.0 - p) * vals[i + 1][j])
            vals[i][j] = max(cont, pay(tree[i][j])) if american else cont
    return vals


# ---- limit order book -------------------------------------------------------------------------------------------
@dataclass
class Order:
    side: str            # "buy" or "sell"
    price: float | None  # None = market order
    qty: int
    oid: str = ""


@dataclass
class Book:
    bids: list[tuple[float, int]] = field(default_factory=list)  # best (highest) first
    asks: list[tuple[float, int]] = field(default_factory=list)  # best (lowest) first

    def best_bid(self) -> float | None:
        return self.bids[0][0] if self.bids else None

    def best_ask(self) -> float | None:
        return self.asks[0][0] if self.asks else None

    def spread(self) -> float | None:
        b, a = self.best_bid(), self.best_ask()
        return None if b is None or a is None else round(a - b, 10)


def match_order(book: Book, order: Order) -> tuple[list[tuple[float, int]], Book, int]:
    """Match against the book at price-time priority (levels are aggregated, queues are FIFO inside a level).

    Returns (trades as (price, qty), the new book, quantity left unfilled). A leftover limit order rests in the book;
    a leftover market order is dropped (no liquidity). Total traded + left over always equals the order quantity.
    """
    bids = [list(x) for x in book.bids]
    asks = [list(x) for x in book.asks]
    trades: list[tuple[float, int]] = []
    left = order.qty
    if order.side == "buy":
        while left > 0 and asks and (order.price is None or asks[0][0] <= order.price + 1e-12):
            px, size = asks[0]
            q = min(left, size)
            trades.append((px, q))
            left -= q
            asks[0][1] -= q
            if asks[0][1] == 0:
                asks.pop(0)
        if left > 0 and order.price is not None:
            _rest(bids, order.price, left, descending=True)
    else:
        while left > 0 and bids and (order.price is None or bids[0][0] >= order.price - 1e-12):
            px, size = bids[0]
            q = min(left, size)
            trades.append((px, q))
            left -= q
            bids[0][1] -= q
            if bids[0][1] == 0:
                bids.pop(0)
        if left > 0 and order.price is not None:
            _rest(asks, order.price, left, descending=False)
    new_book = Book([(p, q) for p, q in bids], [(p, q) for p, q in asks])
    return trades, new_book, left


def _rest(side: list[list[float]], price: float, qty: int, descending: bool) -> None:
    for lvl in side:
        if abs(lvl[0] - price) < 1e-12:
            lvl[1] += qty
            return
    side.append([price, qty])
    side.sort(key=lambda x: x[0], reverse=descending)


def cumulative_depth(levels: list[tuple[float, int]]) -> list[tuple[float, int]]:
    out, tot = [], 0
    for p, q in levels:
        tot += q
        out.append((p, tot))
    return out
