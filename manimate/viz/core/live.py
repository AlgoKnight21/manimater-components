"""Pictures driven by a ValueTracker: curves, bands, strip charts, sliders and readouts (README section 5.3)."""
from __future__ import annotations

from typing import Callable

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, Axes, Dot, Line, Polygon, Text, Transform, ValueTracker, VGroup, VMobject

from .palette import PALETTE

RESAMPLE_POINTS = 120


def _clean(fn: Callable, *args) -> float:
    v = float(fn(*args))
    return v if np.isfinite(v) else 0.0


class LiveCurve(VGroup):
    """Intuition: a curve you can bend with a slider: it is re-drawn from ``fn(x, t)`` on every frame."""

    def __init__(self, axes: Axes, fn: Callable[[float, float], float], *, tracker: ValueTracker | None = None,
                 x_range: tuple[float, float] | None = None, color: str = PALETTE.blue):
        super().__init__()
        self.axes, self.fn, self.tracker, self.color = axes, fn, tracker, color
        self.x_range = x_range or (axes.x_range[0], axes.x_range[1])
        self._curve = self._build()
        self.add(self._curve)
        self.add_updater(lambda m: m._refresh())

    def _build(self) -> VMobject:
        t = self.tracker.get_value() if self.tracker is not None else 0.0
        lo, hi = self.x_range
        return self.axes.plot(lambda x: _clean(self.fn, x, t), x_range=[lo, hi, (hi - lo) / 100], color=self.color)

    def _refresh(self) -> None:
        self._curve.become(self._build())


class AreaBand(VGroup):
    """Intuition: the space between two curves: a variance band, an envelope, a confidence fan."""

    def __init__(self, axes: Axes, lower_fn: Callable[[float], float], upper_fn: Callable[[float], float], *,
                 x_range: tuple[float, float] | None = None, color: str = PALETTE.teal, opacity: float = 0.2, n: int = 60):
        super().__init__()
        lo, hi = x_range or (axes.x_range[0], axes.x_range[1])
        xs = np.linspace(lo, hi, n)
        top = [axes.c2p(x, _clean(upper_fn, x)) for x in xs]
        bottom = [axes.c2p(x, _clean(lower_fn, x)) for x in xs[::-1]]
        self.add(Polygon(*top, *bottom, stroke_width=0, fill_color=color, fill_opacity=opacity))


def _resample(vm: VMobject, n: int = RESAMPLE_POINTS) -> VMobject:
    pts = [vm.point_from_proportion(i / (n - 1)) for i in range(n)]
    return VMobject().set_points_smoothly(pts).match_style(vm)


def CurveMorph(curve_a: VMobject, curve_b: VMobject, run_time: float = 1.5) -> Transform:
    """Intuition: one curve melting into another. Both are resampled to the same point count so the morph is smooth."""
    a, b = _resample(curve_a), _resample(curve_b)
    curve_a.become(a)
    return Transform(curve_a, b, run_time=run_time)


class ScrollingScope(VGroup):
    """Intuition: an oscilloscope. Time is the tracker; the trace scrolls left as it advances."""

    def __init__(self, signal_fn: Callable[[float], float], *, tracker: ValueTracker | None = None, window: float = 4.0,
                 width: float = 6.0, height: float = 2.0, amplitude: float = 1.0, color: str = PALETTE.green, n: int = 120):
        super().__init__()
        if window <= 0:
            raise ValueError("window must be positive")
        self.fn, self.tracker, self.window, self.n = signal_fn, (tracker if tracker is not None else ValueTracker(0.0)), window, n
        self.width_, self.height_, self.amp = width, height, amplitude
        self.baseline = Line((-width / 2, 0, 0), (width / 2, 0, 0), color=PALETTE.muted, stroke_width=1.5)
        self.add(self.baseline)
        self.trace = VMobject(color=color, stroke_width=3)
        self._refresh()
        self.add(self.trace)
        self.add_updater(lambda m: m._refresh())

    def _refresh(self) -> None:
        t = self.tracker.get_value()
        xs = np.linspace(0, self.window, self.n)
        # Anchor to the baseline so the trace follows the scope when it is moved or scaled (``unit`` = screen length of one local unit).
        a, b = self.baseline.get_start(), self.baseline.get_end()
        unit = float(np.linalg.norm(b - a)) / self.width_
        pts = [a + (b - a) * (u / self.window) + np.array([0.0, 0.5 * self.height_ * unit * _clean(self.fn, t - self.window + u) / self.amp, 0.0])
               for u in xs]
        self.trace.set_points_as_corners(pts)


def _fmt(label: str, value: float, unit: str, decimals: int) -> str:
    return f"{label} = {value:.{decimals}f}{(' ' + unit) if unit else ''}"


class Slider(VGroup):
    """Intuition: a control knob you watch instead of touch: the narrator 'drags' it by animating the tracker."""

    def __init__(self, tracker: ValueTracker, value_range: tuple[float, float], label: str, *, unit: str = "",
                 decimals: int = 2, length: float = 3.0):
        super().__init__()
        lo, hi = value_range
        if hi <= lo:
            raise ValueError("value_range must be (low, high) with high > low")
        self.tracker, self.lo, self.hi, self.label, self.unit, self.decimals = tracker, lo, hi, label, unit, decimals
        self.track = Line((-length / 2, 0, 0), (length / 2, 0, 0), color=PALETTE.muted, stroke_width=4)
        self.knob = Dot(radius=0.11, color=PALETTE.yellow)
        self.text = Text(_fmt(label, tracker.get_value(), unit, decimals), font_size=22, color=PALETTE.fg)
        self.add(self.track, self.knob, self.text)
        self._refresh()
        self.add_updater(lambda m: m._refresh())

    def fraction(self) -> float:
        return min(1.0, max(0.0, (self.tracker.get_value() - self.lo) / (self.hi - self.lo)))

    def _refresh(self) -> None:
        self.knob.move_to(self.track.point_from_proportion(self.fraction()))
        new = Text(_fmt(self.label, self.tracker.get_value(), self.unit, self.decimals), font_size=22, color=PALETTE.fg)
        self.text.become(new.next_to(self.track, UP, buff=0.2))


class LiveValue(VGroup):
    """Intuition: a number on a meter, updated every frame from a tracker or a function."""

    def __init__(self, label: str, source: ValueTracker | Callable[[], float], *, unit: str = "", decimals: int = 2,
                 color: str = PALETTE.yellow):
        super().__init__()
        self.label, self.unit, self.decimals, self.color = label, unit, decimals, color
        self._get = source.get_value if isinstance(source, ValueTracker) else source
        self.text = Text(self._render(), font_size=26, color=color)
        self.add(self.text)
        self.add_updater(lambda m: m._refresh())

    def _render(self) -> str:
        return _fmt(self.label, _clean(self._get), self.unit, self.decimals)

    def _refresh(self) -> None:
        self.text.become(Text(self._render(), font_size=26, color=self.color).move_to(self.text))
