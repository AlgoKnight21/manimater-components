import numpy as np
import pytest
from manim import Axes, Line, ValueTracker

from manimate.viz.core.live import AreaBand, CurveMorph, LiveCurve, LiveValue, ScrollingScope, Slider


def axes():
    return Axes(x_range=[0, 4, 1], y_range=[0, 4, 1], x_length=4, y_length=4)


def test_live_curve_redraws_when_the_tracker_moves():
    ax, t = axes(), ValueTracker(0.0)
    c = LiveCurve(ax, lambda x, t: t * x, tracker=t)
    flat_end = c._curve.get_end().copy()
    t.set_value(1.0)
    c.update(0)
    assert c._curve.get_end()[1] > flat_end[1]


def test_area_band_stays_between_its_functions():
    ax = axes()
    band = AreaBand(ax, lambda x: 1.0, lambda x: 2.0)
    ys = [p[1] for p in band[0].get_vertices()]
    assert min(ys) == pytest.approx(ax.c2p(0, 1)[1]) and max(ys) == pytest.approx(ax.c2p(0, 2)[1])


def test_curve_morph_resamples_both_curves():
    ax = axes()
    a, b = ax.plot(lambda x: x), ax.plot(lambda x: 4 - x)
    anim = CurveMorph(a, b, run_time=2)
    assert anim.run_time == 2 and len(a.points) == len(anim.target_mobject.points)


def test_scrolling_scope_trace_moves_with_time():
    t = ValueTracker(0.0)
    s = ScrollingScope(np.sin, tracker=t)
    before = s.trace.points.copy()
    t.set_value(1.3)
    s.update(0)
    assert not np.allclose(before, s.trace.points)
    with pytest.raises(ValueError):
        ScrollingScope(np.sin, window=0)


def test_slider_knob_follows_tracker_and_checks_range():
    t = ValueTracker(0.0)
    s = Slider(t, (0, 10), "gain")
    t.set_value(5)
    s.update(0)
    assert s.knob.get_center()[0] == pytest.approx(0.0, abs=1e-6)
    with pytest.raises(ValueError):
        Slider(t, (1, 1), "x")


def test_live_value_text_changes_with_the_source():
    t = ValueTracker(1.0)
    v = LiveValue("V", t, unit="V", decimals=1)
    w0 = v.text.width
    t.set_value(1234.5)
    v.update(0)
    assert v.text.width > w0


def test_a_tracker_at_zero_is_still_used():
    # ValueTracker(0.0) is falsy, so `tracker or default` would silently replace it
    t = ValueTracker(0.0)
    assert ScrollingScope(np.sin, tracker=t).tracker is t
