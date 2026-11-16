"""Small drawing helpers shared by the signals pictures: axes, pole and zero markers, clipped polylines, hidden-then-revealed marks."""
from __future__ import annotations

import numpy as np
from manim import Axes, Circle, Line, Rectangle, VGroup, VMobject

from ..core.palette import PALETTE


def plain_axes(x_range, y_range, width: float, height: float) -> Axes:
    """Axes with no ticks, tips or numbers: labels are added by hand as short plain words."""
    return Axes(x_range=list(x_range), y_range=list(y_range), x_length=width, y_length=height, tips=False,
                axis_config={"color": PALETTE.muted, "stroke_width": 2, "include_ticks": False, "include_numbers": False})


def frame_for(axes: Axes, width: float, height: float) -> Rectangle:
    """A thin box round the plot area of ``axes`` (its x and y ranges), so the data always sits inside a visible frame."""
    cx = 0.5 * (axes.x_range[0] + axes.x_range[1])
    cy = 0.5 * (axes.y_range[0] + axes.y_range[1])
    return Rectangle(width=width, height=height, stroke_color=PALETTE.muted, stroke_width=1.5, fill_opacity=0).move_to(axes.c2p(cx, cy))


def pole_marker(point=(0.0, 0.0, 0.0), *, color: str = PALETTE.yellow, size: float = 0.11) -> VGroup:
    """The cross that marks a pole. Poles are crosses and zeros are circles, so the two never rely on colour."""
    c = np.array(point, dtype=float)
    return VGroup(Line(c + (-size, -size, 0), c + (size, size, 0), color=color, stroke_width=3.5),
                  Line(c + (-size, size, 0), c + (size, -size, 0), color=color, stroke_width=3.5))


def zero_marker(point=(0.0, 0.0, 0.0), *, color: str = PALETTE.teal, radius: float = 0.09) -> Circle:
    return Circle(radius=radius, stroke_color=color, stroke_width=3.5, fill_opacity=0).move_to(np.array(point, dtype=float))


def polyline(points, *, color: str = PALETTE.blue, width: float = 4.0) -> VMobject:
    return VMobject(stroke_color=color, stroke_width=width, fill_opacity=0).set_points_as_corners([np.array(p, dtype=float) for p in points])


def runs_inside(z: np.ndarray, half_w: float, half_h: float) -> list[np.ndarray]:
    """Split a complex path into the contiguous runs that lie inside ``|re| <= half_w`` and ``|im| <= half_h`` (clipping, not clamping)."""
    inside = (np.abs(z.real) <= half_w) & (np.abs(z.imag) <= half_h)
    runs, start = [], None
    for i, ok in enumerate(inside):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            runs.append(z[start:i])
            start = None
    if start is not None:
        runs.append(z[start:])
    return [r for r in runs if len(r) >= 2]


def hide(*mobjects) -> None:
    """Make marks invisible but present, so the whole frame can fade in at once and the marks appear later with ``reveal``."""
    for m in mobjects:
        m.set_opacity(0.0)


def reveal(group, run_time: float = 1.0):
    """Animation that brings hidden marks to full opacity."""
    return group.animate.set_opacity(1.0).build().set_run_time(run_time)


def match_scale(new, frame, nominal_width: float):
    """Scale text rebuilt inside an updater to match its group, which may have been scaled since it was built (``frame`` is a
    Mobject that was ``nominal_width`` wide when the group was made)."""
    return new.scale(frame.width / nominal_width)
