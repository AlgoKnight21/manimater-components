import math

import pytest

from manimate import theme
from manimate.viz.core.palette import (COLORMAPS, PALETTE, colormap, contrast_ratio, hex_to_rgb, lerp_color,
                                       phase_to_color, relative_luminance)


def test_palette_reuses_theme_and_adds_teal():
    assert PALETTE.blue == theme.ACCENT and PALETTE.yellow == theme.YELLOW and PALETTE.red == theme.DANGER
    assert PALETTE.bg == "#000000" and PALETTE.teal.startswith("#")


@pytest.mark.parametrize("color", PALETTE.text_colors)
def test_every_text_colour_is_readable_on_black(color):
    assert contrast_ratio(color, PALETTE.bg) >= 4.5


def test_contrast_extremes():
    assert contrast_ratio("#ffffff", "#000000") == pytest.approx(21.0)
    assert contrast_ratio("#123456", "#123456") == pytest.approx(1.0)


def test_lerp_color_ends_midpoint_and_clamp():
    assert lerp_color("#000000", "#ffffff", 0) == "#000000"
    assert lerp_color("#000000", "#ffffff", 1) == "#ffffff"
    assert lerp_color("#000000", "#ffffff", 0.5) == "#808080"
    assert lerp_color("#000000", "#ffffff", 7) == "#ffffff" and lerp_color("#000000", "#ffffff", -3) == "#000000"


def test_hex_validation():
    with pytest.raises(ValueError):
        hex_to_rgb("#fff")


@pytest.mark.parametrize("name", COLORMAPS)
def test_colormap_clamps_and_is_visible_on_black(name):
    lo, hi = colormap(name, -5, 0, 1), colormap(name, 5, 0, 1)
    assert lo == colormap(name, 0, 0, 1) and hi == colormap(name, 1, 0, 1)
    assert relative_luminance(lo) > 0.01 and relative_luminance(hi) > 0.01   # the "zero" end must not vanish into the bg


@pytest.mark.parametrize("name", ["viridis", "magma", "blue_yellow"])
def test_sequential_maps_get_brighter(name):
    lum = [relative_luminance(colormap(name, i / 10)) for i in range(11)]
    assert all(b >= a - 1e-9 for a, b in zip(lum, lum[1:]))


def test_colormap_edge_cases():
    assert colormap("viridis", float("nan")) == PALETTE.muted
    assert colormap("viridis", 3, 3, 3) == colormap("viridis", 0.5)         # zero-width range does not divide by zero
    with pytest.raises(ValueError, match="unknown colormap"):
        colormap("rainbow", 0.5)


def test_phase_colours_wrap_and_differ():
    assert phase_to_color(0) == phase_to_color(2 * math.pi)
    assert len({phase_to_color(k * math.pi / 2) for k in range(4)}) == 4
