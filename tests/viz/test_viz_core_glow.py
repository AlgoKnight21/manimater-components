import numpy as np
import pytest
from manim import Circle

from manimate.viz.core.glow import Glow, GlowDot


def test_glow_stacks_the_requested_layers_without_touching_the_source():
    src = Circle()
    g = Glow(src, layers=3)
    assert len(g) == 3 and g.source is src
    assert g[0].get_stroke_width() > g[-1].get_stroke_width()  # widest layer is at the back
    assert g[0].get_fill_opacity() == 0


def test_glow_rejects_silly_layer_counts():
    with pytest.raises(ValueError, match="layers"):
        Glow(Circle(), layers=0)


def test_glowdot_is_centred_on_its_point_with_an_opaque_core():
    d = GlowDot((1.0, 2.0, 0.0), 0.1)
    assert np.allclose(d.get_center(), (1.0, 2.0, 0.0), atol=1e-6)
    assert d[-1].get_fill_opacity() == 1.0
    with pytest.raises(ValueError):
        GlowDot(radius=0)
