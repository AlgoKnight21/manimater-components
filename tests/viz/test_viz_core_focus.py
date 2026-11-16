from manim import Circle, Square, VGroup

from manimate.viz.core.focus import MAX_FADE_IN, FocusRing, Spotlight, fade_in_whole


class FakeCtx:
    def __init__(self):
        self.calls = []

    def play(self, *anims, run_time=1.0, **kw):
        self.calls.append((anims, run_time))


def test_ring_surrounds_its_target():
    sq = Square(side_length=2)
    ring = FocusRing(sq)
    assert ring.width > sq.width and ring.height > sq.height
    assert FocusRing(sq, pulse=False).animate_in().__class__.__name__ == "Create"


def test_spotlight_dims_others_and_restores_them():
    a, b, c = Square(), Circle(), Circle()
    c.set_fill(opacity=1.0)
    spot = Spotlight(a, [a, b, c], dim=0.2)
    assert a not in spot.others
    anim = spot.animate_on()
    anim.begin(); anim.interpolate(1); anim.finish()
    assert abs(c.get_fill_opacity() - 0.2) < 1e-6
    back = spot.animate_off()
    back.begin(); back.interpolate(1); back.finish()
    assert abs(c.get_fill_opacity() - 1.0) < 1e-6


def test_fade_in_whole_is_capped_at_two_seconds():
    ctx, g = FakeCtx(), VGroup(Square())
    assert fade_in_whole(ctx, g, run_time=9) is g
    assert ctx.calls[0][1] == MAX_FADE_IN
