"""The 'look here' devices: one ring, one spotlight, and the single allowed way to bring a static frame in."""
from __future__ import annotations

from manim import AnimationGroup, Create, FadeIn, Indicate, Mobject, Restore, SurroundingRectangle, VGroup

from .palette import PALETTE

MAX_FADE_IN = 2.0  # README section 13: the opening fade may not eat more than 2 s of the clip


class FocusRing(VGroup):
    """Intuition: a ring drawn round the one thing the narrator is talking about."""

    def __init__(self, target: Mobject, *, color: str = PALETTE.yellow, pulse: bool = True, buff: float = 0.12):
        super().__init__()
        self.target, self.pulse_on_show = target, pulse
        self.add(SurroundingRectangle(target, color=color, buff=buff, corner_radius=0.1, stroke_width=4))

    def animate_in(self, run_time: float = 0.6):
        """Draw the ring, then pulse it once if ``pulse`` was set."""
        draw = Create(self[0], run_time=run_time)
        if not self.pulse_on_show:
            return draw
        return AnimationGroup(draw, Indicate(self[0], scale_factor=1.12, color=self[0].get_color(),
                                             run_time=run_time * 2), lag_ratio=1.0)


class Spotlight:
    """Intuition: stage lights. Everything except the target dims; ``animate_off`` brings the lights back up."""

    def __init__(self, target: Mobject, others: list[Mobject], *, dim: float = 0.25):
        if not 0.0 <= dim <= 1.0:
            raise ValueError("dim must be between 0 and 1")
        self.target, self.others, self.dim = target, [o for o in others if o is not target], dim

    def animate_on(self, run_time: float = 0.6):
        for o in self.others:
            o.save_state()
        return AnimationGroup(*(o.animate.set_opacity(self.dim) for o in self.others), run_time=run_time)

    def animate_off(self, run_time: float = 0.6):
        return AnimationGroup(*(Restore(o) for o in self.others), run_time=run_time)


def fade_in_whole(ctx, mob: Mobject, run_time: float = 1.3) -> Mobject:
    """Intuition: the curtain rises on a finished picture, all at once. One FadeIn, never part-by-part (README rule 4.2.3).

    Returns ``mob`` for chaining.
    """
    mob.suspend_updating()  # live parts rebuild their text every frame, which would break the fade's copy of them
    ctx.play(FadeIn(mob), run_time=min(max(run_time, 0.1), MAX_FADE_IN))
    mob.resume_updating()
    return mob
