import pytest
from manim import Circle, FadeIn, Square

from manimate.ctx import Ctx
from manimate.viz.core.focus import fade_in_whole
from manimate.viz.core.plumbing import Beats, RecordingCtx, VizScene, dry_run


def test_beats_share_the_remaining_time_and_respect_floor_and_cap():
    ctx = dry_run(lambda c: None, duration=10.0)
    b = Beats(ctx, 4, cap=1.0, floor=0.3)
    assert [b.next() for _ in range(2)] == [1.0, 1.0]
    short = Beats(dry_run(lambda c: c.wait(9.9), duration=10.0), 4)
    assert short.next() == 0.3
    with pytest.raises(ValueError):
        Beats(ctx, 0)


def test_dry_run_records_animations_time_and_stage_fit():
    def script(ctx):
        sq = Square()
        fade_in_whole(ctx, sq, 1.0)
        ctx.finish()
    ctx = dry_run(script, duration=4.0)
    assert isinstance(ctx, RecordingCtx) and "FadeIn" in ctx.animation_names
    assert abs(ctx.elapsed - 4.0) <= 0.25 and ctx.fade_in_time() == 1.0 and ctx.on_stage()


def test_on_stage_catches_something_off_screen():
    def script(ctx):
        ctx.play(FadeIn(Circle().move_to((9, 0, 0))), run_time=0.5)
    assert dry_run(script).on_stage() is False


def test_vizscene_requires_a_script_override():
    assert issubclass(VizScene, object) and VizScene.duration == 20.0
    with pytest.raises(NotImplementedError):
        VizScene.script(VizScene(), Ctx.__new__(Ctx))


def test_dry_run_puts_the_global_manim_config_back():
    from manim import config
    before = (config.dry_run, config.progress_bar)
    dry_run(lambda c: None)
    assert (config.dry_run, config.progress_bar) == before
