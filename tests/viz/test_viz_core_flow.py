import numpy as np
import pytest
from manim import Line

from manimate.viz.core.flow import MAX_PARTICLES, FlowField, ParticleField, PulseTrain, pulse_along


def wire():
    return Line((-2, 0, 0), (2, 0, 0))


def test_pulse_along_uses_the_requested_time_and_rejects_zero():
    assert pulse_along(wire(), run_time=2.0).run_time == 2.0
    with pytest.raises(ValueError):
        pulse_along(wire(), run_time=0)


def test_pulse_train_staggers_starts_and_checks_lengths():
    t = PulseTrain([wire(), wire()], [0.0, 0.5], run_time=1.0, reverse=True)
    assert t.run_time == pytest.approx(1.5)
    with pytest.raises(ValueError, match="delays"):
        PulseTrain([wire()], [0.0, 1.0])


def test_flowfield_particles_stay_on_the_wire_while_moving():
    f = FlowField(wire(), n=10)
    before = f.submobjects[0].get_center().copy()
    f.update(0.5)
    after = f.submobjects[0].get_center()
    assert not np.allclose(before, after)
    assert all(abs(d.get_center()[1]) < 1e-6 for d in f.submobjects)


def test_particle_field_is_seeded_bounded_and_capped():
    a, b = ParticleField(20, (4, 2), seed=3), ParticleField(20, (4, 2), seed=3)
    assert np.allclose([p.get_center() for p in a], [p.get_center() for p in b])
    assert all(abs(p.get_center()[0]) <= 2 and abs(p.get_center()[1]) <= 1 for p in a)
    with pytest.raises(ValueError):
        ParticleField(MAX_PARTICLES + 1)
    drifting = ParticleField(5, (4, 2), drift=(10.0, 0.0))
    drifting.update(1.0)
    assert all(abs(p.get_center()[0]) <= 2 for p in drifting)
    assert a.remove_particles([0, 1]) is not None and a.recombine([(2, 3)]) is not None
