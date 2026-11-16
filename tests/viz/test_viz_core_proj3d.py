import numpy as np
import pytest

from manimate.viz.core.proj3d import MAX_SURFACE_RES, Axes2p5D, Path3D, Projector3D, Sphere2p5D, Surface2p5D


def test_projection_with_no_rotation_is_x_z_screen_and_y_depth():
    p = Projector3D(yaw=0, pitch=0).project([[1.0, 2.0, 3.0]])
    assert np.allclose(p, [[1.0, 3.0, 2.0]])


def test_projection_is_a_pure_rotation_so_lengths_survive():
    pts = np.random.default_rng(1).normal(size=(5, 3))
    out = Projector3D(yaw=-40, pitch=30).project(pts)
    assert np.allclose(np.linalg.norm(out, axis=1), np.linalg.norm(pts, axis=1))


def test_surface_builds_depth_sorted_quads_and_caps_resolution():
    s = Surface2p5D(lambda u, v: u * u + v * v, resolution=(6, 5))
    assert len(s) == 5 * 4
    with pytest.raises(ValueError, match="resolution"):
        Surface2p5D(lambda u, v: 0.0, resolution=(MAX_SURFACE_RES + 1, 4))
    with pytest.raises(ValueError, match="shape"):
        Surface2p5D(np.zeros((3, 3)), resolution=(6, 5))


def test_surface_rotation_and_morph_return_animations_that_rebuild():
    s = Surface2p5D(lambda u, v: u, resolution=(4, 4))
    a = s.animate_rotate(10, run_time=1)
    a.begin(); a.interpolate(1); a.finish()
    assert len(s) == 9 and s.projector.yaw == 10
    m = s.animate_morph_to(np.ones((4, 4)))
    m.begin(); m.interpolate(1); m.finish()
    assert len(s) == 9


def test_path_axes_and_sphere_build():
    assert len(Path3D([[0, 0, 0], [1, 1, 1]])) == 1
    with pytest.raises(ValueError):
        Path3D([[0, 0, 0]])
    assert len(Axes2p5D()) == 6 and len(Sphere2p5D()) > 10
