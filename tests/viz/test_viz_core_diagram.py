import numpy as np
import pytest

from manimate.viz.core.diagram import Block, Wire, orthogonal_route


def test_orthogonal_route_only_makes_right_angles():
    pts = orthogonal_route((0, 0), (3, 2), via=[(1, 2)])
    for p, q in zip(pts, pts[1:]):
        assert abs(p[0] - q[0]) < 1e-9 or abs(p[1] - q[1]) < 1e-9
    assert np.allclose(pts[0], (0, 0, 0)) and np.allclose(pts[-1], (3, 2, 0))


def test_block_ports_sit_on_its_edges():
    b = Block("Plant", w=2.0, h=1.0)
    assert np.allclose(b.port("right") - b.port("left"), (2.0, 0, 0), atol=1e-6)
    with pytest.raises(ValueError, match="unknown port"):
        b.port("middle")


def test_wire_needs_two_points_and_can_end_in_an_arrow():
    with pytest.raises(ValueError):
        Wire([(0, 0, 0)])
    w = Wire.route((0, 0), (2, 1), arrow=True)
    assert len(w) == 2 and len(Wire.route((0, 0), (2, 1))) == 1
