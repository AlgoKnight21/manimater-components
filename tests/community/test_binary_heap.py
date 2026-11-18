import math
import pytest

from manimate.components.community.binary_heap import _build_events, _is_heap, _values_ok, _count_ok


@pytest.mark.parametrize(("kind", "expected"), [
    ("min", [1, 3, 5, 7, 2, 4]),
    ("max", [7, 3, 5, 1, 2, 4]),
])
def test_builds_valid_heap(kind, expected):
    _, extracted, heap = _build_events([7, 3, 5, 1, 2, 4], 0, kind)
    assert _is_heap(heap, kind)
    assert extracted == []
    assert sorted(heap) == [1, 2, 3, 4, 5, 7]


@pytest.mark.parametrize(("kind", "expected"), [("min", [1, 2, 3]), ("max", [7, 5, 4])])
def test_extracts_in_priority_order(kind, expected):
    _, extracted, heap = _build_events([7, 3, 5, 1, 2, 4], 3, kind)
    assert extracted == expected
    assert _is_heap(heap, kind)


def test_duplicates_do_not_swap_unnecessarily():
    events, extracted, heap = _build_events([4, 4, 4, 4], 2, "min")
    assert extracted == [4, 4]
    assert _is_heap(heap, "min")
    assert all(e["swapped"] is None for e in events if e["phase"] == "sift-up")


def test_negative_and_zero_values():
    _, extracted, heap = _build_events([0, -5, 3, -2, 0], 3, "min")
    assert extracted == [-5, -2, 0]
    assert _is_heap(heap, "min")


def test_empty_input():
    events, extracted, heap = _build_events([], 10, "min")
    assert (events, extracted, heap) == ([], [], [])


def test_extraction_count_clamps_to_heap_size():
    _, extracted, heap = _build_events([3, 1], 99, "min")
    assert extracted == [1, 3]
    assert heap == []


def test_single_element_extraction():
    _, extracted, heap = _build_events([-12], 1, "max")
    assert extracted == [-12]
    assert heap == []


def test_zero_extractions_preserves_heap():
    _, extracted, heap = _build_events([9, 2, 6, 1], 0, "min")
    assert extracted == []
    assert _is_heap(heap, "min")


def test_equal_children_are_compared_together():
    events, _, _ = _build_events([1, 3, 3, 8, 9, 3], 1, "min")
    down = [e for e in events if e["phase"] == "sift-down"]
    assert down and down[0]["compared"] == [0, 1, 2]


def test_input_is_not_mutated():
    values = [8, -1, 3, 3]
    original = values.copy()
    _build_events(values, 2, "min")
    assert values == original


@pytest.mark.parametrize("values", [[math.nan], [math.inf], [-math.inf], [10000], [-10000]])
def test_rejects_non_finite_or_oversized_values(values):
    assert _values_ok(values) is not None


def test_accepts_empty_and_negative_values():
    assert _values_ok([]) is None
    assert _values_ok([-9999, 0, 9999]) is None


def test_extraction_count_validation():
    assert _count_ok(-1) is not None
    assert _count_ok(0) is None
    assert _count_ok(100) is None
