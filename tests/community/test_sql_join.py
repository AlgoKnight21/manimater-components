from manimate.components.community.sql_join import _compute_join


LEFT = [
    {"id": 1, "name": "Alice"},
    {"id": 2, "name": "Bob"},
    {"id": 3, "name": "Carol"},
]

RIGHT = [
    {"id": 1, "department": "Engineering"},
    {"id": 2, "department": "Sales"},
    {"id": 4, "department": "Finance"},
]


def test_inner_join_returns_only_matching_rows():
    result = _compute_join(LEFT, RIGHT, "id", "inner")

    assert [row["key"] for row in result] == [1, 2]
    assert all(row["left"] is not None for row in result)
    assert all(row["right"] is not None for row in result)


def test_left_join_preserves_unmatched_left_rows():
    result = _compute_join(LEFT, RIGHT, "id", "left")

    assert [row["key"] for row in result] == [1, 2, 3]
    assert result[2]["left"]["name"] == "Carol"
    assert result[2]["right"] is None


def test_right_join_preserves_unmatched_right_rows():
    result = _compute_join(LEFT, RIGHT, "id", "right")

    assert [row["key"] for row in result] == [1, 2, 4]
    assert result[2]["left"] is None
    assert result[2]["right"]["department"] == "Finance"


def test_full_join_preserves_unmatched_rows_on_both_sides():
    result = _compute_join(LEFT, RIGHT, "id", "full")

    assert [row["key"] for row in result] == [1, 2, 3, 4]
    assert result[2]["right"] is None
    assert result[3]["left"] is None


def test_inner_join_with_no_matches_is_empty():
    left = [{"id": 1, "name": "Alice"}]
    right = [{"id": 2, "department": "Sales"}]

    assert _compute_join(left, right, "id", "inner") == []


def test_outer_join_with_no_matches_preserves_required_rows():
    left = [{"id": 1, "name": "Alice"}]
    right = [{"id": 2, "department": "Sales"}]

    left_result = _compute_join(left, right, "id", "left")
    right_result = _compute_join(left, right, "id", "right")
    full_result = _compute_join(left, right, "id", "full")

    assert len(left_result) == 1
    assert left_result[0]["right"] is None

    assert len(right_result) == 1
    assert right_result[0]["left"] is None

    assert len(full_result) == 2
    assert full_result[0]["right"] is None
    assert full_result[1]["left"] is None


def test_duplicate_keys_produce_cartesian_fan_out():
    left = [
        {"id": 1, "name": "Alice"},
        {"id": 1, "name": "Alicia"},
    ]
    right = [
        {"id": 1, "department": "Engineering"},
        {"id": 1, "department": "Research"},
        {"id": 1, "department": "Design"},
    ]

    result = _compute_join(left, right, "id", "inner")

    assert len(result) == 6
    assert {
        (row["left"]["name"], row["right"]["department"])
        for row in result
    } == {
        ("Alice", "Engineering"),
        ("Alice", "Research"),
        ("Alice", "Design"),
        ("Alicia", "Engineering"),
        ("Alicia", "Research"),
        ("Alicia", "Design"),
    }


def test_null_join_keys_do_not_match_each_other():
    left = [{"id": None, "name": "Unknown"}]
    right = [{"id": None, "department": "Unknown"}]

    assert _compute_join(left, right, "id", "inner") == []

    result = _compute_join(left, right, "id", "full")
    assert len(result) == 2
    assert result[0]["right"] is None
    assert result[1]["left"] is None


def test_empty_left_table():
    result = _compute_join([], RIGHT, "id", "full")

    assert len(result) == len(RIGHT)
    assert all(row["left"] is None for row in result)


def test_empty_right_table():
    result = _compute_join(LEFT, [], "id", "full")

    assert len(result) == len(LEFT)
    assert all(row["right"] is None for row in result)


def test_rows_missing_join_key_are_ignored():
    left = [
        {"id": 1, "name": "Alice"},
        {"name": "Invalid"},
    ]
    right = [
        {"id": 1, "department": "Engineering"},
        {"department": "Invalid"},
    ]

    result = _compute_join(left, right, "id", "inner")

    assert len(result) == 1
    assert result[0]["key"] == 1


def test_input_rows_are_not_mutated():
    left = [{"id": 1, "name": "Alice"}]
    right = [{"id": 1, "department": "Engineering"}]

    left_before = [dict(row) for row in left]
    right_before = [dict(row) for row in right]

    _compute_join(left, right, "id", "full")

    assert left == left_before
    assert right == right_before