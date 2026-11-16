import pytest

from manimate.safe_expr import compile_expr, validate_expr


def test_multi_variable_expressions_evaluate_in_argument_order():
    f = compile_expr("x**2 + 2*y", variables=("x", "y"))
    assert f(3, 1) == 11.0
    g = compile_expr("-y - 0.1*x", variables=("x", "y"))
    assert g(1.0, 2.0) == pytest.approx(-2.1)


def test_unlisted_names_and_bad_variable_lists_are_rejected():
    with pytest.raises(ValueError, match="unsupported"):
        compile_expr("x + y")  # y is not a variable here
    with pytest.raises(ValueError, match="variables"):
        compile_expr("x", variables=("x", "x"))
    with pytest.raises(ValueError, match="variables"):
        compile_expr("a", variables=("a",))


def test_single_variable_behaviour_is_unchanged():
    assert compile_expr("y = 2*x + 1")(3) == 7.0
    assert validate_expr("__import__('os')") is not None
    assert compile_expr("1/x")(0) == 0.0
