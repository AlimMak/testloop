import pytest
import target


def test_add_positive_numbers():
    assert target.add(2, 3) == 5


def test_add_negative_numbers():
    assert target.add(-2, -3) == -5


def test_add_mixed_sign_numbers():
    assert target.add(-2, 3) == 1


def test_add_floats():
    assert target.add(1.5, 2.5) == 4.0


def test_add_zero():
    assert target.add(0, 0) == 0


def test_divide_normal_case():
    assert target.divide(10, 2) == 5


def test_divide_negative_numbers():
    assert target.divide(-10, 2) == -5


def test_divide_floats():
    assert target.divide(1, 4) == 0.25


def test_divide_by_zero_raises():
    with pytest.raises(ValueError, match="cannot divide by zero"):
        target.divide(10, 0)


def test_divide_zero_numerator():
    assert target.divide(0, 5) == 0


def test_clamp_within_range():
    assert target.clamp(5, 1, 10) == 5


def test_clamp_below_lower_bound():
    assert target.clamp(-5, 1, 10) == 1


def test_clamp_above_upper_bound():
    assert target.clamp(15, 1, 10) == 10


def test_clamp_at_lower_boundary():
    assert target.clamp(1, 1, 10) == 1


def test_clamp_at_upper_boundary():
    assert target.clamp(10, 1, 10) == 10


def test_clamp_lo_equals_hi():
    assert target.clamp(5, 3, 3) == 3


def test_clamp_lo_greater_than_hi_raises():
    with pytest.raises(ValueError, match="lo must be <= hi"):
        target.clamp(5, 10, 1)


def test_clamp_with_floats():
    assert target.clamp(2.5, 0.0, 5.0) == 2.5