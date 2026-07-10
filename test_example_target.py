import target
import pytest
def test_add():
    assert target.add(2, 3) == 5
def test_divide_ok():
    assert target.divide(10, 2) == 5
def test_divide_zero():
    with pytest.raises(ValueError):
        target.divide(1, 0)
def test_clamp():
    assert target.clamp(5, 0, 10) == 5
    assert target.clamp(-1, 0, 10) == 0
    assert target.clamp(99, 0, 10) == 10
def test_clamp_bad_bounds():
    with pytest.raises(ValueError):
        target.clamp(5, 10, 0)
