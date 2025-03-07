from inferplot._utils import _count_n_decimals
import pytest


def test_count_n_decimals():
    assert _count_n_decimals(12.3456) == 4
    assert _count_n_decimals(0.123) == 3
    assert _count_n_decimals(0.0001) == 4
    assert _count_n_decimals(123.456789) == 6
    assert _count_n_decimals(2.0) == 0
    assert _count_n_decimals(100.000) == 0
    assert _count_n_decimals(500) == 0
    assert _count_n_decimals(0.0) == 0
    assert _count_n_decimals(1.0) == 0


def test_count_n_decimals_error():
    with pytest.raises(TypeError):
        _count_n_decimals("1.0")

    with pytest.raises(TypeError):
        _count_n_decimals([1, 2])


if __name__ == "__main__":
    pytest.main()
