import numpy as np
import pytest

from kl_evolution.core.data_objects.serie import Serie


def test_input_all_eq_is_eq():
    input = Serie(values=[1, 1, 1])
    assert input.__all_eq__(other_value=1), "All values should be equal to 1"


def test_input_all_eq_is_not_eq():
    input = Serie(values=[1, 1, 1])
    assert not input.__all_eq__(other_value=2), "All values should not be equal to 2"


def test_input_gets_convert_to_array_of_float():
    input = Serie(values=[1, 2, 3])
    assert all(
        [type(value) == np.float64 for value in input.values]
    ), "Values should be converted to float"


def test_shift_does_shift():
    input = Serie(values=[1.0, 2.0, 3.1])
    shifted = input.shift(shift=1)

    np.testing.assert_array_equal(
        actual=shifted.values, desired=[np.nan, 1.0, 2.0]
    ), "Values should be shifted by 1"


def test_min_does_return_min_of_the_values():
    input = Serie(values=[1.0, 2.0, 3.1])
    assert input.__min__() == 1.0, "Minimum should be 1.0"


def test_max_does_return_max_of_the_values():
    input = Serie(values=[1.0, 2.0, 3.1])
    assert input.__max__() == 3.1, "Maximum should be 3.1"


def test_avg_does_return_avg_of_the_values():
    input = Serie(values=[1.0, 2.0, 3.0])
    assert input.__avg__() == 2.0, "Average should be 2.0"


def test_std_does_return_std_of_the_values():
    input = Serie(values=[1, 1, 1])
    assert input.__std__() == 0, "Standard deviation should be 1.0"


def test_len_does_return_len_of_the_values():
    input = Serie(values=[1.0, 2.0, 3.1])
    assert len(input) == 3, "Length should be 3"


def test_index_and_values_should_have_same_length():
    with pytest.raises(ValueError):
        Serie(values=[1, 2, 3], index=[1, 2, 3, 4])


def test_index_and_values_with_same_length_works_well():
    Serie(
        values=[1, 2, 3],
        index=[
            1,
            2,
            3,
        ],
    )


def test_deseasonalize_without_seasonal_period_raise_error():
    with pytest.raises(ValueError):
        Serie(values=[1, 2, 3], deseasonalize=True)
