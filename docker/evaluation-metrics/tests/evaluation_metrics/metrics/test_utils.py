import numpy as np
import pytest
from evaluation_metrics.metrics.utils import convert_negatives_to_zero


def test_convert_negatives_to_zero_all_positive():
    array = np.array([1, 2, 3, 4, 5])
    result = convert_negatives_to_zero(array)
    expected = np.array([1, 2, 3, 4, 5])
    np.testing.assert_array_equal(result, expected)


def test_convert_negatives_to_zero_all_negative():
    array = np.array([-1, -2, -3, -4, -5])
    result = convert_negatives_to_zero(array)
    expected = np.array([0, 0, 0, 0, 0])
    np.testing.assert_array_equal(result, expected)


def test_convert_negatives_to_zero_mixed_values():
    array = np.array([-1, 2, -3, 4, -5])
    result = convert_negatives_to_zero(array)
    expected = np.array([0, 2, 0, 4, 0])
    np.testing.assert_array_equal(result, expected)


def test_convert_negatives_to_zero_empty_array():
    array = np.array([])
    result = convert_negatives_to_zero(array)
    expected = np.array([])
    np.testing.assert_array_equal(result, expected)


def test_convert_negatives_to_zero_no_negatives():
    array = np.array([0, 1, 2, 3])
    result = convert_negatives_to_zero(array)
    expected = np.array([0, 1, 2, 3])
    np.testing.assert_array_equal(result, expected)


def test_convert_negatives_to_zero_large_array():
    array = np.random.randint(-100, 100, size=1000)
    result = convert_negatives_to_zero(array)
    expected = np.where(array < 0, 0, array)
    np.testing.assert_array_equal(result, expected)
