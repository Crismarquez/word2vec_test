"""
Tests the util.add_plus_one function
"""

from .. import util


def test_add_two_plus_one():
    """
    Checks that 2 + 1 = 3
    """

    result = util.add_plus_one(2)

    assert result == 3


def test_add_larger_than_one():
    """
    Checks that x + 1 > 1
    """

    result = util.add_plus_one(3)

    assert result >= 1
