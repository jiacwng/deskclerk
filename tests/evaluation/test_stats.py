# The interval and the paired test behind the result tables, checked against closed forms.

import pytest

from needtoknow.evaluation.stats import mcnemar_exact, wilson, with_interval


def test_the_wilson_interval_of_zero_leaks_in_195() -> None:
    low, high = wilson(0, 195)

    assert low == 0.0
    # With no successes the upper bound is z^2 / (n + z^2).
    assert high == pytest.approx(1.96**2 / (195 + 1.96**2), abs=0.0001)


def test_a_count_is_printed_with_its_interval() -> None:
    assert with_interval(0, 195) == "0 (0.0% to 1.9%)"
    assert with_interval(0, 254_544) == "0 (0.0% to 0.0015%)"


def test_the_exact_mcnemar_test() -> None:
    assert mcnemar_exact(29, 44) == pytest.approx(0.101, abs=0.001)
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(0, 10) == pytest.approx(2 / 1024)
