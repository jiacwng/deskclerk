# The statistics behind the result tables, from scipy: the Wilson score interval for a count out
# of n, and the exact McNemar test for two approaches scored on the same items, which is a
# two-sided binomial test on the items only one of them got right.

from scipy.stats import binomtest


def wilson(successes: int, trials: int) -> tuple[float, float]:
    interval = binomtest(successes, trials).proportion_ci(method="wilson")
    return float(interval.low), float(interval.high)


def with_interval(successes: int, trials: int) -> str:
    low, high = wilson(successes, trials)
    return f"{successes} ({_percent(low)} to {_percent(high)})"


# A bound under 0.1% keeps two significant digits, so a large clean count does not print as 0.0%.
def _percent(share: float) -> str:
    percent = share * 100
    if 0 < percent < 0.1:
        return f"{percent:.2g}%"
    return f"{percent:.1f}%"


def mcnemar_exact(only_first: int, only_second: int) -> float:
    discordant = only_first + only_second
    if discordant == 0:
        return 1.0
    return float(binomtest(only_first, discordant, 0.5).pvalue)
