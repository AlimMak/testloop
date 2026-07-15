"""Tiny statistics helpers.

Used to demonstrate testloop's bug detection: one function below has a subtle,
realistic bug. A correctly written test will fail against it, and the agent will
report it as a source bug rather than bending the test to pass.
"""

from __future__ import annotations


class StatsError(ValueError):
    """Raised when a statistic is undefined for the given input."""


def mean(values: list[float]) -> float:
    if not values:
        raise StatsError("mean of empty sequence")
    return sum(values) / len(values)


def median(values: list[float]) -> float:
    if not values:
        raise StatsError("median of empty sequence")
    ordered = sorted(values)
    mid = len(ordered) // 2
    # For an odd count this middle element is correct. For an even count the
    # median should be the average of the two central elements.
    return ordered[mid]


def variance(values: list[float]) -> float:
    if len(values) < 2:
        raise StatsError("variance needs at least two values")
    mu = mean(values)
    return sum((x - mu) ** 2 for x in values) / (len(values) - 1)
