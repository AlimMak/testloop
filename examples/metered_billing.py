"""Metered subscription billing: tiered overage plus proration.

A stress-test target for testloop. The exact dollar amounts here come from
tiered bracket math and proration with rounding at each step, so a first pass of
generated tests often asserts slightly-wrong expected values, fails, and then
gets repaired from the real pytest output. It also has several boundary branches
that are easy to miss. The code is correct and every line is reachable, so a
thorough suite can still reach 100%.
"""

from __future__ import annotations


class BillingError(ValueError):
    """Raised for any invalid billing input."""


# Tiers billed like tax brackets: (upto_units, price_per_unit). The final tier
# uses upto=None to mean "this rate applies to everything beyond".
DEFAULT_TIERS = [(1000, 0.010), (10000, 0.008), (None, 0.005)]


def overage_charge(units: int, tiers=DEFAULT_TIERS) -> float:
    """Charge for `units` billed across the given bracket tiers."""
    if units < 0:
        raise BillingError("units cannot be negative")
    if units == 0:
        return 0.0

    charge = 0.0
    lower = 0
    for upto, rate in tiers:
        if upto is None:
            billable = units - lower
        else:
            billable = min(units, upto) - lower
        if billable > 0:
            charge += billable * rate
        if upto is not None and units <= upto:
            break
        lower = upto if upto is not None else lower
    return round(charge, 2)


def prorate(amount: float, days_used: int, days_in_month: int) -> float:
    """Prorate an amount for a partial month. days_used is capped at a full month."""
    if days_in_month <= 0:
        raise BillingError("days_in_month must be positive")
    if days_used < 0:
        raise BillingError("days_used cannot be negative")
    if days_used > days_in_month:
        days_used = days_in_month
    return round(amount * days_used / days_in_month, 2)


def monthly_bill(base_fee: float, units: int, days_used: int,
                 days_in_month: int = 30, tiers=DEFAULT_TIERS) -> float:
    """Full monthly charge: base fee plus tiered overage, prorated for the period."""
    if base_fee < 0:
        raise BillingError("base_fee cannot be negative")
    overage = overage_charge(units, tiers)
    gross = round(base_fee + overage, 2)
    return prorate(gross, days_used, days_in_month)
