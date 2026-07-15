import pytest
import target
from target import BillingError, Coupon, line_total, coupon_discount, tax, checkout


# ---------- line_total ----------

def test_line_total_basic():
    assert line_total(2, 3.5) == 7.0


def test_line_total_zero_quantity():
    assert line_total(0, 100) == 0.0


def test_line_total_zero_price():
    assert line_total(5, 0) == 0.0


def test_line_total_rounding():
    assert line_total(3, 0.1) == 0.3


def test_line_total_negative_quantity_raises():
    with pytest.raises(BillingError):
        line_total(-1, 5)


def test_line_total_negative_price_raises():
    with pytest.raises(BillingError):
        line_total(5, -1)


# ---------- coupon_discount ----------

def test_coupon_discount_none_coupon():
    assert coupon_discount(100, None) == 0.0


def test_coupon_discount_negative_subtotal_raises():
    with pytest.raises(BillingError):
        coupon_discount(-10, None)


def test_coupon_discount_below_min_spend():
    c = Coupon(kind="percent", value=10, min_spend=50)
    assert coupon_discount(40, c) == 0.0


def test_coupon_discount_at_min_spend_boundary():
    c = Coupon(kind="fixed", value=5, min_spend=50)
    assert coupon_discount(50, c) == 5.0


def test_coupon_discount_percent_basic():
    c = Coupon(kind="percent", value=10)
    assert coupon_discount(200, c) == 20.0


def test_coupon_discount_percent_zero():
    c = Coupon(kind="percent", value=0)
    assert coupon_discount(200, c) == 0.0


def test_coupon_discount_percent_hundred():
    c = Coupon(kind="percent", value=100)
    assert coupon_discount(200, c) == 200.0


def test_coupon_discount_percent_out_of_range_negative_raises():
    c = Coupon(kind="percent", value=-1)
    with pytest.raises(BillingError):
        coupon_discount(100, c)


def test_coupon_discount_percent_out_of_range_over_100_raises():
    c = Coupon(kind="percent", value=101)
    with pytest.raises(BillingError):
        coupon_discount(100, c)


def test_coupon_discount_fixed_basic():
    c = Coupon(kind="fixed", value=15)
    assert coupon_discount(100, c) == 15.0


def test_coupon_discount_fixed_negative_raises():
    c = Coupon(kind="fixed", value=-5)
    with pytest.raises(BillingError):
        coupon_discount(100, c)


def test_coupon_discount_unknown_kind_raises():
    c = Coupon(kind="bogus", value=5)
    with pytest.raises(BillingError):
        coupon_discount(100, c)


def test_coupon_discount_max_discount_cap():
    c = Coupon(kind="percent", value=50, max_discount=20)
    assert coupon_discount(100, c) == 20.0


def test_coupon_discount_max_discount_not_triggered():
    c = Coupon(kind="percent", value=10, max_discount=50)
    assert coupon_discount(100, c) == 10.0


def test_coupon_discount_never_exceeds_subtotal():
    c = Coupon(kind="fixed", value=1000)
    assert coupon_discount(50, c) == 50.0


def test_coupon_discount_rounding():
    c = Coupon(kind="percent", value=33.333)
    assert coupon_discount(10, c) == round(10 * 33.333 / 100, 2)


# ---------- tax ----------

def test_tax_basic():
    assert tax(100, 0.2) == 20.0


def test_tax_zero_rate():
    assert tax(100, 0) == 0.0


def test_tax_full_rate():
    assert tax(100, 1) == 100.0


def test_tax_negative_amount_raises():
    with pytest.raises(BillingError):
        tax(-10, 0.1)


def test_tax_rate_below_zero_raises():
    with pytest.raises(BillingError):
        tax(100, -0.1)


def test_tax_rate_above_one_raises():
    with pytest.raises(BillingError):
        tax(100, 1.1)


def test_tax_rounding():
    assert tax(10, 0.333) == round(10 * 0.333, 2)


# ---------- checkout ----------

def test_checkout_empty_cart_raises():
    with pytest.raises(BillingError):
        checkout([])


def test_checkout_none_items_raises():
    with pytest.raises(BillingError):
        checkout(None)


def test_checkout_basic_no_coupon_no_tax():
    items = [(2, 10.0), (1, 5.0)]
    result = checkout(items)
    assert result == {
        "subtotal": 25.0,
        "discount": 0.0,
        "tax": 0.0,
        "total": 25.0,
    }


def test_checkout_with_coupon():
    items = [(1, 100.0)]
    coupon = Coupon(kind="percent", value=10)
    result = checkout(items, coupon=coupon)
    assert result["subtotal"] == 100.0
    assert result["discount"] == 10.0
    assert result["tax"] == 0.0
    assert result["total"] == 90.0


def test_checkout_with_tax():
    items = [(1, 100.0)]
    result = checkout(items, tax_rate=0.1)
    assert result["subtotal"] == 100.0
    assert result["discount"] == 0.0
    assert result["tax"] == 10.0
    assert result["total"] == 110.0


def test_checkout_with_coupon_and_tax():
    items = [(1, 200.0)]
    coupon = Coupon(kind="fixed", value=50)
    result = checkout(items, coupon=coupon, tax_rate=0.05)
    assert result["subtotal"] == 200.0
    assert result["discount"] == 50.0
    assert result["tax"] == round(150.0 * 0.05, 2)
    assert result["total"] == round(150.0 + result["tax"], 2)


def test_checkout_propagates_line_total_error():
    with pytest.raises(BillingError):
        checkout([(-1, 10.0)])


def test_checkout_propagates_coupon_error():
    items = [(1, 100.0)]
    coupon = Coupon(kind="percent", value=200)
    with pytest.raises(BillingError):
        checkout(items, coupon=coupon)


def test_checkout_propagates_tax_error():
    items = [(1, 100.0)]
    with pytest.raises(BillingError):
        checkout(items, tax_rate=2.0)


def test_checkout_multiple_items_rounding():
    items = [(3, 0.1), (2, 0.2)]
    result = checkout(items)
    assert result["subtotal"] == round(3 * 0.1 + 2 * 0.2, 2)