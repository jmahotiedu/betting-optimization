from src.odds_utils import american_to_decimal, to_decimal


def test_american_to_decimal_positive():
    assert round(american_to_decimal(150), 2) == 2.5


def test_american_to_decimal_negative():
    assert round(american_to_decimal(-200), 2) == 1.5


def test_to_decimal_detects_american():
    assert round(to_decimal(-120), 4) == round(1 + 100 / 120, 4)
