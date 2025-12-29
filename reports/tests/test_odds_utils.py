from src.odds_utils import american_to_decimal, decimal_to_american, to_decimal


def test_american_to_decimal():
    assert round(american_to_decimal(150), 2) == 2.5
    assert round(american_to_decimal(-200), 2) == 1.5


def test_decimal_to_american():
    assert round(decimal_to_american(2.5), 0) == 150
    assert round(decimal_to_american(1.5), 0) == -200


def test_to_decimal_detection():
    assert to_decimal(2.2) == 2.2
    assert round(to_decimal(-110), 3) == 1.909
    assert round(to_decimal(0.95), 2) == 1.95
    assert round(to_decimal("+150"), 2) == 2.5
