from src.devig import devig


def test_devig_sum_to_one():
    probs = [0.55, 0.55]
    for method in ["multiplicative", "additive", "power", "shin", "probit", "average", "worst case"]:
        out = devig(method, probs)
        assert abs(sum(out) - 1) < 1e-6
        assert all(p > 0 for p in out)
