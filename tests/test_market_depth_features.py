"""
Unit tests for MarketDepthAnalyser.

Run with:  python -m pytest tests/test_market_depth_features.py -v
"""

import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH so src.* imports resolve
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.market_depth_features import MarketDepthAnalyser


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_book(bid_price=100.0, ask_price=100.1, n_levels=5, bid_sizes=None, ask_sizes=None):
    """Return a minimal synthetic order book."""
    if bid_sizes is None:
        bid_sizes = [10.0] * n_levels
    if ask_sizes is None:
        ask_sizes = [10.0] * n_levels

    bids = [[bid_price - i * 0.01, bid_sizes[i]] for i in range(n_levels)]
    asks = [[ask_price + i * 0.01, ask_sizes[i]] for i in range(n_levels)]
    return {'bids': bids, 'asks': asks}


analyser = MarketDepthAnalyser()


# ── Core feature calculations ──────────────────────────────────────────────────

def test_bid_ask_spread_pct():
    """Spread of 0.1 on a mid of 100.05 ≈ 0.0999%."""
    book = _make_book(bid_price=100.0, ask_price=100.1)
    result = analyser.analyse(book, n_levels=5, signal_direction='long')
    mid = (100.0 + 100.1) / 2
    expected = (100.1 - 100.0) / mid * 100
    assert abs(result['bid_ask_spread_pct'] - expected) < 1e-4, (
        f"Expected spread {expected:.6f}%, got {result['bid_ask_spread_pct']}"
    )


def test_balanced_book_imbalance_near_zero():
    """Equal sizes on both sides → imbalance ≈ 0."""
    book = _make_book()
    result = analyser.analyse(book, n_levels=5)
    assert abs(result['order_book_imbalance']) < 1e-6, (
        f"Expected ~0 imbalance, got {result['order_book_imbalance']}"
    )


def test_bid_heavy_imbalance_positive():
    """More bid volume than ask volume → positive imbalance."""
    bid_sizes = [20.0] * 5
    ask_sizes = [10.0] * 5
    book = _make_book(bid_sizes=bid_sizes, ask_sizes=ask_sizes)
    result = analyser.analyse(book, n_levels=5)
    # (100 - 50) / (100 + 50) = 0.333...
    assert result['order_book_imbalance'] > 0, "Expected positive imbalance for bid-heavy book"
    assert abs(result['order_book_imbalance'] - (1 / 3)) < 1e-4


def test_ask_heavy_imbalance_negative():
    """More ask volume than bid volume → negative imbalance."""
    bid_sizes = [10.0] * 5
    ask_sizes = [20.0] * 5
    book = _make_book(bid_sizes=bid_sizes, ask_sizes=ask_sizes)
    result = analyser.analyse(book, n_levels=5)
    assert result['order_book_imbalance'] < 0, "Expected negative imbalance for ask-heavy book"


def test_depth_ratio_balanced():
    """Equal sizes → depth_ratio ≈ 1.0."""
    book = _make_book()
    result = analyser.analyse(book, n_levels=5)
    assert abs(result['depth_ratio'] - 1.0) < 1e-6


def test_depth_ratio_bid_heavy():
    """Bid sizes double ask sizes → depth_ratio ≈ 2.0."""
    book = _make_book(bid_sizes=[20.0] * 5, ask_sizes=[10.0] * 5)
    result = analyser.analyse(book, n_levels=5)
    assert abs(result['depth_ratio'] - 2.0) < 1e-6


# ── Wall detection ─────────────────────────────────────────────────────────────

def test_large_bid_wall_detected():
    """One enormous bid level → large_bid_wall = True."""
    bid_sizes = [100.0, 5.0, 5.0, 5.0, 5.0]   # 100 >> 2 × avg(24)
    book = _make_book(bid_sizes=bid_sizes, ask_sizes=[10.0] * 5)
    result = analyser.analyse(book, n_levels=5)
    assert result['large_bid_wall'] is True, "Expected large_bid_wall=True"


def test_no_large_bid_wall_uniform():
    """Uniform bid sizes → large_bid_wall = False."""
    book = _make_book()
    result = analyser.analyse(book, n_levels=5)
    assert result['large_bid_wall'] is False


def test_large_ask_wall_detected():
    """One enormous ask level → large_ask_wall = True."""
    ask_sizes = [100.0, 5.0, 5.0, 5.0, 5.0]
    book = _make_book(bid_sizes=[10.0] * 5, ask_sizes=ask_sizes)
    result = analyser.analyse(book, n_levels=5)
    assert result['large_ask_wall'] is True, "Expected large_ask_wall=True"


def test_no_large_ask_wall_uniform():
    """Uniform ask sizes → large_ask_wall = False."""
    book = _make_book()
    result = analyser.analyse(book, n_levels=5)
    assert result['large_ask_wall'] is False


# ── Confidence boost ───────────────────────────────────────────────────────────

def test_boost_neutral_balanced_book():
    """Balanced book → boost ≈ 1.0 for any direction."""
    book = _make_book()
    for direction in ('long', 'short'):
        result = analyser.analyse(book, n_levels=5, signal_direction=direction)
        assert abs(result['depth_confidence_boost'] - 1.0) < 0.01, (
            f"Expected ~1.0 boost for {direction}, got {result['depth_confidence_boost']}"
        )


def test_boost_long_bid_heavy_above_one():
    """Bid-heavy book with long signal → boost > 1.0."""
    book = _make_book(bid_sizes=[30.0] * 5, ask_sizes=[10.0] * 5)
    result = analyser.analyse(book, n_levels=5, signal_direction='long')
    assert result['depth_confidence_boost'] > 1.0, (
        f"Expected boost > 1.0, got {result['depth_confidence_boost']}"
    )


def test_boost_long_ask_heavy_below_one():
    """Ask-heavy book with long signal → boost < 1.0 (penalty)."""
    book = _make_book(bid_sizes=[10.0] * 5, ask_sizes=[30.0] * 5)
    result = analyser.analyse(book, n_levels=5, signal_direction='long')
    assert result['depth_confidence_boost'] < 1.0, (
        f"Expected boost < 1.0, got {result['depth_confidence_boost']}"
    )


def test_boost_short_ask_heavy_above_one():
    """Ask-heavy book with short signal → boost > 1.0 (confirms short)."""
    book = _make_book(bid_sizes=[10.0] * 5, ask_sizes=[30.0] * 5)
    result = analyser.analyse(book, n_levels=5, signal_direction='short')
    assert result['depth_confidence_boost'] > 1.0, (
        f"Expected boost > 1.0 for short on ask-heavy book, got {result['depth_confidence_boost']}"
    )


def test_boost_clamped_between_0_8_and_1_2():
    """Extreme imbalance must be clamped to [0.8, 1.2]."""
    # Extreme bid-heavy, long → should clamp at 1.2
    book = _make_book(bid_sizes=[1000.0] * 5, ask_sizes=[1.0] * 5)
    result = analyser.analyse(book, n_levels=5, signal_direction='long')
    assert result['depth_confidence_boost'] <= 1.2 + 1e-9

    # Extreme ask-heavy, long → should clamp at 0.8
    book = _make_book(bid_sizes=[1.0] * 5, ask_sizes=[1000.0] * 5)
    result = analyser.analyse(book, n_levels=5, signal_direction='long')
    assert result['depth_confidence_boost'] >= 0.8 - 1e-9


# ── Edge cases ─────────────────────────────────────────────────────────────────

def test_empty_bids_returns_neutral():
    """Missing bids → neutral features, no exception."""
    book = {'bids': [], 'asks': [[100.1, 10.0]]}
    result = analyser.analyse(book, n_levels=5)
    assert result['bid_ask_spread_pct'] == 0.0
    assert result['depth_confidence_boost'] == 1.0


def test_empty_asks_returns_neutral():
    """Missing asks → neutral features, no exception."""
    book = {'bids': [[100.0, 10.0]], 'asks': []}
    result = analyser.analyse(book, n_levels=5)
    assert result['bid_ask_spread_pct'] == 0.0
    assert result['depth_confidence_boost'] == 1.0


def test_empty_book_returns_neutral():
    """Completely empty book → neutral feature dict."""
    result = analyser.analyse({'bids': [], 'asks': []})
    assert result == MarketDepthAnalyser._neutral_features()


def test_n_levels_limits_depth():
    """Only n_levels levels used; extra levels ignored."""
    # First 3 bid levels: 10.0 each → total bid @ n=3 = 30
    # Levels 4-10 bid: 100.0 each → total bid @ n=10 = 30 + 7*100 = 730
    # Ask levels: all 10.0 → total ask @ n=3 = 30, @ n=10 = 100
    bid_sizes = [10.0, 10.0, 10.0] + [100.0] * 7
    ask_sizes = [10.0] * 10
    book = _make_book(bid_sizes=bid_sizes, ask_sizes=ask_sizes, n_levels=10)

    result_3  = analyser.analyse(book, n_levels=3)
    result_10 = analyser.analyse(book, n_levels=10)

    # With 3 levels: depth_ratio = 30/30 = 1.0
    assert abs(result_3['depth_ratio'] - 1.0) < 1e-6, (
        f"Expected depth_ratio=1.0 with n=3, got {result_3['depth_ratio']}"
    )

    # With 10 levels: depth_ratio = 730/100 = 7.3
    expected_ratio_10 = 730.0 / 100.0
    assert abs(result_10['depth_ratio'] - expected_ratio_10) < 1e-4, (
        f"Expected depth_ratio={expected_ratio_10} with n=10, got {result_10['depth_ratio']}"
    )


def test_unknown_signal_direction_neutral_boost():
    """Unknown signal_direction → boost = 1.0, no crash."""
    book = _make_book()
    result = analyser.analyse(book, n_levels=5, signal_direction='sideways')
    assert result['depth_confidence_boost'] == 1.0


# ── Return-type guarantees ─────────────────────────────────────────────────────

def test_all_keys_present():
    """analyse() always returns all expected keys."""
    expected_keys = {
        'bid_ask_spread_pct',
        'order_book_imbalance',
        'depth_ratio',
        'large_bid_wall',
        'large_ask_wall',
        'depth_confidence_boost',
    }
    book = _make_book()
    result = analyser.analyse(book)
    assert set(result.keys()) == expected_keys


def test_imbalance_range():
    """order_book_imbalance must always be in [-1, +1]."""
    for bid_mult, ask_mult in [(1, 1), (10, 1), (1, 10), (0.001, 1000)]:
        book = _make_book(
            bid_sizes=[10.0 * bid_mult] * 5,
            ask_sizes=[10.0 * ask_mult] * 5,
        )
        result = analyser.analyse(book, n_levels=5)
        assert -1.0 <= result['order_book_imbalance'] <= 1.0, (
            f"Imbalance out of range: {result['order_book_imbalance']}"
        )


if __name__ == '__main__':
    import traceback

    tests = [
        test_bid_ask_spread_pct,
        test_balanced_book_imbalance_near_zero,
        test_bid_heavy_imbalance_positive,
        test_ask_heavy_imbalance_negative,
        test_depth_ratio_balanced,
        test_depth_ratio_bid_heavy,
        test_large_bid_wall_detected,
        test_no_large_bid_wall_uniform,
        test_large_ask_wall_detected,
        test_no_large_ask_wall_uniform,
        test_boost_neutral_balanced_book,
        test_boost_long_bid_heavy_above_one,
        test_boost_long_ask_heavy_below_one,
        test_boost_short_ask_heavy_above_one,
        test_boost_clamped_between_0_8_and_1_2,
        test_empty_bids_returns_neutral,
        test_empty_asks_returns_neutral,
        test_empty_book_returns_neutral,
        test_n_levels_limits_depth,
        test_unknown_signal_direction_neutral_boost,
        test_all_keys_present,
        test_imbalance_range,
    ]

    passed = failed = 0
    for test in tests:
        try:
            test()
            print(f"  [PASS] {test.__name__}")
            passed += 1
        except Exception:
            print(f"  [FAIL] {test.__name__}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed")
    if failed:
        sys.exit(1)
