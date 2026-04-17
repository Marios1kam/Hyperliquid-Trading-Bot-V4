"""
Market depth feature analyser.

Computes real-time order book signals used as a gate before
trade execution. Does NOT affect ML training or historical
feature pipelines — it is a purely live, additive signal.
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class MarketDepthAnalyser:
    """
    Analyse an L2 order book snapshot and return depth features.

    All methods are stateless — call analyse() with each fresh snapshot.
    """

    def analyse(self, order_book: Dict, n_levels: int = 10, signal_direction: str = 'long') -> Dict:
        """
        Compute market-depth features from a parsed order book.

        Args:
            order_book: Dict with keys 'bids' and 'asks', each a list of
                        [price: float, size: float] pairs, best level first.
            n_levels:   How many levels to use for aggregate calculations.
            signal_direction: 'long' or 'short' — used to calculate the
                        directional confidence boost.

        Returns:
            Dictionary of depth features:
                bid_ask_spread_pct      float   (best_ask - best_bid) / mid * 100
                order_book_imbalance    float   [-1, +1]  positive = buy pressure
                depth_ratio             float   sum(bid sizes) / sum(ask sizes)
                large_bid_wall          bool    any bid level > 2× avg bid size
                large_ask_wall          bool    any ask level > 2× avg ask size
                depth_confidence_boost  float   [0.8, 1.2] multiplier
        """
        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])

        # ── Guard: need at least one level on each side ──────────────────────
        if not bids or not asks:
            logger.warning("Order book is empty; returning neutral depth features")
            return self._neutral_features()

        # Best prices (first entry is best bid / best ask)
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])

        if best_bid <= 0 or best_ask <= 0 or best_ask <= best_bid:
            logger.warning(
                f"Suspicious best_bid={best_bid} best_ask={best_ask}; "
                "returning neutral depth features"
            )
            return self._neutral_features()

        mid_price = (best_bid + best_ask) / 2.0

        # ── (1) Bid-ask spread ────────────────────────────────────────────────
        bid_ask_spread_pct = (best_ask - best_bid) / mid_price * 100.0

        # ── Use only the requested number of levels for aggregate stats ───────
        top_bids = bids[:n_levels]
        top_asks = asks[:n_levels]

        total_bid_size = sum(float(lvl[1]) for lvl in top_bids)
        total_ask_size = sum(float(lvl[1]) for lvl in top_asks)

        # ── (2) Order book imbalance ──────────────────────────────────────────
        denom = total_bid_size + total_ask_size
        if denom > 0:
            order_book_imbalance = (total_bid_size - total_ask_size) / denom
        else:
            order_book_imbalance = 0.0

        # ── (3) Depth ratio ───────────────────────────────────────────────────
        if total_ask_size > 0:
            depth_ratio = total_bid_size / total_ask_size
        else:
            depth_ratio = 1.0  # neutral

        # ── (4 & 5) Wall detection ────────────────────────────────────────────
        avg_bid_size = total_bid_size / len(top_bids) if top_bids else 0.0
        avg_ask_size = total_ask_size / len(top_asks) if top_asks else 0.0

        large_bid_wall = any(float(lvl[1]) > 2.0 * avg_bid_size for lvl in top_bids) if avg_bid_size > 0 else False
        large_ask_wall = any(float(lvl[1]) > 2.0 * avg_ask_size for lvl in top_asks) if avg_ask_size > 0 else False

        # ── (6) Directional confidence boost ─────────────────────────────────
        depth_confidence_boost = self._calc_confidence_boost(
            order_book_imbalance, signal_direction
        )

        features = {
            'bid_ask_spread_pct': round(bid_ask_spread_pct, 6),
            'order_book_imbalance': round(order_book_imbalance, 6),
            'depth_ratio': round(depth_ratio, 6),
            'large_bid_wall': large_bid_wall,
            'large_ask_wall': large_ask_wall,
            'depth_confidence_boost': round(depth_confidence_boost, 6),
        }

        logger.debug(
            f"Depth features | spread={bid_ask_spread_pct:.4f}% "
            f"imbalance={order_book_imbalance:.3f} "
            f"depth_ratio={depth_ratio:.3f} "
            f"bid_wall={large_bid_wall} ask_wall={large_ask_wall} "
            f"boost={depth_confidence_boost:.3f} dir={signal_direction}"
        )

        return features

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _calc_confidence_boost(self, imbalance: float, signal_direction: str) -> float:
        """
        Map order-book imbalance to a confidence multiplier [0.8, 1.2].

        For a LONG signal:
          - Strong positive imbalance (> 0.20) → boost toward 1.2
          - Strong negative imbalance (< -0.20) → penalise toward 0.8
          - Near-zero → 1.0 (neutral)

        For a SHORT signal the logic is inverted.

        The linear mapping is:
          boost = 1.0 + (directional_imbalance / 0.40) * 0.20
        clamped to [0.8, 1.2].
        """
        BOOST_MAX = 1.20
        BOOST_MIN = 0.80
        SCALE = 0.40  # ±0.40 imbalance → ±0.20 boost delta

        direction = signal_direction.lower()
        if direction == 'long':
            directional = imbalance           # positive is confirming
        elif direction == 'short':
            directional = -imbalance          # negative is confirming
        else:
            logger.warning(f"Unknown signal_direction '{signal_direction}'; using neutral boost")
            return 1.0

        raw_boost = 1.0 + (directional / SCALE) * (BOOST_MAX - 1.0)
        return max(BOOST_MIN, min(BOOST_MAX, raw_boost))

    @staticmethod
    def _neutral_features() -> Dict:
        """Return a neutral feature dict when analysis cannot proceed."""
        return {
            'bid_ask_spread_pct': 0.0,
            'order_book_imbalance': 0.0,
            'depth_ratio': 1.0,
            'large_bid_wall': False,
            'large_ask_wall': False,
            'depth_confidence_boost': 1.0,
        }
