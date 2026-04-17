"""
Base strategy class.
Defines interface for all trading strategies.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
import logging

from src.config import config
from src.features.market_depth_features import MarketDepthAnalyser


logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """Trading signal."""
    symbol: str
    action: str  # 'buy', 'sell', 'close', 'hold'
    confidence: float
    regime: str
    size: Optional[float] = None
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reason: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert signal to dictionary."""
        return {
            'symbol': self.symbol,
            'action': self.action,
            'confidence': self.confidence,
            'regime': self.regime,
            'size': self.size,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'reason': self.reason,
        }


class BaseStrategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(self, name: str, client=None):
        """
        Initialize strategy.
        
        Args:
            name:   Strategy name
            client: Optional HyperliquidClient instance used by the depth gate.
                    When None the depth gate becomes a no-op (fail-open).
        """
        self.name = name
        self.min_confidence = config.min_confidence
        self.client = client  # may be None; set by TradingEngine after construction

        logger.info(f"Initialized strategy: {name}")
    
    @abstractmethod
    def generate_signal(
        self,
        symbol: str,
        ml_signal: int,
        ml_confidence: float,
        regime: str,
        features: Dict,
        current_price: float
    ) -> Optional[Signal]:
        """
        Generate trading signal.
        
        Args:
            symbol: Trading symbol
            ml_signal: ML model signal (1, 0, -1)
            ml_confidence: ML model confidence
            regime: Current market regime
            features: Dictionary of features
            current_price: Current market price
            
        Returns:
            Signal object or None
        """
        pass

    # ── Market-depth gate ─────────────────────────────────────────────────────

    def _check_depth_gate(
        self,
        symbol: str,
        signal_direction: str,
    ) -> Tuple[bool, float]:
        """
        Real-time order-book gate that runs *after* ML confidence passes
        MIN_CONFIDENCE and *before* order execution.

        Args:
            symbol:           Trading symbol (e.g. 'BTC').
            signal_direction: 'long' or 'short'.

        Returns:
            (allowed: bool, confidence_multiplier: float)
              - (True,  boost)  → trade may proceed; multiply confidence by boost
              - (False, 0.0)    → trade skipped; reason already logged
        """
        # ── Fast-path: feature disabled in config ─────────────────────────────
        depth_cfg = config.get('MARKET_DEPTH', {})
        if not depth_cfg.get('ENABLED', True):
            return True, 1.0

        # ── Fast-path: no client available (standalone / test mode) ───────────
        if self.client is None:
            logger.debug(
                f"[DepthGate] {symbol}: no client attached, skipping gate (fail-open)"
            )
            return True, 1.0

        # ── Fetch order book ──────────────────────────────────────────────────
        n_levels = depth_cfg.get('LEVELS', 10)
        order_book = self.client.get_order_book(symbol, n_levels=n_levels)

        if order_book is None:
            logger.warning(
                f"[DepthGate] {symbol}: order book fetch failed — "
                "allowing trade (fail-open)"
            )
            return True, 1.0

        # ── Compute depth features ────────────────────────────────────────────
        try:
            depth_features = MarketDepthAnalyser().analyse(
                order_book,
                n_levels=n_levels,
                signal_direction=signal_direction,
            )
        except Exception as exc:
            logger.warning(
                f"[DepthGate] {symbol}: depth analysis error ({exc}) — "
                "allowing trade (fail-open)"
            )
            return True, 1.0

        spread_pct   = depth_features['bid_ask_spread_pct']
        imbalance    = depth_features['order_book_imbalance']
        boost        = depth_features['depth_confidence_boost']

        # ── Rule 1: skip if market is too illiquid ────────────────────────────
        max_spread = depth_cfg.get('MAX_SPREAD_PCT', 0.15)
        if spread_pct > max_spread:
            logger.info(
                f"[DepthGate] {symbol} SKIP — spread {spread_pct:.4f}% "
                f"exceeds MAX_SPREAD_PCT {max_spread}% (direction={signal_direction})"
            )
            return False, 0.0

        # ── Rule 2: skip if imbalance strongly opposes the trade ──────────────
        skip_threshold = depth_cfg.get('IMBALANCE_SKIP_THRESHOLD', -0.30)
        if signal_direction.lower() == 'long' and imbalance < skip_threshold:
            logger.info(
                f"[DepthGate] {symbol} SKIP — imbalance {imbalance:.3f} "
                f"< IMBALANCE_SKIP_THRESHOLD {skip_threshold} for LONG"
            )
            return False, 0.0

        if signal_direction.lower() == 'short' and imbalance > abs(skip_threshold):
            logger.info(
                f"[DepthGate] {symbol} SKIP — imbalance {imbalance:.3f} "
                f"> {abs(skip_threshold)} for SHORT"
            )
            return False, 0.0

        # ── All checks passed ─────────────────────────────────────────────────
        logger.debug(
            f"[DepthGate] {symbol} PASS — spread={spread_pct:.4f}% "
            f"imbalance={imbalance:.3f} boost={boost:.3f} dir={signal_direction}"
        )
        return True, boost

    # ── Signal validation (with depth gate wired in) ──────────────────────────

    def validate_signal(self, signal: Signal) -> bool:
        """
        Validate a trading signal.
        
        Runs the market-depth gate after the minimum-confidence check.
        If the gate passes, the signal's confidence is multiplied by the
        depth_confidence_boost so downstream sizing reflects order-book
        conviction.

        Args:
            signal: Signal to validate
            
        Returns:
            True if valid
        """
        if signal.confidence < self.min_confidence:
            logger.debug(
                f"Signal rejected: confidence {signal.confidence:.3f} "
                f"< minimum {self.min_confidence}"
            )
            return False
        
        if signal.action not in ['buy', 'sell', 'close', 'hold']:
            logger.warning(f"Invalid signal action: {signal.action}")
            return False

        # ── Depth gate ────────────────────────────────────────────────────────
        signal_direction = 'long' if signal.action == 'buy' else 'short'
        gate_pass, boost = self._check_depth_gate(signal.symbol, signal_direction)

        if not gate_pass:
            # Reason already logged inside _check_depth_gate
            return False

        # Apply the depth confidence multiplier in-place so all downstream
        # callers (position sizer, logging) see the adjusted value.
        if boost != 1.0:
            original = signal.confidence
            signal.confidence = round(signal.confidence * boost, 6)
            logger.debug(
                f"[DepthGate] {signal.symbol}: confidence adjusted "
                f"{original:.4f} → {signal.confidence:.4f} (boost={boost:.3f})"
            )

        return True
