"""
Sideways market strategies for range-bound conditions.
"""

import logging
from typing import Optional, Dict

from src.strategies.base_strategy import BaseStrategy, Signal
from src.config import config


logger = logging.getLogger(__name__)


class SidewaysStrategy(BaseStrategy):
    """Mean-reversion strategy for sideways markets."""
    
    def __init__(self):
        """Initialize sideways strategy."""
        super().__init__("SidewaysMeanReversion")
        
        # Load sideways configuration
        self.sideways_config = config.get('SIDEWAYS', {})
        self.min_sideways_confidence = self.sideways_config.get(
            'SIDEWAYS_DETECT_CONFIDENCE_MIN', 0.65
        )
    
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
        Generate sideways mean-reversion signal.
        
        Strategy:
        - SIDEWAYS_QUIET: Tight mean-reversion with tighter stops
        - SIDEWAYS_VOLATILE: Volatility-scaled mean-reversion
        - CHOPPY: Reduce size or disable
        
        Args:
            symbol: Trading symbol
            ml_signal: ML model signal
            ml_confidence: ML model confidence
            regime: Current market regime
            features: Dictionary of features
            current_price: Current market price
            
        Returns:
            Signal object or None
        """
        # Only trade in sideways regimes
        if regime not in ['SIDEWAYS_QUIET', 'SIDEWAYS_VOLATILE']:
            return None
        
        # Check sideways confidence
        sideways_confidence = features.get('sideways_confidence', 0.0)
        if sideways_confidence < self.min_sideways_confidence:
            logger.debug(f"{symbol}: Sideways confidence too low: {sideways_confidence:.3f}")
            return None
        
        # Check for mean-reversion setup
        setup = self._check_mean_reversion_setup(features, regime)
        if not setup:
            return None
        
        action, reason = setup
        
        # Combine ML confidence with sideways confidence
        combined_confidence = (ml_confidence + sideways_confidence) / 2
        
        signal = Signal(
            symbol=symbol,
            action=action,
            confidence=combined_confidence,
            regime=regime,
            entry_price=current_price,
            reason=reason
        )
        
        if self.validate_signal(signal):
            logger.info(f"Generated sideways signal: {action} {symbol} "
                       f"(regime={regime}, confidence={combined_confidence:.3f})")
            return signal
        
        return None
    
    def _check_mean_reversion_setup(
        self,
        features: Dict,
        regime: str
    ) -> Optional[tuple]:
        """
        Check for mean-reversion setup.
        
        Returns:
            Tuple of (action, reason) or None
        """
        # Get range features
        range_position = features.get('range_position')
        bb_position = features.get('bb_position')
        short_rsi = features.get('short_rsi')
        
        if range_position is None or bb_position is None:
            return None
        
        if regime == 'SIDEWAYS_QUIET':
            # Tight mean-reversion in quiet markets
            # Buy near support, sell near resistance
            if range_position < 0.2 and (short_rsi is None or short_rsi < 30):
                return ('buy', 'Near support in quiet sideways')
            elif range_position > 0.8 and (short_rsi is None or short_rsi > 70):
                return ('sell', 'Near resistance in quiet sideways')
        
        elif regime == 'SIDEWAYS_VOLATILE':
            # Wider bands for volatile sideways
            # More conservative entries
            if range_position < 0.15 and bb_position < 0.1:
                return ('buy', 'Extreme oversold in volatile sideways')
            elif range_position > 0.85 and bb_position > 0.9:
                return ('sell', 'Extreme overbought in volatile sideways')
        
        return None


class BreakoutStrategy(BaseStrategy):
    """Breakout strategy for sideways markets."""
    
    def __init__(self):
        """Initialize breakout strategy."""
        super().__init__("SidewaysBreakout")
    
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
        Generate breakout signal from sideways range.
        
        Strategy:
        - Detect breakouts from established ranges
        - Require volume confirmation
        - Higher confidence threshold
        
        Args:
            symbol: Trading symbol
            ml_signal: ML model signal
            ml_confidence: ML model confidence
            regime: Current market regime
            features: Dictionary of features
            current_price: Current market price
            
        Returns:
            Signal object or None
        """
        # Can trade breakouts from any sideways regime
        if not regime.startswith('SIDEWAYS'):
            return None
        
        # Require higher confidence for breakouts
        if ml_confidence < self.min_confidence * 1.3:
            return None
        
        # Check for breakout
        breakout = self._check_breakout(features, ml_signal)
        if not breakout:
            return None
        
        action, reason = breakout
        
        signal = Signal(
            symbol=symbol,
            action=action,
            confidence=ml_confidence,
            regime=regime,
            entry_price=current_price,
            reason=reason
        )
        
        if self.validate_signal(signal):
            logger.info(f"Generated breakout signal: {action} {symbol} "
                       f"(regime={regime}, confidence={ml_confidence:.3f})")
            return signal
        
        return None
    
    def _check_breakout(self, features: Dict, ml_signal: int) -> Optional[tuple]:
        """Check for breakout conditions."""
        # Get breakout indicators
        breakout_up = features.get('breakout_up', 0)
        breakout_down = features.get('breakout_down', 0)
        volume_spike = features.get('volume_spike', 0)
        
        # Require volume confirmation
        if not volume_spike:
            return None
        
        if ml_signal == 1 and breakout_up:
            return ('buy', 'Upside breakout with volume')
        elif ml_signal == -1 and breakout_down:
            return ('sell', 'Downside breakout with volume')
        
        return None


class ChoppyStrategy(BaseStrategy):
    """Conservative strategy for choppy markets."""
    
    def __init__(self):
        """Initialize choppy strategy."""
        super().__init__("ChoppyConservative")
    
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
        Generate signal for choppy markets.
        
        Strategy:
        - Very conservative: only trade with very high confidence
        - Smaller position sizes (handled by position sizer)
        - Prefer to stay flat
        
        Args:
            symbol: Trading symbol
            ml_signal: ML model signal
            ml_confidence: ML model confidence
            regime: Current market regime
            features: Dictionary of features
            current_price: Current market price
            
        Returns:
            Signal object or None
        """
        signal_dir = "LONG" if ml_signal == 1 else "SHORT"
        if regime != 'CHOPPY':
            print(f"❌ {symbol:6} | {signal_dir:5} {ml_confidence:.0%} | {regime:6} | Conservative | → Wrong regime (need CHOPPY)")
            return None
        
        # Use base confidence threshold - ML model already accounts for regime difficulty
        # Additional filtering via _check_strong_signal (MACD/RSI) provides extra validation
        if ml_confidence < self.min_confidence:
            print(f"❌ {symbol:6} | {signal_dir:5} {ml_confidence:.0%} | {regime:6} | Conservative | → Below {self.min_confidence:.0%} threshold")
            return None
        
        # Only trade if ML signal is strong and aligned with short-term momentum
        if not self._check_strong_signal(features, ml_signal):
            print(f"❌ {symbol:6} | {signal_dir:5} {ml_confidence:.0%} | {regime:6} | Conservative | → Weak signal (MACD/RSI)")
            return None
        
        action = 'buy' if ml_signal == 1 else 'sell'
        
        signal = Signal(
            symbol=symbol,
            action=action,
            confidence=ml_confidence,
            regime=regime,
            entry_price=current_price,
            reason="High-confidence signal in choppy market"
        )
        
        if self.validate_signal(signal):
            logger.info(f"Generated choppy signal: {action} {symbol} "
                       f"(confidence={ml_confidence:.3f})")
            return signal
        
        return None
    
    def _check_strong_signal(self, features: Dict, ml_signal: int) -> bool:
        """Check for strong signal confirmation."""
        macd_hist = features.get('macd_hist')
        rsi = features.get('rsi')
        
        if macd_hist is None or rsi is None:
            return False
        
        if ml_signal == 1:
            # Long: MACD positive, RSI not overbought
            return macd_hist > 0 and 40 < rsi < 65
        elif ml_signal == -1:
            # Short: MACD negative, RSI not oversold
            return macd_hist < 0 and 35 < rsi < 60
        
        return False
