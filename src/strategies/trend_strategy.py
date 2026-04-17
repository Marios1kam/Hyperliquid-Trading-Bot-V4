"""
Trend-following strategies for BULL and BEAR regimes.
"""

import logging
from typing import Optional, Dict

from src.strategies.base_strategy import BaseStrategy, Signal
from src.config import config


logger = logging.getLogger(__name__)


class TrendStrategy(BaseStrategy):
    """Trend-following strategy for BULL/BEAR regimes."""
    
    def __init__(self):
        """Initialize trend strategy."""
        super().__init__("TrendFollowing")
        
        # Load regime-specific parameters
        self.bull_config = config.get('REGIME.BULL', {})
        self.bear_config = config.get('REGIME.BEAR', {})
    
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
        Generate trend-following signal.
        
        Strategy:
        - BULL regime: Follow ML long signals
        - BEAR regime: Follow ML short signals
        - Require strong trend confirmation from features
        
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
        # Only trade in BULL or BEAR regimes
        signal_dir = "LONG" if ml_signal == 1 else "SHORT"
        if regime not in ['BULL', 'BEAR']:
            print(f"❌ {symbol:6} | {signal_dir:5} {ml_confidence:.0%} | {regime:6} | Trend Following | → Wrong regime (need BULL/BEAR)")
            return None
        
        # Check ML signal alignment with regime
        # Counter-trend trades require either:
        # 1. High confidence (>75%), OR
        # 2. Moderate confidence (60-75%) + strong technical confirmation (MACD/RSI)
        if regime == 'BULL' and ml_signal == -1:
            if ml_confidence < 0.60:
                print(f"❌ {symbol:6} | SHORT {ml_confidence:.0%} | {regime:6} | Trend Following | → Counter-trend needs >60%")
                return None
            elif ml_confidence < 0.75:
                # Check for technical confirmation
                if not self._check_trend_reversal(features, 'SHORT'):
                    print(f"❌ {symbol:6} | SHORT {ml_confidence:.0%} | {regime:6} | Trend Following | → Needs MACD/RSI confirmation")
                    return None

        
        if regime == 'BEAR' and ml_signal == 1:
            if ml_confidence < 0.60:
                print(f"❌ {symbol:6} | LONG  {ml_confidence:.0%} | {regime:6} | Trend Following | → Counter-trend needs >60%")
                return None
            elif ml_confidence < 0.75:
                # Check for technical confirmation
                if not self._check_trend_reversal(features, 'LONG'):
                    print(f"❌ {symbol:6} | LONG  {ml_confidence:.0%} | {regime:6} | Trend Following | → Needs MACD/RSI confirmation")
                    return None

        
        
        # Verify trend strength from features (only for trend-following trades)
        if (regime == 'BULL' and ml_signal == 1) or (regime == 'BEAR' and ml_signal == -1):
            if not self._confirm_trend(features, regime):
                signal_dir = "LONG" if ml_signal == 1 else "SHORT"
                print(f"❌ {symbol:6} | {signal_dir:5} {ml_confidence:.0%} | {regime:6} | Trend Following | → Trend not confirmed (MA)")
                return None
        
        # Generate signal
        action = 'buy' if ml_signal == 1 else 'sell'
        
        signal = Signal(
            symbol=symbol,
            action=action,
            confidence=ml_confidence,
            regime=regime,
            entry_price=current_price,
            reason=f"Trend following in {regime} regime"
        )
        
        if self.validate_signal(signal):
            logger.info(f"Generated trend signal: {action} {symbol} "
                       f"(regime={regime}, confidence={ml_confidence:.3f})")
            return signal
        
        return None
    
    def _confirm_trend(self, features: Dict, regime: str) -> bool:
        """
        Confirm trend using technical features.
        
        Args:
            features: Dictionary of features
            regime: Current regime
            
        Returns:
            True if trend confirmed
        """
        # Get MA features
        ma_20 = features.get('ma_20')
        ma_50 = features.get('ma_50')
        close = features.get('close')
        
        if ma_20 is None or ma_50 is None or close is None:
            return False
        
        if regime == 'BULL':
            # Confirm uptrend: price > MA20 > MA50
            if close > ma_20 and ma_20 > ma_50:
                return True
        
        elif regime == 'BEAR':
            # Confirm downtrend: price < MA20 < MA50
            if close < ma_20 and ma_20 < ma_50:
                return True
        
        return False
    
    def _check_trend_reversal(self, features: Dict, direction: str) -> bool:
        """
        Check for potential trend reversal using MACD and RSI.
        Used to validate counter-trend trades with moderate confidence (60-75%).
        
        Args:
            features: Dictionary of features
            direction: 'LONG' or 'SHORT'
            
        Returns:
            True if technical indicators support the reversal
        """
        macd_hist = features.get('macd_hist')
        rsi = features.get('rsi')
        
        if macd_hist is None or rsi is None:
            return False
        
        if direction == 'LONG':
            # Looking for bullish reversal signs
            # MACD turning positive (or strongly negative momentum weakening)
            # RSI in oversold/neutral zone (not overbought)
            return macd_hist > -0.5 and 25 < rsi < 65
        
        elif direction == 'SHORT':
            # Looking for bearish reversal signs
            # MACD turning negative (or strongly positive momentum weakening)
            # RSI in overbought/neutral zone (not oversold)
            return macd_hist < 0.5 and 35 < rsi < 75
        
        return False



class MomentumStrategy(BaseStrategy):
    """Momentum-based strategy for strong trends."""
    
    def __init__(self):
        """Initialize momentum strategy."""
        super().__init__("Momentum")
    
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
        Generate momentum signal.
        
        Strategy:
        - Look for strong momentum indicators (RSI, MACD)
        - Align with ML signal
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
        # Only trade in trending regimes
        if regime not in ['BULL', 'BEAR']:
            return None
        
        # Require higher confidence for momentum
        if ml_confidence < self.min_confidence * 1.2:
            return None
        
        # Check momentum indicators
        if not self._check_momentum(features, ml_signal):
            return None
        
        action = 'buy' if ml_signal == 1 else 'sell'
        
        signal = Signal(
            symbol=symbol,
            action=action,
            confidence=ml_confidence,
            regime=regime,
            entry_price=current_price,
            reason=f"Strong momentum in {regime} regime"
        )
        
        if self.validate_signal(signal):
            return signal
        
        return None
    
    def _check_momentum(self, features: Dict, ml_signal: int) -> bool:
        """Check momentum indicators."""
        rsi = features.get('rsi')
        macd_hist = features.get('macd_hist')
        
        if rsi is None or macd_hist is None:
            return False
        
        if ml_signal == 1:  # Long
            # RSI not overbought, MACD positive
            if rsi < 70 and macd_hist > 0:
                return True
        
        elif ml_signal == -1:  # Short
            # RSI not oversold, MACD negative
            if rsi > 30 and macd_hist < 0:
                return True
        
        return False
