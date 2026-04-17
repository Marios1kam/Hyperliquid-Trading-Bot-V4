"""
Position sizer.
Calculates optimal position sizes based on risk parameters and regime.
All parameters loaded from configuration.
"""

import logging
from typing import Optional

from src.config import config


logger = logging.getLogger(__name__)


class PositionSizer:
    """Calculate position sizes with regime-aware adjustments."""
    
    def __init__(self):
        """Initialize position sizer with configuration."""
        self.base_position_size_pct = config.position_size_percent
        self.max_position_size_pct = config.get('MAX_POSITION_SIZE_PERCENT', 5.0)
        self.leverage = config.leverage
        
        # Regime multipliers
        self.bull_multiplier = config.get('REGIME.BULL.POSITION_MULTIPLIER', 1.0)
        self.bear_multiplier = config.get('REGIME.BEAR.POSITION_MULTIPLIER', 0.8)
        self.sideways_multiplier = config.get('SIDEWAYS.POSITION_MULTIPLIER', 0.6)
        
        logger.info(f"Initialized position sizer: base={self.base_position_size_pct}%, "
                   f"max={self.max_position_size_pct}%, leverage={self.leverage}x")
    
    def calculate_position_size(
        self,
        account_equity: float,
        regime: str,
        confidence: float,
        volatility: Optional[float] = None
    ) -> float:
        """
        Calculate position size.
        
        Args:
            account_equity: Total account equity
            regime: Current market regime
            confidence: ML model confidence (0-1)
            volatility: Optional volatility measure for scaling
            
        Returns:
            Position size in quote currency
        """
        # Start with base size
        size_pct = self.base_position_size_pct
        
        # Apply regime multiplier
        regime_multiplier = self._get_regime_multiplier(regime)
        size_pct *= regime_multiplier
        
        # Apply confidence scaling
        confidence_multiplier = self._get_confidence_multiplier(confidence)
        size_pct *= confidence_multiplier
        
        # Apply volatility scaling if provided
        if volatility is not None:
            vol_multiplier = self._get_volatility_multiplier(volatility)
            size_pct *= vol_multiplier
        
        # Cap at maximum
        size_pct = min(size_pct, self.max_position_size_pct)
        
        # Calculate actual size
        position_size = (account_equity * size_pct / 100.0) * self.leverage
        
        logger.debug(f"Position size: {size_pct:.2f}% of equity = ${position_size:.2f} "
                    f"(regime={regime}, confidence={confidence:.3f})")
        
        return position_size
    
    def _get_regime_multiplier(self, regime: str) -> float:
        """Get position size multiplier for regime."""
        regime_multipliers = {
            'BULL': self.bull_multiplier,
            'BEAR': self.bear_multiplier,
            'SIDEWAYS_QUIET': self.sideways_multiplier,
            'SIDEWAYS_VOLATILE': self.sideways_multiplier,
            'CHOPPY': 0.5,  # Reduce size in choppy markets
        }
        
        return regime_multipliers.get(regime, 1.0)
    
    def _get_confidence_multiplier(self, confidence: float) -> float:
        """
        Get position size multiplier based on confidence.
        
        Linear scaling: confidence 0.6 -> 0.6x, confidence 1.0 -> 1.0x
        """
        min_confidence = config.min_confidence
        
        if confidence < min_confidence:
            return 0.0
        
        # Scale from min_confidence to 1.0
        multiplier = (confidence - min_confidence) / (1.0 - min_confidence)
        multiplier = 0.5 + (multiplier * 0.5)  # Scale to 0.5-1.0 range
        
        return multiplier
    
    def _get_volatility_multiplier(self, volatility: float) -> float:
        """
        Get position size multiplier based on volatility.
        
        Higher volatility -> smaller position size
        """
        # Normalize volatility (assuming typical range 0-0.1)
        normalized_vol = min(volatility / 0.1, 1.0)
        
        # Inverse relationship: high vol -> low multiplier
        multiplier = 1.0 - (normalized_vol * 0.5)  # Range: 0.5-1.0
        
        return max(multiplier, 0.5)
    
    def calculate_quantity(
        self,
        position_size_usd: float,
        price: float,
        min_quantity: float = 0.001
    ) -> float:
        """
        Convert position size in USD to quantity.
        
        Args:
            position_size_usd: Position size in USD
            price: Current price
            min_quantity: Minimum order quantity
            
        Returns:
            Order quantity
        """
        quantity = position_size_usd / price
        
        # Round to appropriate precision
        quantity = round(quantity, 8)
        
        # Ensure minimum quantity
        if quantity < min_quantity:
            logger.warning(f"Calculated quantity {quantity} below minimum {min_quantity}")
            return 0.0
        
        return quantity
    
    def validate_position_size(
        self,
        position_size: float,
        account_equity: float,
        current_margin_used: float
    ) -> bool:
        """
        Validate that position size is within limits.
        
        Args:
            position_size: Proposed position size
            account_equity: Total account equity
            current_margin_used: Currently used margin
            
        Returns:
            True if valid
        """
        # Check maximum position size
        max_size = account_equity * self.max_position_size_pct / 100.0
        if position_size > max_size:
            logger.warning(f"Position size {position_size} exceeds maximum {max_size}")
            return False
        
        # Check available margin
        required_margin = position_size / self.leverage
        available_margin = account_equity - current_margin_used
        
        if required_margin > available_margin:
            logger.warning(f"Insufficient margin: required={required_margin}, "
                         f"available={available_margin}")
            return False
        
        return True
