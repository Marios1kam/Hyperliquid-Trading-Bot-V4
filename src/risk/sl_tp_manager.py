"""
Stop-loss and take-profit manager.
Manages SL/TP orders with regime-aware adjustments.
"""

import logging
from typing import Optional
from dataclasses import dataclass

from src.config import config
from src.exchange.order_manager import OrderManager, OrderSide, Order


logger = logging.getLogger(__name__)


@dataclass
class SLTPLevels:
    """Stop-loss and take-profit levels."""
    stop_loss_price: float
    take_profit_price: float
    stop_loss_pct: float
    take_profit_pct: float


class SLTPManager:
    """Manage stop-loss and take-profit orders."""
    
    def __init__(self, order_manager: OrderManager):
        """
        Initialize SL/TP manager.
        
        Args:
            order_manager: Order manager instance
        """
        self.order_manager = order_manager
        
        # Base SL/TP from config
        self.base_sl_pct = config.stop_loss_percent
        self.base_tp_pct = config.take_profit_percent
        
        # Regime multipliers
        self.bull_tp_multiplier = config.get('REGIME.BULL.TAKE_PROFIT_MULTIPLIER', 1.2)
        self.bear_sl_multiplier = config.get('REGIME.BEAR.STOP_LOSS_MULTIPLIER', 1.2)
        
        # Track SL/TP orders
        self.sl_orders = {}  # symbol -> order_id
        self.tp_orders = {}  # symbol -> order_id
        
        logger.info(f"Initialized SL/TP manager: SL={self.base_sl_pct}%, TP={self.base_tp_pct}%")
    
    def calculate_levels(
        self,
        entry_price: float,
        is_long: bool,
        regime: str,
        atr: Optional[float] = None
    ) -> SLTPLevels:
        """
        Calculate SL/TP levels.
        
        Args:
            entry_price: Entry price
            is_long: True if long position
            regime: Current market regime
            atr: Optional ATR for dynamic levels
            
        Returns:
            SLTPLevels object
        """
        # Get regime-adjusted percentages
        sl_pct = self._get_sl_percentage(regime)
        tp_pct = self._get_tp_percentage(regime)
        
        # Use ATR if provided for dynamic levels
        if atr is not None:
            sl_pct = max(sl_pct, (atr / entry_price) * 100 * 1.5)  # 1.5x ATR minimum
        
        # Calculate prices
        if is_long:
            stop_loss_price = entry_price * (1 - sl_pct / 100)
            take_profit_price = entry_price * (1 + tp_pct / 100)
        else:  # Short
            stop_loss_price = entry_price * (1 + sl_pct / 100)
            take_profit_price = entry_price * (1 - tp_pct / 100)
        
        levels = SLTPLevels(
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            stop_loss_pct=sl_pct,
            take_profit_pct=tp_pct
        )
        
        logger.debug(f"Calculated SL/TP: SL={stop_loss_price:.2f} ({sl_pct:.2f}%), "
                    f"TP={take_profit_price:.2f} ({tp_pct:.2f}%)")
        
        return levels
    
    def _get_sl_percentage(self, regime: str) -> float:
        """Get stop-loss percentage for regime."""
        sl_pct = self.base_sl_pct
        
        if regime == 'BEAR':
            sl_pct *= self.bear_sl_multiplier
        elif regime == 'CHOPPY':
            sl_pct *= 0.8  # Tighter stops in choppy markets
        
        return sl_pct
    
    def _get_tp_percentage(self, regime: str) -> float:
        """Get take-profit percentage for regime."""
        tp_pct = self.base_tp_pct
        
        if regime == 'BULL':
            tp_pct *= self.bull_tp_multiplier
        elif regime in ['SIDEWAYS_QUIET', 'SIDEWAYS_VOLATILE']:
            tp_pct *= 0.8  # Smaller targets in sideways
        
        return tp_pct
    
    def place_sl_tp_orders(
        self,
        symbol: str,
        is_long: bool,
        size: float,
        levels: SLTPLevels
    ) -> tuple[Optional[Order], Optional[Order]]:
        """
        Place stop-loss and take-profit orders.
        
        Args:
            symbol: Trading symbol
            is_long: True if long position
            size: Position size
            levels: SL/TP levels
            
        Returns:
            Tuple of (sl_order, tp_order)
        """
        # Stop-loss order (opposite side, reduce only)
        sl_side = OrderSide.SELL if is_long else OrderSide.BUY
        sl_order = self.order_manager.create_limit_order(
            symbol=symbol,
            side=sl_side,
            size=size,
            price=levels.stop_loss_price,
            reduce_only=True
        )
        
        if sl_order and sl_order.order_id:
            self.sl_orders[symbol] = sl_order.order_id
            logger.info(f"Placed SL order for {symbol}: {levels.stop_loss_price:.2f}")
        
        # Take-profit order (opposite side, reduce only)
        tp_side = OrderSide.SELL if is_long else OrderSide.BUY
        tp_order = self.order_manager.create_limit_order(
            symbol=symbol,
            side=tp_side,
            size=size,
            price=levels.take_profit_price,
            reduce_only=True
        )
        
        if tp_order and tp_order.order_id:
            self.tp_orders[symbol] = tp_order.order_id
            logger.info(f"Placed TP order for {symbol}: {levels.take_profit_price:.2f}")
        
        return sl_order, tp_order
    
    def update_sl_tp(
        self,
        symbol: str,
        new_levels: SLTPLevels,
        is_long: bool,
        size: float
    ) -> bool:
        """
        Update SL/TP orders for a position.
        
        Args:
            symbol: Trading symbol
            new_levels: New SL/TP levels
            is_long: True if long position
            size: Position size
            
        Returns:
            True if successful
        """
        # Cancel existing orders
        self.cancel_sl_tp(symbol)
        
        # Place new orders
        sl_order, tp_order = self.place_sl_tp_orders(symbol, is_long, size, new_levels)
        
        return sl_order is not None and tp_order is not None
    
    def cancel_sl_tp(self, symbol: str) -> bool:
        """
        Cancel SL/TP orders for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if successful
        """
        success = True
        
        # Cancel SL order
        if symbol in self.sl_orders:
            if self.order_manager.cancel_order(self.sl_orders[symbol]):
                del self.sl_orders[symbol]
            else:
                success = False
        
        # Cancel TP order
        if symbol in self.tp_orders:
            if self.order_manager.cancel_order(self.tp_orders[symbol]):
                del self.tp_orders[symbol]
            else:
                success = False
        
        return success
    
    def implement_trailing_stop(
        self,
        symbol: str,
        current_price: float,
        entry_price: float,
        is_long: bool,
        trailing_pct: float = 1.0
    ) -> Optional[float]:
        """
        Calculate trailing stop price.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            entry_price: Entry price
            is_long: True if long position
            trailing_pct: Trailing percentage
            
        Returns:
            New stop price or None
        """
        if is_long:
            # Only trail if in profit
            if current_price > entry_price:
                new_stop = current_price * (1 - trailing_pct / 100)
                return new_stop
        else:  # Short
            if current_price < entry_price:
                new_stop = current_price * (1 + trailing_pct / 100)
                return new_stop
        
        return None
    
    def check_manual_exit(
        self,
        symbol: str,
        current_price: float,
        entry_price: float,
        is_long: bool,
        levels: SLTPLevels
    ) -> Optional[str]:
        """
        Check if position should be manually exited.
        
        Args:
            symbol: Trading symbol
            current_price: Current price
            entry_price: Entry price
            is_long: True if long position
            levels: SL/TP levels
            
        Returns:
            Exit reason or None
        """
        if is_long:
            if current_price <= levels.stop_loss_price:
                return "stop_loss"
            elif current_price >= levels.take_profit_price:
                return "take_profit"
        else:  # Short
            if current_price >= levels.stop_loss_price:
                return "stop_loss"
            elif current_price <= levels.take_profit_price:
                return "take_profit"
        
        return None
    
    def get_summary(self) -> dict:
        """Get SL/TP manager summary."""
        return {
            'active_sl_orders': len(self.sl_orders),
            'active_tp_orders': len(self.tp_orders),
            'symbols_with_sl_tp': list(set(self.sl_orders.keys()) | set(self.tp_orders.keys())),
        }
