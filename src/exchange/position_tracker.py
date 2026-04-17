"""
Position tracker.
Monitors and manages open positions.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass

from src.config import config
from src.exchange.hyperliquid_client import HyperliquidClient


logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    size: float  # Positive for long, negative for short
    entry_price: float
    current_price: float
    leverage: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    liquidation_price: Optional[float]
    margin_used: float
    opened_at: datetime
    updated_at: datetime
    
    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.size > 0
    
    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.size < 0
    
    @property
    def notional_value(self) -> float:
        """Get notional value of position."""
        return abs(self.size) * self.current_price
    
    def to_dict(self) -> Dict:
        """Convert position to dictionary."""
        return {
            'symbol': self.symbol,
            'size': self.size,
            'side': 'long' if self.is_long else 'short',
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'leverage': self.leverage,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pnl_pct': self.unrealized_pnl_pct,
            'liquidation_price': self.liquidation_price,
            'margin_used': self.margin_used,
            'notional_value': self.notional_value,
            'opened_at': self.opened_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
        }


class PositionTracker:
    """Track and manage open positions."""
    
    def __init__(self, client: HyperliquidClient):
        """
        Initialize position tracker.
        
        Args:
            client: Hyperliquid client instance
        """
        self.client = client
        self.positions: Dict[str, Position] = {}  # symbol -> Position
        self.max_positions = config.max_positions
        
        logger.info(f"Initialized position tracker: max_positions={self.max_positions}")
    
    def update_positions(self) -> bool:
        """
        Update positions from exchange.
        
        Returns:
            True if successful
        """
        try:
            exchange_positions = self.client.get_positions()
            
            # Keep track of active symbols to identify closed positions
            active_symbols = set()
            
            # Update with exchange positions
            for pos_data in exchange_positions:
                new_pos = self._parse_position(pos_data)
                if not new_pos:
                    continue
                
                # Ignore dust positions (< $2 notional)
                if new_pos.notional_value <= 2.0:
                    logger.debug(f"Ignoring dust position for {new_pos.symbol}: ${new_pos.notional_value:.2f}")
                    continue
                
                symbol = new_pos.symbol
                active_symbols.add(symbol)
                
                # RECONCILE: If we already track this symbol, preserve the 'opened_at' time
                if symbol in self.positions:
                    existing_pos = self.positions[symbol]
                    # Update fields but keep original opened_at
                    new_pos.opened_at = existing_pos.opened_at
                
                self.positions[symbol] = new_pos
            
            # Remove positions that are no longer active on the exchange
            current_tracked_symbols = list(self.positions.keys())
            for symbol in current_tracked_symbols:
                if symbol not in active_symbols:
                    logger.info(f"Position for {symbol} closed (no longer on exchange)")
                    del self.positions[symbol]
            
            logger.debug(f"Updated {len(self.positions)} positions")
            return True
            
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
            return False
    
    def _parse_position(self, pos_data: Dict) -> Optional[Position]:
        """Parse position data from exchange."""
        try:
            symbol = pos_data['symbol']
            size = float(pos_data['size'])
            entry_price = float(pos_data['entry_price'])
            current_price = float(pos_data.get('current_price', entry_price))
            
            # Calculate PnL
            if size > 0:  # Long
                unrealized_pnl = (current_price - entry_price) * size
            else:  # Short
                unrealized_pnl = (entry_price - current_price) * abs(size)
            
            unrealized_pnl_pct = (unrealized_pnl / (entry_price * abs(size))) * 100
            
            position = Position(
                symbol=symbol,
                size=size,
                entry_price=entry_price,
                current_price=current_price,
                leverage=float(pos_data.get('leverage', config.leverage)),
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_pct=unrealized_pnl_pct,
                liquidation_price=pos_data.get('liquidation_price'),
                margin_used=float(pos_data.get('margin_used', 0.0)),
                opened_at=datetime.fromisoformat(pos_data.get('opened_at', datetime.now().isoformat())),
                updated_at=datetime.now()
            )
            
            return position
            
        except Exception as e:
            logger.error(f"Error parsing position data: {e}")
            return None
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Position or None
        """
        return self.positions.get(symbol)
    
    def has_position(self, symbol: str) -> bool:
        """Check if there's an open position for a symbol."""
        return symbol in self.positions
    
    def get_all_positions(self) -> List[Position]:
        """Get all open positions."""
        return list(self.positions.values())
    
    def get_position_count(self) -> int:
        """Get number of open positions."""
        return len(self.positions)
    
    def can_open_position(self) -> bool:
        """Check if we can open a new position."""
        return self.get_position_count() < self.max_positions
    
    def get_total_pnl(self) -> float:
        """Get total unrealized PnL across all positions."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    def get_total_margin_used(self) -> float:
        """Get total margin used across all positions."""
        return sum(pos.margin_used for pos in self.positions.values())
    
    def get_total_notional(self) -> float:
        """Get total notional value across all positions."""
        return sum(pos.notional_value for pos in self.positions.values())
    
    def get_long_positions(self) -> List[Position]:
        """Get all long positions."""
        return [pos for pos in self.positions.values() if pos.is_long]
    
    def get_short_positions(self) -> List[Position]:
        """Get all short positions."""
        return [pos for pos in self.positions.values() if pos.is_short]
    
    def close_position(self, symbol: str) -> bool:
        """
        Close a position.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if successful
        """
        if not self.has_position(symbol):
            logger.warning(f"No position to close for {symbol}")
            return True
        
        success = self.client.close_position(symbol)
        
        if success:
            # Remove from tracking
            if symbol in self.positions:
                del self.positions[symbol]
            logger.info(f"Closed position for {symbol}")
        
        return success
    
    def close_all_positions(self) -> bool:
        """
        Close all positions.
        
        Returns:
            True if all positions closed successfully
        """
        if not self.positions:
            logger.info("No positions to close")
            return True
        
        symbols = list(self.positions.keys())
        success = True
        
        for symbol in symbols:
            if not self.close_position(symbol):
                success = False
        
        return success
    
    def get_position_summary(self) -> Dict:
        """Get summary of all positions."""
        positions = self.get_all_positions()
        
        if not positions:
            return {
                'position_count': 0,
                'total_pnl': 0.0,
                'total_margin_used': 0.0,
                'total_notional': 0.0,
            }
        
        long_positions = self.get_long_positions()
        short_positions = self.get_short_positions()
        
        return {
            'position_count': len(positions),
            'long_count': len(long_positions),
            'short_count': len(short_positions),
            'total_pnl': self.get_total_pnl(),
            'total_margin_used': self.get_total_margin_used(),
            'total_notional': self.get_total_notional(),
            'positions': [pos.to_dict() for pos in positions],
        }
    
    def check_liquidation_risk(self) -> List[str]:
        """
        Check positions at risk of liquidation.
        
        Returns:
            List of symbols at risk
        """
        at_risk = []
        
        for symbol, position in self.positions.items():
            if position.liquidation_price is None:
                continue
            
            # Check if current price is within 10% of liquidation price
            if position.is_long:
                distance_pct = (position.current_price - position.liquidation_price) / position.current_price
            else:
                distance_pct = (position.liquidation_price - position.current_price) / position.current_price
            
            if distance_pct < 0.1:  # Within 10% of liquidation
                at_risk.append(symbol)
                logger.warning(f"Position {symbol} at liquidation risk: "
                             f"current={position.current_price}, "
                             f"liquidation={position.liquidation_price}")
        
        return at_risk
