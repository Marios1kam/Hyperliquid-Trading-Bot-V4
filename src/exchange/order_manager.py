"""
Order management system.
Handles order lifecycle and tracking.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum

from src.config import config
from src.exchange.hyperliquid_client import HyperliquidClient


logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"


class Order:
    """Represents a trading order."""
    
    def __init__(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        size: float,
        price: Optional[float] = None,
        reduce_only: bool = False
    ):
        """Initialize order."""
        self.symbol = symbol
        self.side = side
        self.order_type = order_type
        self.size = size
        self.price = price
        self.reduce_only = reduce_only
        
        self.order_id = None
        self.status = OrderStatus.PENDING
        self.filled_size = 0.0
        self.filled_price = 0.0
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.error_message = None
    
    def to_dict(self) -> Dict:
        """Convert order to dictionary."""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'type': self.order_type.value,
            'size': self.size,
            'price': self.price,
            'reduce_only': self.reduce_only,
            'status': self.status.value,
            'filled_size': self.filled_size,
            'filled_price': self.filled_price,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'error_message': self.error_message,
        }


class OrderManager:
    """Manage order lifecycle and tracking."""
    
    def __init__(self, client: HyperliquidClient):
        """
        Initialize order manager.
        
        Args:
            client: Hyperliquid client instance
        """
        self.client = client
        self.orders: Dict[str, Order] = {}  # order_id -> Order
        self.active_orders: Dict[str, List[str]] = {}  # symbol -> [order_ids]
        
        logger.info("Initialized order manager")
    
    def create_market_order(
        self,
        symbol: str,
        side: OrderSide,
        size: float,
        reduce_only: bool = False
    ) -> Optional[Order]:
        """
        Create and submit a market order.
        
        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            size: Order size
            reduce_only: If True, only reduce position
            
        Returns:
            Order object or None
        """
        order = Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            size=size,
            reduce_only=reduce_only
        )
        
        logger.info(f"Creating market order: {side.value} {size} {symbol}")
        
        # Submit to exchange
        result = self.client.place_market_order(
            symbol=symbol,
            side=side.value,
            size=size,
            reduce_only=reduce_only
        )
        
        if result:
            order.order_id = result.get('order_id')
            order.status = OrderStatus.FILLED if result.get('status') == 'filled' else OrderStatus.OPEN
            order.filled_price = result.get('filled_price', 0.0)
            order.filled_size = result.get('filled_size', size if order.status == OrderStatus.FILLED else 0.0)
            order.updated_at = datetime.now()
            
            # Store order
            if order.order_id:
                self.orders[order.order_id] = order
                self._add_active_order(symbol, order.order_id)
            
            logger.info(f"Market order created: {order.order_id} status={order.status.value}")
            return order
        else:
            order.status = OrderStatus.REJECTED
            order.error_message = "Failed to submit order"
            logger.error(f"Failed to create market order for {symbol}")
            return None
    
    def create_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        size: float,
        price: float,
        reduce_only: bool = False,
        post_only: bool = False
    ) -> Optional[Order]:
        """
        Create and submit a limit order.
        
        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            size: Order size
            price: Limit price
            reduce_only: If True, only reduce position
            post_only: If True, ensure maker order
            
        Returns:
            Order object or None
        """
        order = Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            size=size,
            price=price,
            reduce_only=reduce_only
        )
        
        logger.info(f"Creating limit order: {side.value} {size} {symbol} @ {price}")
        
        # Submit to exchange
        result = self.client.place_limit_order(
            symbol=symbol,
            side=side.value,
            size=size,
            price=price,
            reduce_only=reduce_only,
            post_only=post_only
        )
        
        if result:
            order.order_id = result.get('order_id')
            order.status = OrderStatus.OPEN
            order.updated_at = datetime.now()
            
            # Store order
            if order.order_id:
                self.orders[order.order_id] = order
                self._add_active_order(symbol, order.order_id)
            
            logger.info(f"Limit order created: {order.order_id}")
            return order
        else:
            order.status = OrderStatus.REJECTED
            order.error_message = "Failed to submit order"
            logger.error(f"Failed to create limit order for {symbol}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if successful
        """
        if order_id not in self.orders:
            logger.warning(f"Order {order_id} not found")
            return False
        
        order = self.orders[order_id]
        
        if order.status not in [OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]:
            logger.warning(f"Cannot cancel order {order_id} with status {order.status.value}")
            return False
        
        success = self.client.cancel_order(order_id, order.symbol)
        
        if success:
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.now()
            self._remove_active_order(order.symbol, order_id)
            logger.info(f"Order {order_id} cancelled")
        
        return success
    
    def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Cancel all orders for a symbol or all symbols.
        
        Args:
            symbol: Optional symbol to cancel orders for
            
        Returns:
            Number of orders cancelled
        """
        cancelled_count = 0
        
        if symbol:
            order_ids = self.active_orders.get(symbol, []).copy()
        else:
            order_ids = list(self.orders.keys())
        
        for order_id in order_ids:
            if self.cancel_order(order_id):
                cancelled_count += 1
        
        logger.info(f"Cancelled {cancelled_count} orders for {symbol or 'all symbols'}")
        return cancelled_count
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)
    
    def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Get active orders.
        
        Args:
            symbol: Optional symbol to filter by
            
        Returns:
            List of active orders
        """
        if symbol:
            order_ids = self.active_orders.get(symbol, [])
            return [self.orders[oid] for oid in order_ids if oid in self.orders]
        else:
            return [
                order for order in self.orders.values()
                if order.status in [OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]
            ]
    
    def get_order_history(self, symbol: Optional[str] = None, limit: int = 100) -> List[Order]:
        """
        Get order history.
        
        Args:
            symbol: Optional symbol to filter by
            limit: Maximum number of orders to return
            
        Returns:
            List of orders
        """
        orders = list(self.orders.values())
        
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        
        # Sort by creation time (newest first)
        orders.sort(key=lambda x: x.created_at, reverse=True)
        
        return orders[:limit]
    
    def _add_active_order(self, symbol: str, order_id: str):
        """Add order to active orders tracking."""
        if symbol not in self.active_orders:
            self.active_orders[symbol] = []
        if order_id not in self.active_orders[symbol]:
            self.active_orders[symbol].append(order_id)
    
    def _remove_active_order(self, symbol: str, order_id: str):
        """Remove order from active orders tracking."""
        if symbol in self.active_orders and order_id in self.active_orders[symbol]:
            self.active_orders[symbol].remove(order_id)
    
    def get_summary(self) -> Dict:
        """Get order manager summary."""
        total_orders = len(self.orders)
        active_orders = len(self.get_active_orders())
        
        status_counts = {}
        for order in self.orders.values():
            status = order.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            'total_orders': total_orders,
            'active_orders': active_orders,
            'status_counts': status_counts,
            'symbols_with_active_orders': list(self.active_orders.keys()),
        }
