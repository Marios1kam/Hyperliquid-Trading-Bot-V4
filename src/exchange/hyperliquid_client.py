"""
Hyperliquid API client with actual SDK integration.
Handles all interactions with Hyperliquid exchange.
All settings loaded from configuration.
"""

import os
import logging
import time
from typing import Dict, List, Optional, Tuple
from decimal import Decimal

from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants

from src.config import config


logger = logging.getLogger(__name__)


class HyperliquidClient:
    """
    Hyperliquid exchange API client with full SDK integration.
    
    Supports both testnet and mainnet via configuration.
    """
    
    def __init__(self):
        """Initialize Hyperliquid client."""
        # Get configuration
        self.use_testnet = config.use_testnet
        self.app_env = os.getenv('APP_ENV', config.get('APP_ENV', 'testnet'))
        
        # Get credentials from environment
        self.private_key = os.getenv('HYPERLIQUID_PRIVATE_KEY')
        self.wallet_address = os.getenv('HYPERLIQUID_WALLET_ADDRESS')
        
        # Connection settings
        self.timeout = config.get('API_TIMEOUT', 30)
        self.max_retries = config.get('API_MAX_RETRIES', 3)
        
        # Determine API URL
        if self.use_testnet:
            self.base_url = constants.TESTNET_API_URL
        else:
            self.base_url = constants.MAINNET_API_URL
        
        # Initialize SDK components
        self.info = None
        self.exchange = None
        self._initialize_client()
        
        # Size/price precision for Hyperliquid (CRITICAL for order acceptance)
        # These are fetched directly from Hyperliquid API - DO NOT MODIFY without checking API
        self.size_precision = {
            'BTC': {'decimals': 5, 'step': 0.00001, 'tick': 0.0001, 'min_size': 0.00001},
            'SOL': {'decimals': 2, 'step': 0.01, 'tick': 0.01, 'min_size': 0.01},
            'XRP': {'decimals': 0, 'step': 1, 'tick': 0.001, 'min_size': 1},
            'HBAR': {'decimals': 0, 'step': 1, 'tick': 0.01, 'min_size': 1},
            'XLM': {'decimals': 0, 'step': 1, 'tick': 0.01, 'min_size': 1},
            'BNB': {'decimals': 3, 'step': 0.001, 'tick': 0.001, 'min_size': 0.001},
            'TON': {'decimals': 1, 'step': 0.1, 'tick': 0.01, 'min_size': 0.1},
            'DOGE': {'decimals': 0, 'step': 1, 'tick': 0.01, 'min_size': 1},
            'ADA': {'decimals': 0, 'step': 1, 'tick': 0.01, 'min_size': 1},
            'TRX': {'decimals': 0, 'step': 1, 'tick': 0.00001, 'min_size': 1},
            'AVAX': {'decimals': 2, 'step': 0.01, 'tick': 0.01, 'min_size': 0.01},
            'LTC': {'decimals': 2, 'step': 0.01, 'tick': 0.01, 'min_size': 0.01},
            'LINK': {'decimals': 1, 'step': 0.1, 'tick': 0.01, 'min_size': 0.1},
            'DOT': {'decimals': 1, 'step': 0.1, 'tick': 0.01, 'min_size': 0.1},
            'UNI': {'decimals': 1, 'step': 0.1, 'tick': 0.01, 'min_size': 0.1},
            'ICP': {'decimals': 2, 'step': 0.01, 'tick': 0.01, 'min_size': 0.01},
            'FIL': {'decimals': 1, 'step': 0.1, 'tick': 0.01, 'min_size': 0.1},
            'AAVE': {'decimals': 2, 'step': 0.01, 'tick': 0.01, 'min_size': 0.01},
            'ETC': {'decimals': 2, 'step': 0.01, 'tick': 0.01, 'min_size': 0.01},
            'ARB': {'decimals': 1, 'step': 0.1, 'tick': 0.01, 'min_size': 0.1},
            'OP': {'decimals': 1, 'step': 0.1, 'tick': 0.01, 'min_size': 0.1},
            'NEAR': {'decimals': 1, 'step': 0.1, 'tick': 0.01, 'min_size': 0.1},
            'SUI': {'decimals': 1, 'step': 0.1, 'tick': 0.01, 'min_size': 0.1},
            'HYPE': {'decimals': 2, 'step': 0.01, 'tick': 0.01, 'min_size': 0.01},
            'CAKE': {'decimals': 1, 'step': 0.1, 'tick': 0.01, 'min_size': 0.1},
            'MKR': {'decimals': 4, 'step': 0.0001, 'tick': 0.0001, 'min_size': 0.0001},
            'GALA': {'decimals': 0, 'step': 1, 'tick': 0.01, 'min_size': 1},
            'RUNE': {'decimals': 1, 'step': 0.1, 'tick': 0.01, 'min_size': 0.1},
            'XTZ': {'decimals': 1, 'step': 0.1, 'tick': 0.01, 'min_size': 0.1},
            'SNX': {'decimals': 1, 'step': 0.1, 'tick': 0.01, 'min_size': 0.1},
            'ZEC': {'decimals': 0, 'step': 1, 'tick': 0.01, 'min_size': 1},
            # Default for unknown coins
            'DEFAULT': {'decimals': 1, 'step': 0.1, 'tick': 0.001, 'min_size': 0.1}
        }
        
        # Cache of available symbols (fetched on first use)
        self._available_symbols = None
        
        logger.info(f"Initialized Hyperliquid client: "
                   f"testnet={self.use_testnet}, url={self.base_url}")
    
    def _initialize_client(self):
        """Initialize the Hyperliquid SDK client."""
        try:
            # Pre-fetch and sanitize spot_meta to avoid "list index out of range"
            # on testnet, which returns spot market entries with out-of-bounds token indices.
            safe_spot_meta = self._fetch_safe_spot_meta()

            # Initialize Info client (read-only, no credentials needed)
            # Pass pre-validated spot_meta so the SDK doesn't re-fetch the broken data
            self.info = Info(self.base_url, skip_ws=True, spot_meta=safe_spot_meta)
            logger.info("Hyperliquid Info client initialized")

            # Initialize Exchange client (requires credentials)
            if self.private_key and self.wallet_address:
                # Create account object from private key (same as working bot)
                from eth_account import Account
                account = Account.from_key(self.private_key)

                # Initialize Exchange with account object and pre-validated spot_meta
                # so the Exchange's internal Info doesn't re-fetch broken testnet data
                self.exchange = Exchange(
                    account,
                    self.base_url,
                    account_address=self.wallet_address,
                    spot_meta=safe_spot_meta,
                )
                logger.info("Hyperliquid Exchange client initialized successfully")
            else:
                logger.warning("Hyperliquid credentials not found in environment")
                logger.warning("Trading will be disabled - set HYPERLIQUID_PRIVATE_KEY and HYPERLIQUID_WALLET_ADDRESS")

        except Exception as e:
            logger.error(f"Failed to initialize Hyperliquid client: {e}")
            self.info = None
            self.exchange = None

    def _fetch_safe_spot_meta(self) -> dict:
        """
        Fetch spot metadata from the API and sanitize it to prevent
        'list index out of range' errors caused by inconsistent testnet data.

        The Hyperliquid testnet sometimes returns spot market entries whose
        token indices (base/quote) exceed the length of the tokens list.
        This method filters those bad entries out so the SDK can initialize safely.

        Returns:
            A sanitized spot_meta dict safe to pass to Info() and Exchange().
        """
        try:
            from hyperliquid.api import API
            api = API(self.base_url)
            raw = api.post("/info", {"type": "spotMeta"})

            tokens = raw.get("tokens", [])
            universe = raw.get("universe", [])
            n_tokens = len(tokens)

            sanitized_universe = []
            skipped = 0
            for entry in universe:
                token_indices = entry.get("tokens", [])
                if len(token_indices) >= 2:
                    base_idx, quote_idx = token_indices[0], token_indices[1]
                    if base_idx < n_tokens and quote_idx < n_tokens:
                        sanitized_universe.append(entry)
                    else:
                        skipped += 1
                        logger.debug(
                            f"Skipping spot market '{entry.get('name')}' — token indices "
                            f"[{base_idx}, {quote_idx}] out of range (tokens list has {n_tokens} entries)"
                        )
                else:
                    skipped += 1

            if skipped:
                logger.warning(
                    f"Sanitized spot_meta: removed {skipped} invalid spot market entries "
                    f"(out-of-range token indices). This is common on testnet."
                )

            return {"universe": sanitized_universe, "tokens": tokens}

        except Exception as e:
            logger.warning(f"Could not fetch spot_meta, using empty fallback: {e}")
            # Return a safe empty spot_meta — perp trading will still work
            return {"universe": [], "tokens": []}
    
    def is_connected(self) -> bool:
        """Check if client is connected and ready."""
        return self.info is not None
    
    def can_trade(self) -> bool:
        """Check if client can execute trades."""
        return self.exchange is not None and self.wallet_address is not None
    
    def quantize_size(self, symbol: str, size: float, round_up_to_min: bool = False) -> float:
        """
        Quantize order size to meet Hyperliquid's requirements.
        CRITICAL: Orders will be rejected if size doesn't match exchange specs.
        
        Args:
            symbol: Trading symbol
            size: Desired order size
            round_up_to_min: If True, round up to minimum size instead of returning 0
            
        Returns:
            Quantized size (or 0 if below minimum and round_up_to_min=False)
        """
        info = self.size_precision.get(symbol, self.size_precision['DEFAULT'])
        step = info['step']
        decimals = info['decimals']
        min_size = info['min_size']
        
        # Round DOWN to nearest step (always round down for safety)
        quantized = int(size / step) * step
        quantized = round(quantized, decimals)
        
        # Handle below minimum
        if quantized < min_size:
            if round_up_to_min:
                logger.info(f"{symbol}: Size {size:.4f} below minimum {min_size}, rounding up to {min_size}")
                return min_size
            else:
                logger.warning(f"{symbol}: Size {size} below minimum {min_size}, returning 0")
                return 0
        
        return quantized
    
    def is_symbol_available(self, symbol: str) -> bool:
        """
        Check if symbol is available on Hyperliquid.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if symbol is available for trading
        """
        # Lazy load available symbols on first call
        if self._available_symbols is None:
            try:
                meta = self.info.meta()
                self._available_symbols = {asset['name'] for asset in meta['universe']}
                logger.info(f"Loaded {len(self._available_symbols)} available symbols from Hyperliquid")
            except Exception as e:
                logger.error(f"Failed to fetch available symbols: {e}")
                self._available_symbols = set()  # Empty set to avoid repeated failures
        
        return symbol in self._available_symbols
    
    def quantize_price(self, symbol: str, price: float) -> float:
        """
        Quantize price to meet Hyperliquid's tick size requirements.
        CRITICAL: Orders will be rejected if price doesn't match tick size.
        
        Args:
            symbol: Trading symbol
            price: Desired price
            
        Returns:
            Quantized price
        """
        info = self.size_precision.get(symbol, self.size_precision['DEFAULT'])
        tick = info['tick']
        
        # Price must be a multiple of tick size
        quantized = round(price / tick) * tick
        # Round to 6 decimals for safety
        return round(quantized, 6)
    
    def get_account_info(self) -> Optional[Dict]:
        """
        Get account information.
        
        Returns:
            Dictionary with account info or None
        """
        if not self.is_connected() or not self.wallet_address:
            logger.error("Client not connected or no wallet address")
            return None
        
        try:
            # Get user state from Info API
            user_state = self.info.user_state(self.wallet_address)
            
            if not user_state:
                return None
            
            # Extract account info
            margin_summary = user_state.get('marginSummary', {})
            
            account_info = {
                'wallet_address': self.wallet_address,
                'account_value': float(margin_summary.get('accountValue', 0)),
                'total_margin_used': float(margin_summary.get('totalMarginUsed', 0)),
                'total_ntl_pos': float(margin_summary.get('totalNtlPos', 0)),
                'total_raw_usd': float(margin_summary.get('totalRawUsd', 0)),
                'withdrawable': float(user_state.get('withdrawable', 0)),
            }
            
            # Calculate available margin
            account_info['available_margin'] = (
                account_info['account_value'] - account_info['total_margin_used']
            )
            
            # Legacy field names for compatibility
            account_info['equity'] = account_info['account_value']
            account_info['margin_used'] = account_info['total_margin_used']
            
            return account_info
            
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None
    
    def get_positions(self) -> List[Dict]:
        """
        Get all open positions.
        
        Returns:
            List of position dictionaries
        """
        if not self.is_connected() or not self.wallet_address:
            logger.error("Client not connected or no wallet address")
            return []
        
        try:
            # Get user state
            user_state = self.info.user_state(self.wallet_address)
            
            if not user_state:
                return []
            
            # Extract positions
            asset_positions = user_state.get('assetPositions', [])
            
            positions = []
            for pos in asset_positions:
                position = pos.get('position', {})
                
                # Skip if no position
                coin = position.get('coin')
                szi = float(position.get('szi', 0))
                
                if not coin or szi == 0:
                    continue
                
                # Parse position data
                entry_px = float(position.get('entryPx', 0))
                position_value = float(position.get('positionValue', 0))
                unrealized_pnl = float(position.get('unrealizedPnl', 0))
                return_on_equity = float(position.get('returnOnEquity', 0))
                leverage = float(position.get('leverage', {}).get('value', 1))
                liquidation_px = position.get('liquidationPx')
                
                positions.append({
                    'symbol': coin,
                    'size': szi,
                    'entry_price': entry_px,
                    'current_price': entry_px + (unrealized_pnl / szi) if szi != 0 else entry_px,
                    'position_value': position_value,
                    'unrealized_pnl': unrealized_pnl,
                    'return_on_equity': return_on_equity,
                    'leverage': leverage,
                    'liquidation_price': float(liquidation_px) if liquidation_px else None,
                    'margin_used': abs(position_value / leverage) if leverage != 0 else 0,
                })
            
            return positions
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """
        Get position for a specific symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Position dictionary or None
        """
        positions = self.get_positions()
        for pos in positions:
            if pos.get('symbol') == symbol:
                return pos
        return None
    
    def get_market_price(self, symbol: str) -> Optional[float]:
        """
        Get current market price for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current price or None
        """
        if not self.is_connected():
            logger.error("Client not connected")
            return None
        
        try:
            # Get all mids (market prices)
            all_mids = self.info.all_mids()
            
            if symbol in all_mids:
                return float(all_mids[symbol])
            
            logger.warning(f"Symbol {symbol} not found in market data")
            return None
            
        except Exception as e:
            logger.error(f"Error getting market price for {symbol}: {e}")
            return None
    
    def get_order_book(self, symbol: str, n_levels: int = 10) -> Optional[Dict]:
        """
        Get the order book (L2 snapshot) for a symbol.

        Args:
            symbol: Trading symbol (e.g. 'BTC')
            n_levels: Number of price levels to return on each side

        Returns:
            Dictionary with 'bids' and 'asks' lists of [price, size] pairs,
            or None on failure.
        """
        if not self.is_connected():
            logger.error("Client not connected")
            return None

        try:
            raw = self.info.l2_snapshot(symbol)

            # The SDK returns a dict with 'levels': [bids_list, asks_list]
            # Each entry is {'px': str, 'sz': str, 'n': int}
            if not raw or 'levels' not in raw:
                logger.warning(f"Empty or unexpected order book response for {symbol}: {raw}")
                return None

            levels = raw['levels']
            if len(levels) < 2:
                logger.warning(f"Malformed order book levels for {symbol}: {levels}")
                return None

            def _parse_side(side_levels):
                result = []
                for entry in side_levels[:n_levels]:
                    try:
                        result.append([float(entry['px']), float(entry['sz'])])
                    except (KeyError, ValueError, TypeError) as parse_err:
                        logger.debug(f"Skipping malformed order book entry for {symbol}: {entry} ({parse_err})")
                return result

            bids = _parse_side(levels[0])
            asks = _parse_side(levels[1])

            if not bids or not asks:
                logger.warning(f"Order book for {symbol} has empty bids or asks after parsing")
                return None

            return {'bids': bids, 'asks': asks}

        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {e}")
            return None

    def place_market_order(
        self,
        symbol: str,
        side: str,
        size: float,
        reduce_only: bool = False
    ) -> Optional[Dict]:
        """
        Place a market order (implemented as IOC limit order with slippage).
        This matches the working bot's approach for better price control.
        Includes retry logic for temporary liquidity issues.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            size: Order size
            reduce_only: If True, only reduce position
            
        Returns:
            Order result dictionary or None
        """
        if not self.can_trade():
            logger.error("Exchange client not initialized - cannot trade")
            return None
        
        # Retry logic for liquidity errors
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            result = self._attempt_market_order(symbol, side, size, reduce_only)
            
            # Check if we got a retryable error
            if result and 'error' in result:
                error_msg = result['error']
                
                # Permanent errors - don't retry
                if 'Trading is halted' in error_msg or 'Invalid' in error_msg:
                    logger.warning(f"Permanent error for {symbol}: {error_msg}")
                    return result
                
                # Liquidity error - retry
                if 'could not immediately match' in error_msg:
                    if attempt < max_retries - 1:
                        logger.info(f"Liquidity issue for {symbol}, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})")
                        print(f"⏳ Retrying {symbol} order in {retry_delay}s due to low liquidity...")
                        import time
                        time.sleep(retry_delay)
                        continue
                    else:
                        logger.warning(f"Failed to execute {symbol} after {max_retries} attempts: {error_msg}")
                        return result
            
            # Success or non-retryable error
            return result
        
        return None
    
    def _attempt_market_order(
        self,
        symbol: str,
        side: str,
        size: float,
        reduce_only: bool = False
    ) -> Optional[Dict]:
        """
        Single attempt to place a market order.
        Called by place_market_order with retry logic.
        """
        try:
            # Get current market price from Hyperliquid (CRITICAL!)
            current_price = self.get_market_price(symbol)
            if not current_price:
                logger.error(f"Could not get market price for {symbol}")
                return None
            
            # Convert side to boolean (True = buy, False = sell)
            is_buy = side.lower() == 'buy'
            
            # Apply slippage to protect against adverse price movement
            # This matches the working bot's approach
            slippage = 0.01  # 1% slippage tolerance
            if is_buy:
                limit_price = current_price * (1 + slippage)
            else:
                limit_price = current_price * (1 - slippage)
            
            # CRITICAL: Quantize price and size to meet Hyperliquid requirements
            limit_price = self.quantize_price(symbol, limit_price)
            size = self.quantize_size(symbol, size)
            
            # Check minimum size
            min_size = self.size_precision.get(symbol, self.size_precision['DEFAULT'])['min_size']
            if size < min_size:
                logger.error(f"Order size {size} below minimum {min_size} for {symbol}, skipping order")
                return None
            

            
            # Place as IOC (Immediate or Cancel) limit order
            # This executes immediately like a market order but with price protection
            # SDK EXPECTS: order(name, is_buy, sz, limit_px, order_type, reduce_only, cloid=None)
            try:
                order_result = self.exchange.order(
                    symbol,           # name (positional)
                    is_buy,           # is_buy (positional)
                    size,             # sz (positional)
                    limit_price,      # limit_px (positional)
                    {'limit': {'tif': 'Ioc'}},  # order_type (positional)
                    reduce_only       # reduce_only (positional)
                )
            except TypeError as e:
                # SDK compatibility issue - try without reduce_only
                logger.warning(f"SDK TypeError for {symbol}, retrying without reduce_only: {e}")
                try:
                    order_result = self.exchange.order(
                        symbol,
                        is_buy,
                        size,
                        limit_price,
                        {'limit': {'tif': 'Ioc'}}
                    )
                except TypeError as e2:
                    # Still failing - SDK version incompatibility
                    logger.error(f"SDK TypeError persists for {symbol}: {e2}")
                    return {'error': f'SDK incompatibility: {str(e2)}', 'symbol': symbol}
            
            logger.info(f"Market order result: {order_result}")
            
            # Parse result
            if order_result and order_result.get('status') == 'ok':
                response = order_result.get('response', {})
                data = response.get('data', {})
                
                statuses = data.get('statuses', [])
                print(f"Statuses: {statuses}")
                if statuses and len(statuses) > 0:
                    status = statuses[0]
                    
                    # Check for error first
                    if 'error' in status:
                        error_msg = status['error']
                        logger.warning(f"Exchange rejected order for {symbol}: {error_msg}")
                        return {'error': error_msg, 'symbol': symbol}
                    
                    filled = status.get('filled', {})
                    
                    result = {
                        'order_id': str(status.get('oid', '')),
                        'symbol': symbol,
                        'side': side,
                        'size': size,
                        'status': 'filled' if filled else 'open',
                        'filled_price': float(filled.get('avgPx', 0)) if filled else 0.0,
                        'filled_size': float(filled.get('totalSz', 0)) if filled else 0.0,
                    }
                    print(f"Returning order result: {result}")
                    return result
            
            logger.error(f"Market order failed: {order_result}")
            return None
            
        except Exception as e:
            logger.error(f"Error placing market order: {e}")
            return None
    
    # COMMENTED OUT TO USE HYPERLIQUID DEFAULT LEVERAGE
    # def update_leverage(self, symbol: str, leverage: int, cross_margin: bool = True) -> bool:
    #     """
    #     Update leverage for a symbol.
    #     
    #     Args:
    #         symbol: Trading symbol
    #         leverage: Leverage value (integer)
    #         cross_margin: True for cross margin, False for isolated
    #         
    #     Returns:
    #         True if successful
    #     """
    #     if not self.can_trade():
    #         return False
    #         
    #     try:
    #         # Validate leverage is a proper integer
    #         leverage = int(leverage)
    #         if leverage < 1 or leverage > 50:
    #             logger.error(f"Invalid leverage value: {leverage} (must be 1-50)")
    #             return False
    #         
    #         logger.info(f"Updating leverage for {symbol} to {leverage}x (Cross: {cross_margin})")
    #         # SDK expects: update_leverage(leverage, coin, is_cross=True)
    #         result = self.exchange.update_leverage(leverage, symbol, is_cross=cross_margin)
    #         
    #         if result and result.get('status') == 'ok':
    #             return True
    #         
    #         logger.error(f"Failed to update leverage: {result}")
    #         return False
    #     except TypeError as e:
    #         # SDK version compatibility issue - log but don't fail the trade
    #         logger.warning(f"Leverage update skipped due to SDK issue for {symbol}: {e}")
    #         return True  # Return True to allow trade to proceed
    #     except Exception as e:
    #         logger.error(f"Error updating leverage: {e}")
    #         return False

    def place_tpsl_orders(self, symbol: str, entry_price: float, size: float, is_long: bool, 
                         stop_loss_pct: float, take_profit_pct: float) -> Dict:
        """
        Place Take Profit and Stop Loss orders using bulk_orders.
        Matches example_trader.py logic.
        """
        try:
            # Calculate raw prices
            if is_long:
                sl_price = entry_price * (1 - stop_loss_pct)
                tp_price = entry_price * (1 + take_profit_pct)
                sl_is_buy = False # SELL to close
                tp_is_buy = False # SELL to close
            else: # Short
                sl_price = entry_price * (1 + stop_loss_pct)
                tp_price = entry_price * (1 - take_profit_pct)
                sl_is_buy = True # BUY to close
                tp_is_buy = True # BUY to close

            # Quantize prices
            sl_price = self.quantize_price(symbol, sl_price)
            tp_price = self.quantize_price(symbol, tp_price)
            size = self.quantize_size(symbol, size)

            if size == 0:
                return {'error': 'Size 0 for TP/SL'}

            # Construct orders
            # Trigger order format for SDK bulk_orders
            sl_order = {
                "coin": symbol,
                "is_buy": sl_is_buy,
                "sz": size,
                "limit_px": sl_price,
                "order_type": {
                    "trigger": {
                        "triggerPx": sl_price,
                        "isMarket": True,
                        "tpsl": "sl"
                    }
                },
                "reduce_only": True
            }
            
            tp_order = {
                "coin": symbol,
                "is_buy": tp_is_buy,
                "sz": size,
                "limit_px": tp_price,
                "order_type": {
                    "trigger": {
                        "triggerPx": tp_price,
                        "isMarket": True,
                        "tpsl": "tp"
                    }
                },
                "reduce_only": True
            }
            
            logger.info(f"Placing TP/SL for {symbol}: SL={sl_price}, TP={tp_price}, Size={size}")
            
            # Execute bulk orders
            result = self.exchange.bulk_orders([sl_order, tp_order])
            
            if result and result.get('status') == 'ok':
                return {
                    'status': 'ok', 
                    'sl_price': sl_price, 
                    'tp_price': tp_price,
                    'result': result
                }
            
            logger.error(f"TP/SL failed: {result}")
            return {'error': 'Failed to place TP/SL', 'details': result}
            
        except Exception as e:
            logger.error(f"Error placing TP/SL: {e}")
            return {'error': str(e)}

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float,
        reduce_only: bool = False,
        post_only: bool = False
    ) -> Optional[Dict]:
        """
        Place a limit order.
        """
        if not self.can_trade():
            logger.error("Exchange client not initialized - cannot trade")
            return None
        
        try:
            logger.info(f"Placing limit order: {side} {size} {symbol} @ {price}")
            
            # Convert side to boolean
            is_buy = side.lower() == 'buy'
            
            # Place limit order via SDK using POSITIONAL arguments
            # order(name, is_buy, sz, limit_px, order_type, reduce_only)
            order_type = {'limit': {'tif': 'Gtc'}} if not post_only else {'limit': {'tif': 'Alo'}}
            
            order_result = self.exchange.order(
                symbol,      # name
                is_buy,      # is_buy
                size,        # sz
                price,       # limit_px
                order_type,  # order_type
                reduce_only  # reduce_only
            )
            
            logger.info(f"Limit order result: {order_result}")
            
            # Parse result
            if order_result and order_result.get('status') == 'ok':
                response = order_result.get('response', {})
                data = response.get('data', {})
                
                statuses = data.get('statuses', [])
                if statuses and len(statuses) > 0:
                    status = statuses[0]
                    
                    return {
                        'order_id': str(status.get('oid', '')),
                        'symbol': symbol,
                        'side': side,
                        'size': size,
                        'price': price,
                        'status': 'open',
                    }
            
            logger.error(f"Limit order failed: {order_result}")
            return None
            
        except Exception as e:
            logger.error(f"Error placing limit order: {e}")
            return None
    
    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            symbol: Trading symbol
            
        Returns:
            True if successful
        """
        if not self.can_trade():
            logger.error("Exchange client not initialized - cannot trade")
            return False
        
        try:
            logger.info(f"Cancelling order {order_id} for {symbol}")
            
            # Cancel via SDK
            result = self.exchange.cancel(
                coin=symbol,
                oid=int(order_id)
            )
            
            if result and result.get('status') == 'ok':
                logger.info(f"Order {order_id} cancelled successfully")
                return True
            
            logger.error(f"Cancel failed: {result}")
            return False
            
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False
    
    def cancel_all_orders(self, symbol: Optional[str] = None) -> bool:
        """
        Cancel all orders for a symbol or all symbols.
        
        Args:
            symbol: Optional symbol to cancel orders for
            
        Returns:
            True if successful
        """
        if not self.can_trade():
            logger.error("Exchange client not initialized - cannot trade")
            return False
        
        try:
            logger.info(f"Cancelling all orders for {symbol or 'all symbols'}")
            
            # Get open orders
            open_orders = self.info.open_orders(self.wallet_address)
            
            if not open_orders:
                logger.info("No open orders to cancel")
                return True
            
            # Filter by symbol if provided
            if symbol:
                open_orders = [o for o in open_orders if o.get('coin') == symbol]
            
            # Cancel each order
            success = True
            for order in open_orders:
                coin = order.get('coin')
                oid = order.get('oid')
                
                if not self.cancel_order(str(oid), coin):
                    success = False
            
            return success
            
        except Exception as e:
            logger.error(f"Error cancelling all orders: {e}")
            return False

    def close_all_positions(self) -> bool:
        """
        Close all positions.
        
        Returns:
            True if all positions closed successfully
        """
        positions = self.get_positions()
        if not positions:
            logger.info("No positions to close")
            return True
        
        success = True
        for pos in positions:
            symbol = pos['symbol']
            if not self.close_position(symbol):
                success = False
        
        return success
    
    def close_position(self, symbol: str) -> bool:
        """
        Close an open position.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if successful
        """
        position = self.get_position(symbol)
        if not position:
            logger.info(f"No position to close for {symbol}")
            return True
        
        try:
            size = abs(position['size'])
            side = 'sell' if position['size'] > 0 else 'buy'
            
            logger.info(f"Closing position: {symbol} size={size}")
            
            order = self.place_market_order(
                symbol=symbol,
                side=side,
                size=size,
                reduce_only=True
            )
            
            return order is not None
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
            return False
