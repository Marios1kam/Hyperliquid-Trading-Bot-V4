"""
Trading engine.
Integrates all components for complete trading system.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
from .database.trade_tracker import TradeTracker
import pandas as pd

from src.config import config
from src.exchange.hyperliquid_client import HyperliquidClient
from src.exchange.order_manager import OrderManager, OrderSide
from src.exchange.position_tracker import PositionTracker
from src.risk.position_sizer import PositionSizer
from src.risk.sl_tp_manager import SLTPManager
from src.risk.portfolio_risk import PortfolioRisk
from src.strategies.trend_strategy import TrendStrategy, MomentumStrategy
from src.strategies.sideways_strategy import SidewaysStrategy, BreakoutStrategy, ChoppyStrategy
from src.ml.random_forest_models import RandomForestModelManager


logger = logging.getLogger(__name__)


class TradingEngine:
    """
    Complete trading engine integrating all components.
    
    Workflow:
    1. Get ML signals from models
    2. Generate strategy signals
    3. Check risk limits
    4. Calculate position size
    5. Execute trades
    6. Manage SL/TP
    7. Monitor positions
    """
    
    def __init__(
        self,
        model_manager: RandomForestModelManager,
        enable_trading: bool = False
    ):
        """
        Initialize trading engine.
        
        Args:
            model_manager: ML model manager
            enable_trading: Enable live trading
        """
        self.model_manager = model_manager
        self.enable_trading = enable_trading
        
        # Initialize components
        self.client = HyperliquidClient()
        self.order_manager = OrderManager(self.client)
        self.position_tracker = PositionTracker(self.client)
        self.position_sizer = PositionSizer()
        self.sl_tp_manager = SLTPManager(self.order_manager)
        self.portfolio_risk = PortfolioRisk()
        self.trade_tracker = TradeTracker()
        
        # Initialize strategies
        self.strategies = {
            'trend': TrendStrategy(),
            'momentum': MomentumStrategy(),
            'sideways': SidewaysStrategy(),
            'breakout': BreakoutStrategy(),
            'choppy': ChoppyStrategy(),
        }

        # Attach the live client to every strategy so the market-depth gate
        # can call get_order_book(). This is purely additive — the gate is a
        # no-op when MARKET_DEPTH.ENABLED is false or no client is available.
        for strategy in self.strategies.values():
            strategy.client = self.client

        # State
        self.last_update = None
        
        logger.info(f"Initialized trading engine: trading_enabled={enable_trading}")


    def _get_strategy_name(self, regime: str) -> str:
        """Get strategy name for logging."""
        if regime in ['BULL', 'BEAR']:
            return 'Trend Following'
        elif regime == 'CHOPPY':
            return 'Conservative'
        elif regime in ['SIDEWAYS_QUIET', 'SIDEWAYS_VOLATILE']:
            return 'Mean Reversion'
        return 'N/A'
    
    
    def process_trading_opportunity(
        self,
        symbol: str,
        ml_signal: int,
        ml_confidence: float,
        regime: str,
        features: Dict,
        current_price: float
    ) -> Optional[Dict]:
        """
        Process a trading opportunity.
        
        Args:
            symbol: Trading symbol
            ml_signal: ML model signal (1, 0, -1)
            ml_confidence: ML model confidence
            regime: Current market regime
            features: Dictionary of features
            current_price: Current market price
            
        Returns:
            Execution result dictionary or None
        """
        # Determine signal direction for display
        signal_dir = "LONG" if ml_signal == 1 else "SHORT" if ml_signal == -1 else "NEUTRAL"
        strategy_name = self._get_strategy_name(regime)
        
        logger.info(f"Processing opportunity: {symbol} signal={ml_signal} "
                   f"confidence={ml_confidence:.3f} regime={regime}")
        
        # Update positions
        self.position_tracker.update_positions()
        
        # Check if we already have a position
        existing_position = self.position_tracker.get_position(symbol)
        if existing_position:
            logger.info(f"Existing position for {symbol}: {existing_position.size}")
            print(f"⏭️  {symbol:6} | {signal_dir:5} {ml_confidence:.0%} | {regime:6} | {strategy_name} | → Has position ({existing_position.size})")
            return self._manage_existing_position(symbol, existing_position, regime, features)
        # No signal or neutral signal
        if ml_signal == 0:
            logger.debug(f"Neutral signal for {symbol}, no action")
            print(f"⏭️  {symbol:6} | NEUTRAL {ml_confidence:.0%} | {regime:6} | {strategy_name} | → No action")
            return None
        # Generate strategy signal
        strategy_signal = self._generate_strategy_signal(
            symbol, ml_signal, ml_confidence, regime, features, current_price
        )

        # Fix 8: Log when strategy filters out a non-zero ML signal
        if not strategy_signal:
            if ml_signal != 0:
                logger.info(
                    f"[SIGNAL FILTERED] {symbol}: ml={ml_signal} \u2192 strategy=None "
                    f"(regime={regime}, confidence={ml_confidence:.3f})"
                )
            else:
                logger.debug(f"No strategy signal generated for {symbol}")
            return None

        # Check risk limits
        can_open, risk_reason = self._check_risk_limits()
        if not can_open:
            logger.warning(f"Risk limits prevent opening position for {symbol}: {risk_reason}")
            print(f"❌ {symbol:6} | {signal_dir:5} {ml_confidence:.0%} | {regime:6} | {strategy_name} | → Risk limits exceeded: {risk_reason}")
            return None
        # Execute trade
        if self.enable_trading:
            result = self._execute_entry(symbol, strategy_signal, regime, features)
            # Print execution result
            if result and 'error' in result:
                print(f"⚠️  {symbol:6} | {signal_dir:5} {ml_confidence:.0%} | {regime:6} | {strategy_name} | → FAILED: {result['error']}")
            elif result and result.get('status') == 'filled':
                filled_size = result.get('filled_size', 0)
                filled_price = result.get('filled_price', 0)
                tp_price = result.get('tpsl', {}).get('tp_price', 0)
                sl_price = result.get('tpsl', {}).get('sl_price', 0)
                print(f"✅ {symbol:6} | {signal_dir:5} {ml_confidence:.0%} | {regime:6} | {strategy_name} | → FILLED {filled_size} @ ${filled_price:.4f} (TP: ${tp_price:.2f}, SL: ${sl_price:.2f})")
            else:
                print(f"⏳ {symbol:6} | {signal_dir:5} {ml_confidence:.0%} | {regime:6} | {strategy_name} | → Order pending...")
            return result
        else:
            logger.info(f"Paper trade: Would {strategy_signal.action} {symbol} "
                       f"@ {current_price:.2f}")
            return {
                'symbol': symbol,
                'action': 'paper_trade',
                'signal': strategy_signal.to_dict()
            }
    
    def _generate_strategy_signal(
        self,
        symbol: str,
        ml_signal: int,
        ml_confidence: float,
        regime: str,
        features: Dict,
        current_price: float
    ):
        """Generate signal from appropriate strategy."""
        # Select strategy based on regime
        if regime in ['BULL', 'BEAR']:
            # Try trend strategy first
            signal = self.strategies['trend'].generate_signal(
                symbol, ml_signal, ml_confidence, regime, features, current_price
            )
            
            # Try momentum if trend fails
            if not signal:
                signal = self.strategies['momentum'].generate_signal(
                    symbol, ml_signal, ml_confidence, regime, features, current_price
                )
            
            return signal
        
        elif regime in ['SIDEWAYS_QUIET', 'SIDEWAYS_VOLATILE']:
            # Try mean-reversion first
            signal = self.strategies['sideways'].generate_signal(
                symbol, ml_signal, ml_confidence, regime, features, current_price
            )
            
            # Try breakout if mean-reversion fails
            if not signal:
                signal = self.strategies['breakout'].generate_signal(
                    symbol, ml_signal, ml_confidence, regime, features, current_price
                )
            
            return signal
        
        elif regime == 'CHOPPY':
            return self.strategies['choppy'].generate_signal(
                symbol, ml_signal, ml_confidence, regime, features, current_price
            )
        
        return None
    
    def _check_risk_limits(self) -> tuple:
        """Check if we can open a new position. Returns (can_open, reason)."""
        # Get account info
        account_info = self.client.get_account_info()
        if not account_info:
            logger.error("Could not get account info")
            return False, "Could not get account info"
        
        # Calculate risk metrics
        positions = self.position_tracker.get_all_positions()
        total_pnl = sum(p.unrealized_pnl for p in positions)
        margin_used = sum(p.margin_used for p in positions)
        at_risk = self.position_tracker.check_liquidation_risk()
        
        metrics = self.portfolio_risk.calculate_metrics(
            account_equity=account_info['equity'],
            margin_used=margin_used,
            total_pnl=total_pnl,
            position_count=len(positions),
            at_risk_positions=at_risk
        )
        
        # Check if we can open position
        can_open, reason = self.portfolio_risk.can_open_position(metrics)
        
        if not can_open:
            logger.warning(f"Cannot open position: {reason}")
        
        return can_open, reason
    
    def _execute_entry(self, symbol: str, signal, regime: str, features: Dict) -> Dict:
        """Execute entry trade."""
        logger.info(f"Executing entry: {signal.action} {symbol}")
        
        # Get account info
        account_info = self.client.get_account_info()
        if not account_info:
            return {'error': 'Could not get account info'}
            
        # Calculate position size
        volatility = features.get('atr_pct')
        position_size_usd = self.position_sizer.calculate_position_size(
            account_equity=account_info['equity'],
            regime=regime,
            confidence=signal.confidence,
            volatility=volatility
        )

        
        # Convert to quantity
        quantity = self.position_sizer.calculate_quantity(
            position_size_usd=position_size_usd,
            price=signal.entry_price
        )

        
        if quantity == 0:
            return {'error': 'Position size too small'}
        
        # 1. Set Leverage - COMMENTED OUT TO USE HYPERLIQUID DEFAULT
        # leverage = 10  # Default fallback
        # try:
        #     # Check for dynamic leverage settings
        #     dynamic_leverage = getattr(config, 'DYNAMIC_LEVERAGE', False)
        #     leverage_map = getattr(config, 'LEVERAGE_BY_REGIME', {})
        #     
        #     if dynamic_leverage and regime in leverage_map:
        #         leverage = int(leverage_map[regime])
        #         # print(f"DEBUG: Using dynamic leverage {leverage}x for {regime} regime")
        #     else:
        #         leverage = int(getattr(config, 'LEVERAGE', 10))
        #         
        #     self.client.update_leverage(symbol, leverage)
        # except Exception as e:
        #     logger.warning(f"Failed to set leverage for {symbol}: {e}")
        
        # Using Hyperliquid's default leverage
        leverage = None  # Will be set by Hyperliquid
        
        # 2. Place market order (Entry)
        side = 'buy' if signal.action == 'buy' else 'sell'
        
        # CRITICAL: Quantize size and round up to minimum if needed
        # This ensures all valid signals can execute even with small account sizes
        quantized_size = self.client.quantize_size(symbol, quantity, round_up_to_min=True)
        if quantized_size == 0:
            return {'error': f'Invalid position size for {symbol}'}
        
        # CRITICAL: Check minimum order value ($10 on Hyperliquid)
        order_value = quantized_size * signal.entry_price
        min_order_value = 10.0
        
        if order_value < min_order_value:
            # Calculate size needed for $10 minimum with 2% buffer to account for quantization rounding
            min_size_for_value = (min_order_value * 1.02) / signal.entry_price
            # Quantize it
            quantized_size = self.client.quantize_size(symbol, min_size_for_value, round_up_to_min=True)
            new_order_value = quantized_size * signal.entry_price
            
            # Iteratively verify and adjust until we're above minimum
            max_iterations = 5
            iteration = 0
            while new_order_value < min_order_value and iteration < max_iterations:
                # Add one more step size
                size_info = self.client.size_precision.get(symbol, self.client.size_precision['DEFAULT'])
                quantized_size += size_info['step']
                quantized_size = round(quantized_size, size_info['decimals'])
                new_order_value = quantized_size * signal.entry_price
                iteration += 1
            
            # Final safety check
            if new_order_value < min_order_value:
                logger.warning(f"Unable to meet minimum order value for {symbol}: ${new_order_value:.2f} < ${min_order_value}")
                return {'error': f'Cannot meet minimum order value of ${min_order_value}'}

        
        order_result = self.client.place_market_order(
            symbol=symbol,
            side=side,
            size=quantized_size  # Use pre-quantized size
        )
        
        if not order_result or 'error' in order_result:
            error_msg = order_result.get('error', 'Failed to place order') if order_result else 'Failed to place order'
            logger.warning(f"❌ Order for {symbol} not executed: {error_msg}")
            print(f"❌ Order rejected: {error_msg}")
            return order_result or {'error': 'Failed to place order'}
            
        # 3. Place TP/SL if successful and filled
        if order_result.get('status') == 'filled':
            try:
                filled_price = order_result.get('filled_price', signal.entry_price)
                filled_size = order_result.get('filled_size', quantity)
                
                # Calculate percentages from signal prices if available
                if signal.stop_loss:
                    sl_pct = abs(filled_price - signal.stop_loss) / filled_price
                else:
                    sl_pct = 0.02 # Default 2%
                    
                if signal.take_profit:
                    tp_pct = abs(signal.take_profit - filled_price) / filled_price
                else:
                    tp_pct = 0.04 # Default 4%
                
                # Ensure minimums
                sl_pct = max(sl_pct, 0.005)
                tp_pct = max(tp_pct, 0.005)
                
                print(f"Placing TP/SL (SL: {sl_pct:.2%}, TP: {tp_pct:.2%})")
                
                tpsl_result = self.client.place_tpsl_orders(
                    symbol=symbol,
                    entry_price=filled_price,
                    size=filled_size,
                    is_long=(side == 'buy'),
                    stop_loss_pct=sl_pct,
                    take_profit_pct=tp_pct
                )
                
                order_result['tpsl'] = tpsl_result
                
                # Calculate actual TP/SL prices for logging
                if side == 'buy':
                    sl_price = filled_price * (1 - sl_pct)
                    tp_price = filled_price * (1 + tp_pct)
                else:
                    sl_price = filled_price * (1 + sl_pct)
                    tp_price = filled_price * (1 - tp_pct)
                
                # Log trade to database
                strategy_name = self._get_strategy_name(regime)
                self.trade_tracker.log_trade({
                    'symbol': symbol,
                    'side': side,
                    'entry_price': filled_price,
                    'quantity': filled_size,
                    'leverage': 'default',  # Using Hyperliquid's default leverage
                    'regime': regime,
                    'confidence': signal.confidence,
                    'strategy_name': strategy_name,
                    'stop_loss_price': sl_price,
                    'take_profit_price': tp_price,
                    'order_id': order_result.get('order_id', '')
                })
                
            except Exception as e:
                logger.error(f"Failed to place TP/SL or log trade for {symbol}: {e}")
                order_result['tpsl_error'] = str(e)
        
        return order_result

    def _place_tp_sl_orders(self, symbol, entry_side, quantity, signal):
        """Place Take Profit and Stop Loss orders.

        DEPRECATED: Superseded by self.client.place_tpsl_orders() called
        inside _execute_entry(). This method is unreachable (zero call sites)
        and will be removed in a future cleanup. Do not call.
        """
        # DEPRECATED: This method has no call sites in the codebase and
        # contains undefined variable references (order, features, regime).
        # TP/SL is handled by self.client.place_tpsl_orders() in _execute_entry().
        raise NotImplementedError(
            "_place_tp_sl_orders is deprecated. "
            "Use self.client.place_tpsl_orders() via _execute_entry() instead."
        )
    
    def _manage_existing_position(self, symbol: str, position, regime: str, features: Dict) -> Optional[Dict]:
        """Manage existing position."""
        # Fix 3: Minimum hold time check
        min_hold_minutes = config.min_hold_minutes
        hold_time_delta = datetime.now() - position.opened_at
        hold_minutes = hold_time_delta.total_seconds() / 60.0

        if hold_minutes < min_hold_minutes:
            logger.debug(
                f"{symbol}: skip exit \u2014 held only {hold_minutes:.1f}min / "
                f"{min_hold_minutes}min"
            )
            return None

        # Check if we should exit
        is_long = position.is_long
        
        # Get SL/TP levels
        atr = features.get('atr')
        sl_tp_levels = self.sl_tp_manager.calculate_levels(
            entry_price=position.entry_price,
            is_long=is_long,
            regime=regime,
            atr=atr
        )
        
        # Check for manual exit
        exit_reason = self.sl_tp_manager.check_manual_exit(
            symbol=symbol,
            current_price=position.current_price,
            entry_price=position.entry_price,
            is_long=is_long,
            levels=sl_tp_levels
        )
        
        if exit_reason and self.enable_trading:
            logger.info(f"Exiting position {symbol}: {exit_reason}")
            self.position_tracker.close_position(symbol)
            self.sl_tp_manager.cancel_sl_tp(symbol)
            
            return {
                'symbol': symbol,
                'action': 'exit',
                'reason': exit_reason,
                'pnl': position.unrealized_pnl
            }
        
        return None
    
    def get_status(self) -> Dict:
        """Get trading engine status."""
        # Update positions
        self.position_tracker.update_positions()
        
        # Get account info
        account_info = self.client.get_account_info()
        
        # Get positions
        positions = self.position_tracker.get_all_positions()
        position_summary = self.position_tracker.get_position_summary()
        
        # Get risk metrics
        if account_info:
            total_pnl = sum(p.unrealized_pnl for p in positions)
            margin_used = sum(p.margin_used for p in positions)
            at_risk = self.position_tracker.check_liquidation_risk()
            
            risk_metrics = self.portfolio_risk.calculate_metrics(
                account_equity=account_info['equity'],
                margin_used=margin_used,
                total_pnl=total_pnl,
                position_count=len(positions),
                at_risk_positions=at_risk
            )
            
            risk_summary = self.portfolio_risk.get_risk_summary(risk_metrics)
        else:
            risk_summary = None
        
        return {
            'trading_enabled': self.enable_trading,
            'connected': self.client.is_connected(),
            'account': account_info,
            'positions': position_summary,
            'risk': risk_summary,
            'orders': self.order_manager.get_summary(),
            'sl_tp': self.sl_tp_manager.get_summary(),
            'last_update': datetime.now().isoformat(),
        }
