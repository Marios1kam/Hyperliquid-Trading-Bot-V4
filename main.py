"""
Main entry point for the Hyperliquid ML Trading Bot.
Orchestrates all components for 24/7 trading operation.

Usage:
    python main.py --config config/settings.yaml
"""

import argparse
import sys
import signal
import time
import logging
import threading
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables from .env file
load_dotenv('config/.env')

from src.config import config
from src.utils.logging_setup import setup_logging
from src.ml.trainer import ModelTrainer
from src.ml.random_forest_models import RandomForestModelManager
from src.data.binance_collector import BinanceCollector
from src.data.data_manager import DataManager
from src.features.feature_pipeline import FeaturePipeline
from src.regime.regime_classifier import RegimeClassifier
from src.trading_engine import TradingEngine


# Global flag for graceful shutdown
shutdown_requested = False
interrupt_count = 0


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    global shutdown_requested, interrupt_count
    logger = logging.getLogger(__name__)
    
    interrupt_count += 1
    
    if interrupt_count == 1:
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        logger.info("Press Ctrl+C again to force exit immediately")
        shutdown_requested = True
    else:
        logger.warning("Force exit requested!")
        sys.exit(0)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Hyperliquid ML Trading Bot'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/settings.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--train-only',
        action='store_true',
        help='Only train models and exit'
    )
    
    parser.add_argument(
        '--backtest',
        action='store_true',
        help='Run in backtest mode (no live trading)'
    )
    
    return parser.parse_args()


class TradingBot:
    """Main trading bot orchestrator."""
    
    def __init__(self):
        """Initialize trading bot."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.data_collector = BinanceCollector()
        self.data_manager = DataManager()
        self.feature_pipeline = FeaturePipeline()
        self.regime_classifier = RegimeClassifier()
        self.model_manager = RandomForestModelManager()
        self.trainer = ModelTrainer()
        
        # Initialize trading engine
        self.trading_engine = None  # Will be initialized after models are loaded
        
        # Configuration
        self.coin_list = config.coin_list
        self.enable_live_trading = config.enable_live_trading
        self.use_testnet = config.use_testnet
        self.max_positions = config.max_positions
        self.min_confidence = config.min_confidence
        
        # Data update configuration
        self.timeframe_minutes = config.get('TIMEFRAME_MINUTES', 1)  # Candle size
        self.data_update_interval = config.get('DATA_UPDATE_INTERVAL_MINUTES', 15)  # How often to fetch new data
        
        # State
        self.last_retrain_check = datetime.now()
        self.retrain_check_interval = timedelta(hours=6)
        self.last_data_update = datetime.now() - timedelta(minutes=self.data_update_interval)  # Force update on first run

        # Fix 4: Circuit breaker state
        self._trading_day = None
        self._start_of_day_equity = None
        self._daily_halted = False

        # Fix 5: Per-symbol cooldown and daily trade cap state
        self._cooldown_until = {}      # symbol -> datetime
        self._daily_trade_count = {}   # symbol -> int  (reset daily)
        self._known_positions = set()  # symbols with open positions last iteration
        self._state_lock = threading.Lock()  # protects the dicts above

        # Fix 8: Regime instrumentation state
        self._iteration_count = 0
        self._last_regime = {}   # symbol -> last logged regime

        # Depth-gate telemetry (reset every iteration)
        self._depth_gate_stats = {'checked': 0, 'passed': 0, 'blocked': 0}
        
        self.logger.info(f"Initialized trading bot: "
                        f"coins={len(self.coin_list)}, "
                        f"live_trading={self.enable_live_trading}, "
                        f"testnet={self.use_testnet}")
    
    def initialize(self, skip_training=False):
        """Initialize bot - load models or train if needed.
        
        Args:
            skip_training: If True, skip automatic training even if no models found
        """
        self.logger.info("Initializing bot...")
        
        # Try to load existing models
        loaded_count = self.model_manager.load_all_models()
        
        if loaded_count == 0 and not skip_training:
            self.logger.warning("No models found, training initial models...")
            self.train_all_models()
        elif loaded_count == 0:
            self.logger.warning("No models found, but training skipped (will train separately)")
        else:
            self.logger.info(f"Loaded {loaded_count} existing models")
        
        # Initialize trading engine with loaded models
        self.trading_engine = TradingEngine(
            model_manager=self.model_manager,
            enable_trading=self.enable_live_trading
        )
        
        self.logger.info("Bot initialization complete")

        # Instrument the depth gate on every strategy so we can count
        # checked / passed / blocked calls without touching any other file.
        self._patch_depth_gate()
    
    def _patch_depth_gate(self):
        """
        Monkey-patch BaseStrategy._check_depth_gate on every live strategy
        instance so that each call increments self._depth_gate_stats.
        No existing file is modified — we only wrap the bound method.
        """
        bot = self  # capture for closure

        for strategy in self.trading_engine.strategies.values():
            original = strategy._check_depth_gate  # bound method

            def _instrumented(symbol, signal_direction, _orig=original):
                result = _orig(symbol, signal_direction)
                allowed, _ = result
                with bot._state_lock:
                    bot._depth_gate_stats['checked'] += 1
                    if allowed:
                        bot._depth_gate_stats['passed'] += 1
                    else:
                        bot._depth_gate_stats['blocked'] += 1
                return result

            strategy._check_depth_gate = _instrumented

        self.logger.info(
            f"[DepthGate] Instrumented {len(self.trading_engine.strategies)} strategies"
        )

    def train_all_models(self):
        """Train models for all coins."""
        self.logger.info("Training models for all coins...")
        results = self.trainer.train_all_coins(update_data=True)
        
        successful = sum(1 for r in results.values() if 'error' not in r)
        self.logger.info(f"Training complete: {successful}/{len(results)} coins successful")
        
        return results
    
    def check_retraining_needed(self):
        """Check if any models need retraining."""
        now = datetime.now()
        
        if now - self.last_retrain_check < self.retrain_check_interval:
            return
        
        self.logger.info("Checking if retraining is needed...")
        self.last_retrain_check = now
        
        for coin in self.coin_list:
            try:
                self.trainer.retrain_if_needed(coin, force=False)
            except Exception as e:
                self.logger.error(f"Error checking retrain for {coin}: {e}")

    # ── Fix 4 + 5: Risk / Cooldown helpers ────────────────────────────────

    def _get_current_equity(self) -> float:
        """Fetch live account equity from Hyperliquid."""
        account_info = self.trading_engine.client.get_account_info()
        if not account_info:
            raise RuntimeError("get_account_info() returned None")
        return float(account_info['account_value'])

    def _check_daily_circuit_breaker(self) -> bool:
        """
        Returns True if trading is allowed, False if daily loss limit is hit.
        Called at the top of every main loop iteration.
        """
        if not self.trading_engine:
            return True  # Not yet initialized — allow through

        today = datetime.utcnow().date()

        # Reset at start of each new trading day
        if self._trading_day != today:
            # Fix 8: Log yesterday's summary before resetting
            if self._trading_day is not None:
                self.logger.warning(
                    f"[DAY SUMMARY {self._trading_day}] "
                    f"Total iterations: {self._iteration_count} | "
                    f"Last regime distribution: {dict(self._last_regime)} | "
                    f"Daily trades: {dict(self._daily_trade_count)}"
                )
            self._trading_day = today
            self._daily_halted = False
            with self._state_lock:
                self._daily_trade_count = {}
                self._known_positions = set()
            try:
                self._start_of_day_equity = self._get_current_equity()
                self.logger.info(
                    f"[CIRCUIT BREAKER] New day {today}. "
                    f"Starting equity: ${self._start_of_day_equity:.2f}"
                )
            except Exception as e:
                self.logger.error(f"[CIRCUIT BREAKER] Failed to get start equity: {e}")
                self._start_of_day_equity = None

        # If already halted today, keep halted
        if self._daily_halted:
            self.logger.debug("[CIRCUIT BREAKER] Trading halted for today.")
            return False

        # Can't check without a valid baseline
        if not self._start_of_day_equity or self._start_of_day_equity <= 0:
            return True

        # Fetch current equity and compare
        try:
            current_equity = self._get_current_equity()
        except Exception as e:
            self.logger.warning(
                f"[CIRCUIT BREAKER] Could not fetch equity: {e}. Allowing trade."
            )
            return True

        daily_pnl_pct = (
            (current_equity - self._start_of_day_equity) / self._start_of_day_equity
        )
        limit_pct = -abs(config.get('DAILY_LOSS_LIMIT_PERCENT', 3.0)) / 100.0

        if daily_pnl_pct <= limit_pct:
            self._daily_halted = True
            self.logger.warning(
                f"[CIRCUIT BREAKER TRIGGERED] Daily loss "
                f"{daily_pnl_pct * 100:.2f}% hit limit {limit_pct * 100:.2f}%. "
                f"Halting all trading for today. "
                f"Start equity: ${self._start_of_day_equity:.2f} | "
                f"Current equity: ${current_equity:.2f}"
            )
            return False

        return True

    def _set_cooldown(self, symbol: str, minutes: int = 15):
        """Set a post-close cooldown for a symbol."""
        until = datetime.utcnow() + timedelta(minutes=minutes)
        with self._state_lock:
            self._cooldown_until[symbol] = until
        self.logger.info(
            f"[COOLDOWN] {symbol} cooling down for {minutes}min "
            f"until {until.strftime('%H:%M:%S')} UTC"
        )

    def _is_on_cooldown(self, symbol: str) -> bool:
        """Return True if symbol is still in its post-close cooldown window."""
        with self._state_lock:
            if symbol not in self._cooldown_until:
                return False
            if datetime.utcnow() < self._cooldown_until[symbol]:
                return True
            del self._cooldown_until[symbol]
            return False

    # ──────────────────────────────────────────────────────────────────────

    def _reconcile_positions(self):
        """Compare in-memory vs live exchange position state. Log and correct desyncs."""
        try:
            raw_positions = self.trading_engine.client.get_positions()
            exchange_positions = {p['symbol']: p for p in raw_positions if p.get('symbol')}
            memory_positions = dict(self.trading_engine.position_tracker.positions)
            all_symbols = set(memory_positions) | set(exchange_positions)
            for symbol in all_symbols:
                in_memory = symbol in memory_positions
                on_exchange = symbol in exchange_positions
                if in_memory != on_exchange:
                    self.logger.warning(
                        f"[POSITION DESYNC] {symbol}: "
                        f"memory={'OPEN' if in_memory else 'FLAT'} "
                        f"exchange={'OPEN' if on_exchange else 'FLAT'} "
                        f"\u2014 syncing to exchange state"
                    )
                    if on_exchange:
                        self.trading_engine.position_tracker.update_positions()
                    else:
                        self.trading_engine.position_tracker.positions.pop(symbol, None)
        except Exception as e:
            self.logger.error(f"[RECONCILIATION] Failed: {e}")

    def process_coin(self, coin: str) -> dict:
        """
        Process a single coin: get data, features, regime, and signal.
        
        Args:
            coin: Trading symbol
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Check if symbol is available on Hyperliquid before processing
            if self.trading_engine and not self.trading_engine.client.is_symbol_available(coin):
                self.logger.info(f"⏭️  Skipping {coin} - not available on Hyperliquid testnet")
                return {'error': 'symbol_not_available', 'skipped': True}

            # ── Fix 5: Detect position closes → trigger cooldown ──────────
            if self.trading_engine:
                with self._state_lock:
                    had_position = coin in self._known_positions
                current_pos = self.trading_engine.position_tracker.get_position(coin)
                has_position = current_pos is not None

                if had_position and not has_position:
                    # Position disappeared since last iteration (TP/SL/liquidation)
                    self.logger.info(
                        f"[COOLDOWN] {coin} position closed — starting 15-min cooldown"
                    )
                    self._set_cooldown(coin, minutes=15)

                with self._state_lock:
                    if has_position:
                        self._known_positions.add(coin)
                    else:
                        self._known_positions.discard(coin)

            # Cooldown gate
            if self._is_on_cooldown(coin):
                self.logger.debug(f"[COOLDOWN] Skipping {coin} — in cooldown")
                return {'skipped': True, 'reason': 'cooldown'}

            # Daily trade cap gate
            max_trades = config.get('MAX_DAILY_TRADES_PER_SYMBOL', 6)
            with self._state_lock:
                today_count = self._daily_trade_count.get(coin, 0)
            if today_count >= max_trades:
                self.logger.info(
                    f"[TRADE CAP] Skipping {coin} — daily limit reached "
                    f"({today_count}/{max_trades})"
                )
                return {'skipped': True, 'reason': 'daily_cap'}
            # ──────────────────────────────────────────────────────────────

            # Load latest data (already updated in batch)
            df = self.data_manager.load_data(coin)
            
            if df.empty or len(df) < 100:
                self.logger.warning(f"Insufficient data for {coin}: {len(df)} rows")
                return {'error': 'insufficient_data'}
            
            # OPTIMIZATION: Only process recent data for prediction (last 10k rows)
            # This dramatically speeds up feature computation and regime detection
            # Full data is still saved, but we only analyze recent history for trading
            lookback_for_prediction = 10000
            if len(df) > lookback_for_prediction:
                df_recent = df.tail(lookback_for_prediction).copy()
                self.logger.debug(f"Processing last {lookback_for_prediction} rows for {coin} (total: {len(df)})")
            else:
                df_recent = df.copy()
            
            # Compute features on recent data only
            df_recent = self.feature_pipeline.compute_features(df_recent)
            
            # Detect regime on recent data only
            df_recent = self.regime_classifier.classify_regimes(df_recent)
            
            # Get current regime and confidence
            current_regime, regime_confidence = self.regime_classifier.get_current_regime(df_recent)

            # Fix 8: Log regime changes (WARNING level so it bypasses module-level filter)
            prev_regime = self._last_regime.get(coin)
            if current_regime != prev_regime:
                self.logger.warning(
                    f"[REGIME CHANGE] {coin}: {prev_regime or 'INIT'} \u2192 {current_regime} "
                    f"(confidence: {regime_confidence:.3f})"
                )
                self._last_regime[coin] = current_regime

            # Prepare features for prediction (last row)
            feature_cols = [col for col in df_recent.columns 
                           if col not in ['target', 'regime', 'open', 'high', 'low', 'close', 'volume',
                                         'trend_regime', 'sideways_regime']]
            
            # Add regime as feature
            df_with_regime = df_recent.copy()
            regime_dummies = pd.get_dummies(df_with_regime['regime'], prefix='regime')
            X = pd.concat([df_with_regime[feature_cols], regime_dummies], axis=1)
            
            # Get latest features
            X_latest = X.tail(1)
            
            # Get prediction from model
            signal, confidence = self.model_manager.predict_for_coin(
                coin, current_regime, X_latest
            )

            # Fix 8: ML signal debug log
            self.logger.debug(
                f"[ML SIGNAL] {coin}: prediction={signal} "
                f"confidence={confidence:.3f} regime={current_regime}"
            )

            # Get current price from Binance data (for reference)
            binance_price = df_recent['close'].iloc[-1]
            
            # Get LIVE price from Hyperliquid for trading decisions
            # This ensures we use the most current price when placing orders
            if self.trading_engine and self.trading_engine.client:
                live_price = self.trading_engine.client.get_market_price(coin)
                current_price = live_price if live_price else binance_price
                if live_price:
                    self.logger.debug(f"{coin}: Binance=${binance_price:.2f}, Hyperliquid=${live_price:.2f}")
            else:
                current_price = binance_price
            
            # Prepare features dictionary for trading engine
            features_dict = df_recent.iloc[-1].to_dict()
            
            # Process trading opportunity through trading engine
            execution_result = None
            if self.trading_engine:
                execution_result = self.trading_engine.process_trading_opportunity(
                    symbol=coin,
                    ml_signal=signal,
                    ml_confidence=confidence,
                    regime=current_regime,
                    features=features_dict,
                    current_price=current_price
                )

                # Fix 5: Track new fills for daily cap + known-positions
                if execution_result and execution_result.get('status') == 'filled':
                    with self._state_lock:
                        self._daily_trade_count[coin] = (
                            self._daily_trade_count.get(coin, 0) + 1
                        )
                        self._known_positions.add(coin)
            
            result = {
                'coin': coin,
                'regime': current_regime,
                'regime_confidence': regime_confidence,
                'signal': signal,
                'signal_confidence': confidence,
                'price': current_price,
                'timestamp': df_recent.index[-1],
                'execution': execution_result
            }
            
            self.logger.debug(f"{coin}: regime={current_regime}, signal={signal}, "
                           f"confidence={confidence:.3f}, price={result['price']:.2f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing {coin}: {e}", exc_info=True)
            return {'error': str(e)}
    
    def trading_loop(self):
        """Main trading loop."""
        self.logger.info("Starting trading loop...")
        
        # Display startup banner
        print("\n" + "="*60)
        print("  🤖 HYPERLIQUID TRADING BOT - STARTED")
        print("="*60)
        print(f"  Mode: {'🔴 LIVE TRADING' if self.enable_live_trading else '📝 PAPER TRADING'}")
        print(f"  Coins: {len(self.coin_list)}")
        print(f"  Min Confidence: {self.min_confidence*100:.0f}%")
        print(f"  Data Update: Every {self.data_update_interval} minutes")
        print(f"  Environment: {'🧪 TESTNET' if self.use_testnet else '⚠️  MAINNET'}")
        print("="*60 + "\n")
        
        iteration = 0
        
        while not shutdown_requested:
            iteration += 1
            print(f"\n⏰ Iteration {iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"=== Trading iteration {iteration} ===")

            # Fix 4: Circuit breaker — halt loop if daily loss limit is exceeded
            if not self._check_daily_circuit_breaker():
                sleep_seconds = config.get('TIMEFRAME_MINUTES', 1) * 60
                self.logger.info(f"[CIRCUIT BREAKER] Sleeping {sleep_seconds}s before next check...")
                for _ in range(sleep_seconds):
                    if shutdown_requested:
                        break
                    time.sleep(1)
                continue
            
            # Fix 8: Per-iteration regime + signal counters (aggregated after coin loop)
            regime_counts = {}
            signal_counts = {'long': 0, 'short': 0, 'neutral': 0, 'skipped': 0}

            # Reset depth-gate telemetry for this iteration
            with self._state_lock:
                self._depth_gate_stats = {'checked': 0, 'passed': 0, 'blocked': 0}

            # Display account summary
            try:
                if self.trading_engine and self.trading_engine.client.can_trade():
                    account_info = self.trading_engine.client.get_account_info()
                    positions = self.trading_engine.client.get_positions()
                    
                    if account_info:
                        account_value = account_info.get('account_value', 0)
                        total_pnl = sum(pos.get('unrealized_pnl', 0) for pos in positions)
                        
                        print(f"💰 Account Value: ${account_value:,.2f}")
                        print(f"📊 Open Positions: {len(positions)}")
                        if positions:
                            for pos in positions:
                                symbol = pos.get('symbol')
                                size = pos.get('size', 0)
                                pnl = pos.get('unrealized_pnl', 0)
                                pnl_sign = "🟢" if pnl >= 0 else "🔴"
                                side = "LONG" if size > 0 else "SHORT"
                                print(f"   {pnl_sign} {symbol}: {side} {abs(size)} | PnL: ${pnl:+.2f}")
                        print(f"💵 Total PnL: ${total_pnl:+.2f}")
                        print("-" * 60)
            except Exception as e:
                self.logger.error(f"Error displaying account summary: {e}")
            
            try:
                # Check if retraining is needed
                self.check_retraining_needed()
                
                # Check if we need to update data
                now = datetime.now()
                time_since_last_update = (now - self.last_data_update).total_seconds() / 60
                should_update_data = time_since_last_update >= self.data_update_interval
                
                if should_update_data:
                    # OPTIMIZATION: Fetch all coin data in parallel
                    self.logger.info(f"Fetching latest data for {len(self.coin_list)} coins in parallel (last update: {time_since_last_update:.1f} min ago)...")
                    
                    def update_coin_data(coin):
                        """Helper function to update data for a single coin."""
                        try:
                            df = self.data_manager.load_data(coin)
                            if df.empty:
                                self.logger.warning(f"No data for {coin}, fetching...")
                                df = self.data_collector.fetch_ohlcv(coin)
                            else:
                                # Update with latest candles
                                df = self.data_collector.update_data(coin, df)
                            
                            # Save updated data
                            self.data_manager.save_data(coin, df)
                            return (coin, True)
                        except Exception as e:
                            self.logger.error(f"Error updating data for {coin}: {e}")
                            return (coin, False)
                    
                    # Use ThreadPoolExecutor for parallel data fetching
                    with ThreadPoolExecutor(max_workers=10) as executor:
                        futures = {executor.submit(update_coin_data, coin): coin for coin in self.coin_list}
                        for future in as_completed(futures):
                            coin, success = future.result()
                            if success:
                                self.logger.debug(f"Updated data for {coin}")
                    
                    self.last_data_update = now
                    self.logger.info("Data fetch complete, processing signals...")
                else:
                    self.logger.info(f"Using cached data (next update in {self.data_update_interval - time_since_last_update:.1f} min)")
                
                # Process each coin in parallel (using cached or fresh data)
                signals = {}
                with ThreadPoolExecutor(max_workers=10) as executor:
                    future_to_coin = {executor.submit(self.process_coin, coin): coin for coin in self.coin_list}
                    for future in as_completed(future_to_coin):
                        coin = future_to_coin[future]
                        try:
                            result = future.result()
                            # Exclude error results AND skipped results (no signal_confidence)
                            if 'error' not in result and not result.get('skipped'):
                                signals[coin] = result
                        except Exception as e:
                            self.logger.error(f"Error processing {coin}: {e}")
                
                # Fix 8: Aggregate regime + signal counts from coin results
                for _coin, _sig in signals.items():
                    if _sig.get('skipped'):
                        signal_counts['skipped'] += 1
                        continue
                    _r = _sig.get('regime', 'UNKNOWN')
                    regime_counts[_r] = regime_counts.get(_r, 0) + 1
                    _s = _sig.get('signal', 0)
                    if _s == 1:
                        signal_counts['long'] += 1
                    elif _s == -1:
                        signal_counts['short'] += 1
                    else:
                        signal_counts['neutral'] += 1

                # Filter signals by confidence
                valid_signals = {
                    coin: sig for coin, sig in signals.items()
                    if sig.get('signal_confidence', 0) >= self.min_confidence and sig.get('signal', 0) != 0
                }
                
                filtered_signals = {
                    coin: sig for coin, sig in signals.items()
                    if sig.get('signal_confidence', 0) < self.min_confidence and sig.get('signal', 0) != 0
                }
                
                # Fix 8: Full iteration summary (WARNING so it appears despite module filter)
                self._iteration_count += 1
                with self._state_lock:
                    _cooldowns_active = sum(
                        1 for t in self._cooldown_until.values()
                        if datetime.utcnow() < t
                    )
                    _daily_trades_copy = dict(self._daily_trade_count)
                with self._state_lock:
                    _dg = dict(self._depth_gate_stats)
                self.logger.warning(
                    f"[ITER {self._iteration_count:04d}] "
                    f"Regimes: {regime_counts} | "
                    f"Signals: {signal_counts} | "
                    f"DepthGate: checked={_dg['checked']} passed={_dg['passed']} blocked={_dg['blocked']} | "
                    f"Cooldowns: {_cooldowns_active} | "
                    f"Daily trades: {_daily_trades_copy} | "
                    f"Halted: {self._daily_halted}"
                )

                # Every 10 iterations: reconcile in-memory vs exchange positions
                if self._iteration_count % 10 == 0:
                    self._reconcile_positions()

                # Show message if no signals at all
                if not valid_signals and not filtered_signals:
                    print(f"\u23f8\ufe0f  No signals generated this iteration")

                print("-" * 80 + "\n")
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}", exc_info=True)
            
            # Sleep before next iteration (1 minute for 1m timeframe)
            sleep_seconds = config.get('TIMEFRAME_MINUTES', 1) * 60
            self.logger.info(f"Sleeping for {sleep_seconds} seconds...")
            
            for _ in range(sleep_seconds):
                if shutdown_requested:
                    break
                time.sleep(1)
        
        self.logger.info("Trading loop stopped")
    
    def shutdown(self):
        """Graceful shutdown."""
        self.logger.info("Shutting down bot...")
        
        # Close positions if configured
        if config.get('CLOSE_POSITIONS_ON_SHUTDOWN', True):
            self.logger.info("Closing all positions...")
            if self.trading_engine:
                try:
                    self.trading_engine.position_tracker.close_all_positions()
                    self.logger.info("All positions closed")
                except Exception as e:
                    self.logger.error(f"Error closing positions: {e}")
        
        self.logger.info("Shutdown complete")


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config.load_config(args.config)
    
    # Setup logging
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("Hyperliquid ML Trading Bot Starting")
    logger.info("=" * 80)
    logger.info(f"Configuration loaded from: {args.config}")
    logger.info(f"Live trading: {config.enable_live_trading}")
    logger.info(f"Testnet mode: {config.use_testnet}")
    logger.info(f"Coins: {len(config.coin_list)}")
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize bot
        bot = TradingBot()
        
        # Train-only mode
        if args.train_only:
            logger.info("Train-only mode: training models and exiting")
            bot.initialize(skip_training=True)  # Skip auto-training in initialize
            bot.train_all_models()  # Train once here
            logger.info("Training complete, exiting")
            return 0
        
        # Normal mode: initialize with auto-training if needed
        bot.initialize()
        
        # Start trading loop
        bot.trading_loop()
        
        # Graceful shutdown
        bot.shutdown()
        
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
