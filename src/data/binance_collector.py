"""
Binance data collector using CCXT.
Fetches OHLCV data for configured coin universe.
"""

import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import List, Optional
import logging

from src.config import config


logger = logging.getLogger(__name__)


class BinanceCollector:
    """Collect OHLCV data from Binance using CCXT."""
    
    def __init__(self):
        """Initialize Binance CCXT exchange."""
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
            }
        })
        
        # Get configuration
        self.timeframe = self._get_timeframe()
        self.lookback_periods = config.get('LOOKBACK_PERIODS', 200)
        
        logger.info(f"Initialized Binance collector with timeframe: {self.timeframe}")
    
    def _get_timeframe(self) -> str:
        """Convert timeframe minutes to CCXT format."""
        minutes = config.get('TIMEFRAME_MINUTES', 1)
        timeframe_map = {
            1: '1m',
            3: '3m',
            5: '5m',
            15: '15m',
            30: '30m',
            60: '1h',
            120: '2h',
            240: '4h',
            360: '6h',
            480: '8h',
            720: '12h',
            1440: '1d',
        }
        return timeframe_map.get(minutes, '1m')
    
    def fetch_ohlcv(
        self,
        symbol: str,
        limit: Optional[int] = None,
        since: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC', 'ETH')
            limit: Number of candles to fetch
            since: Start datetime for fetching
            
        Returns:
            DataFrame with OHLCV data
        """
        # Convert symbol to Binance format (e.g., BTC -> BTC/USDT)
        trading_pair = f"{symbol}/USDT"
        
        if limit is None:
            limit = self.lookback_periods
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Fetch OHLCV data
                since_ms = int(since.timestamp() * 1000) if since else None
                
                ohlcv = self.exchange.fetch_ohlcv(
                    trading_pair,
                    timeframe=self.timeframe,
                    since=since_ms,
                    limit=limit
                )
                
                # Convert to DataFrame
                df = pd.DataFrame(
                    ohlcv,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                logger.info(f"Fetched {len(df)} candles for {symbol}")
                return df
                
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Error fetching OHLCV for {symbol} after {max_retries} attempts: {e}")
                    raise
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for {symbol}: {e}. Retrying...")
                time.sleep(1 + attempt)
    
    def fetch_all_data_since(self, symbol: str, since_date: datetime) -> pd.DataFrame:
        """
        Fetch all historical data for a symbol from a specific start date.
        Due to Binance limits, this paginates across multiple requests.
        """
        logger.info(f"Starting historical data collection for {symbol} since {since_date.strftime('%Y-%m-%d')}...")
        all_data = []
        current_since = since_date
        
        while True:
            # Binance limits to 1000 per request
            batch_df = self.fetch_ohlcv(
                symbol,
                limit=1000,
                since=current_since
            )
            
            if batch_df.empty:
                break
                
            all_data.append(batch_df)
            
            # If we returned fewer than 1000 candles, we hit the present time
            if len(batch_df) < 1000:
                break
                
            # Set the next request to start 1 minute after the last candle we just fetched
            current_since = batch_df.index[-1] + timedelta(minutes=1)
            time.sleep(0.1)  # Throttle to avoid getting rate limited immediately
            
        if not all_data:
            return pd.DataFrame()
            
        final_df = pd.concat(all_data)
        final_df = final_df[~final_df.index.duplicated(keep='last')]
        final_df.sort_index(inplace=True)
        logger.info(f"Successfully collected {len(final_df)} historical candles for {symbol}")
        return final_df
    
    def fetch_latest_candle(self, symbol: str) -> Optional[pd.Series]:
        """
        Fetch the latest candle for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Series with latest OHLCV data or None
        """
        try:
            df = self.fetch_ohlcv(symbol, limit=1)
            if not df.empty:
                return df.iloc[-1]
            return None
        except Exception:
            return None
    
    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        limit: Optional[int] = None
    ) -> dict:
        """
        Fetch OHLCV data for multiple symbols.
        
        Args:
            symbols: List of trading symbols
            limit: Number of candles to fetch per symbol
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        results = {}
        
        for symbol in symbols:
            try:
                df = self.fetch_ohlcv(symbol, limit=limit)
                if not df.empty:
                    results[symbol] = df
            except Exception:
                pass
            
            # Rate limiting
            time.sleep(0.1)
        
        logger.info(f"Fetched data for {len(results)}/{len(symbols)} symbols")
        return results
    
    def update_data(self, symbol: str, existing_df: pd.DataFrame) -> pd.DataFrame:
        """
        Update existing data with new candles.
        Fetches ALL available candles since last timestamp by making multiple requests if needed.
        
        Args:
            symbol: Trading symbol
            existing_df: Existing OHLCV DataFrame
            
        Returns:
            Updated DataFrame
        """
        if existing_df.empty:
            return self.fetch_ohlcv(symbol)
        
        # Get last timestamp
        last_timestamp = existing_df.index[-1]
        
        # Calculate expected number of new candles based on time gap
        time_gap = datetime.utcnow() - last_timestamp.to_pydatetime()
        expected_candles = int(time_gap.total_seconds() / 60) + 5  # Add 5 as buffer
        
        # Always fetch at least the last 5 minutes of candles for a rolling window
        min_candles = 10  # 5 minutes + 5 buffer
        smart_limit = max(min_candles, min(expected_candles, 1000))
        
        # Fetch all new data since last timestamp (may require multiple requests)
        all_new_data = []
        current_since = last_timestamp + timedelta(minutes=1)
        
        while True:
            # Fetch batch of candles
            batch_df = self.fetch_ohlcv(
                symbol,
                since=current_since,
                limit=smart_limit  # Use calculated limit instead of always 1000
            )
            
            if batch_df.empty:
                break
            
            all_new_data.append(batch_df)
            
            # If we got less than requested, we've fetched all available data
            if len(batch_df) < smart_limit:
                break
            
            # Update since timestamp for next batch
            current_since = batch_df.index[-1] + timedelta(minutes=1)
            
            # Small delay to respect rate limits
            time.sleep(0.1)
        
        if not all_new_data:
            return existing_df
        
        # Combine all batches
        new_df = pd.concat(all_new_data)
        
        # Concatenate with existing data and remove duplicates
        updated_df = pd.concat([existing_df, new_df])
        updated_df = updated_df[~updated_df.index.duplicated(keep='last')]
        updated_df.sort_index(inplace=True)
        
        logger.info(f"Updated {symbol}: added {len(new_df)} new candles")
        return updated_df
