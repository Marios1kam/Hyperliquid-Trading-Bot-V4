"""
Technical indicators with configurable parameters.
All indicator settings loaded from configuration.
"""

import pandas as pd
import numpy as np
from typing import List
import logging

from src.config import config


logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Compute technical indicators from OHLCV data."""
    
    def __init__(self):
        """Initialize with configuration parameters."""
        # Load all indicator parameters from config
        self.ma_periods = config.get('MA_PERIODS', [5, 10, 20, 50])
        self.rsi_period = config.get('RSI_PERIOD', 14)
        self.atr_period = config.get('ATR_PERIOD', 14)
        
        self.macd_fast = config.get('MACD_FAST_PERIOD', 12)
        self.macd_slow = config.get('MACD_SLOW_PERIOD', 26)
        self.macd_signal = config.get('MACD_SIGNAL_PERIOD', 9)
        
        self.bb_window = config.get('BB_WINDOW', 20)
        self.bb_std_dev = config.get('BB_STD_DEV', 2.0)
        
        self.volume_ma_periods = config.get('VOLUME_MA_PERIODS', [20, 50])
        self.volume_spike_multiplier = config.get('VOLUME_SPIKE_MULTIPLIER', 2.0)
        
        logger.info("Initialized technical indicators with config parameters")
    
    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all technical indicators.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with all indicators added
        """
        df = df.copy()
        
        # Moving averages
        df = self.add_moving_averages(df)
        
        # RSI
        df = self.add_rsi(df)
        
        # MACD
        df = self.add_macd(df)
        
        # Bollinger Bands
        df = self.add_bollinger_bands(df)
        
        # ATR
        df = self.add_atr(df)
        
        # Volume indicators
        df = self.add_volume_indicators(df)
        
        return df
    
    def add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add moving averages with configured periods."""
        for period in self.ma_periods:
            df[f'ma_{period}'] = df['close'].rolling(window=period).mean()
        return df
    
    def add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add RSI indicator."""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add MACD indicator."""
        ema_fast = df['close'].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.macd_slow, adjust=False).mean()
        
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=self.macd_signal, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        return df
    
    def add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Bollinger Bands."""
        df['bb_middle'] = df['close'].rolling(window=self.bb_window).mean()
        bb_std = df['close'].rolling(window=self.bb_window).std()
        
        df['bb_upper'] = df['bb_middle'] + (bb_std * self.bb_std_dev)
        df['bb_lower'] = df['bb_middle'] - (bb_std * self.bb_std_dev)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Position within bands
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    def add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=self.atr_period).mean()
        
        # ATR as percentage of price
        df['atr_pct'] = df['atr'] / df['close']
        
        return df
    
    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""
        # Volume moving averages
        for period in self.volume_ma_periods:
            df[f'volume_ma_{period}'] = df['volume'].rolling(window=period).mean()
        
        # Volume spike detection
        if self.volume_ma_periods:
            base_period = self.volume_ma_periods[0]
            df['volume_spike'] = (
                df['volume'] > df[f'volume_ma_{base_period}'] * self.volume_spike_multiplier
            ).astype(int)
        
        # Volume trend
        if len(self.volume_ma_periods) >= 2:
            short_period = self.volume_ma_periods[0]
            long_period = self.volume_ma_periods[1]
            df['volume_trend'] = (
                df[f'volume_ma_{short_period}'] / df[f'volume_ma_{long_period}']
            )
        
        # On-Balance Volume (OBV)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        return df
    
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic price-derived features."""
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # High-Low range
        df['hl_range'] = (df['high'] - df['low']) / df['close']
        
        # Close position in daily range
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        return df
    
    def add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        # Rate of change
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = df['close'].pct_change(periods=period)
        
        # Price momentum
        df['momentum'] = df['close'] - df['close'].shift(10)
        
        return df
