"""
Sideways-specific features for range detection and mean-reversion.
All parameters loaded from configuration.
"""

import pandas as pd
import numpy as np
from typing import Tuple
import logging

from src.config import config


logger = logging.getLogger(__name__)


class SidewaysFeatures:
    """Compute features specific to sideways market detection and trading."""
    
    def __init__(self):
        """Initialize with configuration parameters."""
        self.support_resist_lookback = config.get('SUPPORT_RESIST_LOOKBACK', 100)
        self.rolling_range_window = config.get('ROLLING_RANGE_WINDOW', 100)
        self.short_rsi_period = config.get('SHORT_RSI_PERIOD', 7)
        self.realized_vol_windows = config.get('REALIZED_VOL_WINDOWS', [20, 50])
        
        logger.info("Initialized sideways features with config parameters")
    
    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all sideways-specific features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with sideways features added
        """
        df = df.copy()
        
        # Support and resistance
        df = self.add_support_resistance(df)
        
        # Rolling range features
        df = self.add_rolling_range_features(df)
        
        # Short-term RSI
        df = self.add_short_rsi(df)
        
        # Realized volatility
        df = self.add_realized_volatility(df)
        
        # Band features
        df = self.add_band_features(df)
        
        return df
    
    def add_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add support and resistance levels and distances.
        
        Uses rolling window to find recent highs/lows.
        """
        # Rolling support (lowest low in lookback window)
        df['support'] = df['low'].rolling(
            window=self.support_resist_lookback,
            min_periods=1
        ).min()
        
        # Rolling resistance (highest high in lookback window)
        df['resistance'] = df['high'].rolling(
            window=self.support_resist_lookback,
            min_periods=1
        ).max()
        
        # Distance to support/resistance as percentage
        df['dist_to_support'] = (df['close'] - df['support']) / df['close']
        df['dist_to_resistance'] = (df['resistance'] - df['close']) / df['close']
        
        # Position between support and resistance
        df['sr_position'] = (
            (df['close'] - df['support']) / 
            (df['resistance'] - df['support'])
        ).clip(0, 1)
        
        return df
    
    def add_rolling_range_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features based on rolling range."""
        # Rolling high and low
        df['rolling_high'] = df['high'].rolling(
            window=self.rolling_range_window,
            min_periods=1
        ).max()
        
        df['rolling_low'] = df['low'].rolling(
            window=self.rolling_range_window,
            min_periods=1
        ).min()
        
        # Range width
        df['range_width'] = (
            (df['rolling_high'] - df['rolling_low']) / df['close']
        )
        
        # Position within rolling range
        df['range_position'] = (
            (df['close'] - df['rolling_low']) / 
            (df['rolling_high'] - df['rolling_low'])
        ).clip(0, 1)
        
        # Distance from range edges
        df['dist_from_high'] = (df['rolling_high'] - df['close']) / df['close']
        df['dist_from_low'] = (df['close'] - df['rolling_low']) / df['close']
        
        return df
    
    def add_short_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add short-term RSI for sideways detection."""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(
            window=self.short_rsi_period
        ).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(
            window=self.short_rsi_period
        ).mean()
        
        rs = gain / loss
        df['short_rsi'] = 100 - (100 / (1 + rs))
        
        # RSI extremes (for mean-reversion signals)
        df['rsi_oversold'] = (df['short_rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['short_rsi'] > 70).astype(int)
        
        return df
    
    def add_realized_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add realized volatility over multiple windows."""
        # Log returns for volatility calculation
        log_returns = np.log(df['close'] / df['close'].shift(1))
        
        for window in self.realized_vol_windows:
            # Realized volatility (annualized)
            # Assuming 1-minute data: 60*24*365 = 525,600 minutes per year
            periods_per_year = 525600 / config.get('TIMEFRAME_MINUTES', 1)
            
            df[f'realized_vol_{window}'] = (
                log_returns.rolling(window=window).std() * 
                np.sqrt(periods_per_year)
            )
        
        # Volatility ratio (short vs long)
        if len(self.realized_vol_windows) >= 2:
            short_window = self.realized_vol_windows[0]
            long_window = self.realized_vol_windows[1]
            
            df['vol_ratio'] = (
                df[f'realized_vol_{short_window}'] / 
                df[f'realized_vol_{long_window}']
            )
        
        return df
    
    def add_band_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility-scaled band features.
        
        These bands are used for structural sideways definition.
        """
        # Use ATR if available, otherwise calculate
        if 'atr' not in df.columns:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean()
        else:
            atr = df['atr']
        
        # Volatility-scaled bands
        band_multiplier = config.get('SIDEWAYS.BAND_MAX_DRAWUP', 0.01)
        
        # Simple moving average as anchor
        anchor_period = config.get('SIDEWAYS.RANGE_WINDOW', 100)
        df['band_anchor'] = df['close'].rolling(window=anchor_period).mean()
        
        # Upper and lower bands based on volatility
        df['vol_band_upper'] = df['band_anchor'] * (1 + band_multiplier)
        df['vol_band_lower'] = df['band_anchor'] * (1 - band_multiplier)
        
        # Check if price is within bands
        df['within_bands'] = (
            (df['close'] >= df['vol_band_lower']) & 
            (df['close'] <= df['vol_band_upper'])
        ).astype(int)
        
        # Band width as percentage
        df['vol_band_width'] = (
            (df['vol_band_upper'] - df['vol_band_lower']) / df['band_anchor']
        )
        
        return df
    
    def add_trend_flip_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features for detecting choppy/directionless markets.
        
        Counts trend sign flips over a window.
        """
        trend_window = config.get('SIDEWAYS.TREND_SIGN_WINDOW', 10)
        
        # Short-term trend direction
        df['short_trend'] = np.sign(df['close'].diff(periods=1))
        
        # Count sign flips in window
        def count_flips(series):
            """Count number of sign changes in a series."""
            if len(series) < 2:
                return 0
            changes = (series.diff() != 0).sum()
            return changes
        
        df['trend_flips'] = df['short_trend'].rolling(
            window=trend_window
        ).apply(count_flips, raw=False)
        
        # Flip rate (normalized)
        df['trend_flip_rate'] = df['trend_flips'] / trend_window
        
        return df
    
    def detect_range_breakout(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add range breakout detection features."""
        # Breakout above resistance
        df['breakout_up'] = (
            (df['close'] > df['resistance']) & 
            (df['close'].shift(1) <= df['resistance'].shift(1))
        ).astype(int)
        
        # Breakdown below support
        df['breakout_down'] = (
            (df['close'] < df['support']) & 
            (df['close'].shift(1) >= df['support'].shift(1))
        ).astype(int)
        
        return df
