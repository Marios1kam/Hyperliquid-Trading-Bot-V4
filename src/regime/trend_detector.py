"""
BULL/BEAR trend detection.
All parameters loaded from configuration.
"""

import pandas as pd
import numpy as np
import logging

from src.config import config


logger = logging.getLogger(__name__)


class TrendDetector:
    """Detect BULL and BEAR market regimes."""
    
    def __init__(self):
        """Initialize with configuration parameters."""
        # Load BULL regime parameters
        self.bull_config = config.get('REGIME.BULL', {})
        self.bull_fast_ma = self.bull_config.get('FAST_MA_PERIOD', 20)
        self.bull_slow_ma = self.bull_config.get('SLOW_MA_PERIOD', 50)
        self.bull_slope_lookback = self.bull_config.get('MA_SLOPE_LOOKBACK', 10)
        self.bull_uptrend_threshold = self.bull_config.get('UPTREND_THRESHOLD', 0.001)
        
        # Load BEAR regime parameters
        self.bear_config = config.get('REGIME.BEAR', {})
        self.bear_fast_ma = self.bear_config.get('FAST_MA_PERIOD', 20)
        self.bear_slow_ma = self.bear_config.get('SLOW_MA_PERIOD', 50)
        self.bear_slope_lookback = self.bear_config.get('MA_SLOPE_LOOKBACK', 10)
        self.bear_downtrend_threshold = self.bear_config.get('DOWNTREND_THRESHOLD', -0.001)
        
        logger.info(f"Initialized trend detector: BULL MA({self.bull_fast_ma}/{self.bull_slow_ma}), "
                   f"BEAR MA({self.bear_fast_ma}/{self.bear_slow_ma})")
    
    def detect_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect BULL/BEAR regimes for entire DataFrame.
        
        Args:
            df: DataFrame with OHLCV and features
            
        Returns:
            DataFrame with 'trend_regime' column added
        """
        df = df.copy()
        
        # Calculate MAs if not present
        if f'ma_{self.bull_fast_ma}' not in df.columns:
            df[f'ma_{self.bull_fast_ma}'] = df['close'].rolling(
                window=self.bull_fast_ma
            ).mean()
        
        if f'ma_{self.bull_slow_ma}' not in df.columns:
            df[f'ma_{self.bull_slow_ma}'] = df['close'].rolling(
                window=self.bull_slow_ma
            ).mean()
        
        # Calculate MA slopes
        df['fast_ma_slope'] = self._calculate_slope(
            df[f'ma_{self.bull_fast_ma}'],
            self.bull_slope_lookback
        )
        
        df['slow_ma_slope'] = self._calculate_slope(
            df[f'ma_{self.bull_slow_ma}'],
            self.bull_slope_lookback
        )
        
        # Detect BULL regime
        bull_conditions = (
            (df['close'] > df[f'ma_{self.bull_fast_ma}']) &
            (df[f'ma_{self.bull_fast_ma}'] > df[f'ma_{self.bull_slow_ma}']) &
            (df['fast_ma_slope'] > self.bull_uptrend_threshold) &
            (df['slow_ma_slope'] > self.bull_uptrend_threshold)
        )
        
        # Detect BEAR regime
        bear_conditions = (
            (df['close'] < df[f'ma_{self.bear_fast_ma}']) &
            (df[f'ma_{self.bear_fast_ma}'] < df[f'ma_{self.bear_slow_ma}']) &
            (df['fast_ma_slope'] < self.bear_downtrend_threshold) &
            (df['slow_ma_slope'] < self.bear_downtrend_threshold)
        )
        
        # Assign regimes
        df['trend_regime'] = 'NEUTRAL'
        df.loc[bull_conditions, 'trend_regime'] = 'BULL'
        df.loc[bear_conditions, 'trend_regime'] = 'BEAR'
        
        # Log regime distribution
        regime_counts = df['trend_regime'].value_counts()
        logger.info(f"Regime distribution: {regime_counts.to_dict()}")
        
        return df
    
    def _calculate_slope(self, series: pd.Series, lookback: int) -> pd.Series:
        """
        Calculate slope of a series over a lookback window.
        
        Args:
            series: Time series data
            lookback: Lookback period
            
        Returns:
            Series of slopes (normalized by value)
        """
        # Calculate percentage change over lookback period
        slope = series.pct_change(periods=lookback)
        return slope
    
    def get_current_regime(self, df: pd.DataFrame) -> str:
        """
        Get the current (most recent) regime.
        
        Args:
            df: DataFrame with trend_regime column
            
        Returns:
            Current regime ('BULL', 'BEAR', or 'NEUTRAL')
        """
        if 'trend_regime' not in df.columns:
            df = self.detect_regime(df)
        
        return df['trend_regime'].iloc[-1]
    
    def get_regime_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate regime strength indicators.
        
        Args:
            df: DataFrame with regime detection
            
        Returns:
            DataFrame with regime strength columns
        """
        df = df.copy()
        
        # MA separation (how far apart are the MAs)
        df['ma_separation'] = (
            (df[f'ma_{self.bull_fast_ma}'] - df[f'ma_{self.bull_slow_ma}']) /
            df[f'ma_{self.bull_slow_ma}']
        )
        
        # Price distance from slow MA
        df['price_ma_distance'] = (
            (df['close'] - df[f'ma_{self.bull_slow_ma}']) /
            df[f'ma_{self.bull_slow_ma}']
        )
        
        # Trend strength (combination of slope and separation)
        df['trend_strength'] = np.abs(df['ma_separation']) + np.abs(df['slow_ma_slope'])
        
        return df
    
    def smooth_regime_transitions(
        self,
        df: pd.DataFrame,
        min_regime_duration: int = 10
    ) -> pd.DataFrame:
        """
        Smooth regime transitions to avoid rapid switching.
        
        Args:
            df: DataFrame with trend_regime column
            min_regime_duration: Minimum bars to stay in a regime
            
        Returns:
            DataFrame with smoothed regimes
        """
        df = df.copy()
        
        if 'trend_regime' not in df.columns:
            return df
        
        # Forward fill short regime periods
        regime_changes = df['trend_regime'] != df['trend_regime'].shift(1)
        regime_groups = regime_changes.cumsum()
        
        # Count consecutive regime occurrences
        regime_counts = df.groupby(regime_groups)['trend_regime'].transform('count')
        
        # Mark short regimes for smoothing
        short_regimes = regime_counts < min_regime_duration
        
        # Forward fill from previous longer regime
        df.loc[short_regimes, 'trend_regime'] = np.nan
        df['trend_regime'] = df['trend_regime'].fillna(method='ffill')
        
        return df
