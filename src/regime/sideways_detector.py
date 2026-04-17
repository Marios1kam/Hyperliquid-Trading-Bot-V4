"""
Sideways market detector with sub-regime classification.
Decomposes NEUTRAL regime into SIDEWAYS_QUIET, SIDEWAYS_VOLATILE, and CHOPPY.
All parameters loaded from configuration.
"""

import pandas as pd
import numpy as np
import logging

from src.config import config


logger = logging.getLogger(__name__)


class SidewaysDetector:
    """
    Detect and classify sideways market sub-regimes.
    
    Implements structural sideways definition using:
    - Trend filters (near-zero slope)
    - Volatility constraints (quiet vs volatile)
    - Band constraints (range-bound behavior)
    - Choppy detection (frequent trend flips)
    """
    
    def __init__(self):
        """Initialize with configuration parameters."""
        sideways_config = config.get('SIDEWAYS', {})
        
        # Volatility thresholds
        self.vol_quiet_max = sideways_config.get('VOL_QUIET_MAX', 0.005)
        self.vol_volatile_min = sideways_config.get('VOL_VOLATILE_MIN', 0.015)
        
        # Band constraints
        self.range_window = sideways_config.get('RANGE_WINDOW', 100)
        self.band_max_drawup = sideways_config.get('BAND_MAX_DRAWUP', 0.01)
        self.band_max_drawdown = sideways_config.get('BAND_MAX_DRAWDOWN', 0.01)
        self.band_width_quiet_max = sideways_config.get('BAND_WIDTH_QUIET_MAX', 0.008)
        
        # Trend filters
        self.sideways_slope_window = sideways_config.get('SIDEWAYS_SLOPE_WINDOW', 20)
        self.sideways_slope_threshold = sideways_config.get('SIDEWAYS_SLOPE_THRESHOLD', 0.0005)
        
        # Choppy detection
        self.trend_sign_window = sideways_config.get('TREND_SIGN_WINDOW', 10)
        self.trend_flip_rate_min = sideways_config.get('TREND_FLIP_RATE_MIN', 0.4)
        
        logger.info(f"Initialized sideways detector: vol_quiet<{self.vol_quiet_max}, "
                   f"vol_volatile>{self.vol_volatile_min}, slope_threshold={self.sideways_slope_threshold}")
    
    def detect_sideways_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and classify sideways sub-regimes.
        
        Args:
            df: DataFrame with features and trend_regime
            
        Returns:
            DataFrame with 'sideways_regime' column added
        """
        df = df.copy()
        
        # Only process NEUTRAL segments
        neutral_mask = df['trend_regime'] == 'NEUTRAL'
        
        if not neutral_mask.any():
            df['sideways_regime'] = None
            logger.info("No NEUTRAL regime detected, skipping sideways classification")
            return df
        
        # Calculate sideways detection features if not present
        df = self._ensure_sideways_features(df)
        
        # Initialize sideways regime column
        df['sideways_regime'] = None
        
        # Apply structural sideways filters
        is_sideways = self._apply_sideways_filters(df)
        
        # Among sideways segments, classify into sub-regimes
        df.loc[neutral_mask & is_sideways, 'sideways_regime'] = self._classify_sideways_subregime(
            df[neutral_mask & is_sideways]
        )
        
        # Remaining NEUTRAL that doesn't meet sideways criteria is CHOPPY
        df.loc[neutral_mask & ~is_sideways, 'sideways_regime'] = 'CHOPPY'
        
        # Log sub-regime distribution
        sideways_counts = df['sideways_regime'].value_counts()
        logger.info(f"Sideways sub-regime distribution: {sideways_counts.to_dict()}")
        
        return df
    
    def _ensure_sideways_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required sideways features are computed."""
        # Realized volatility (if not present)
        if 'realized_vol_20' not in df.columns:
            log_returns = np.log(df['close'] / df['close'].shift(1))
            periods_per_year = 525600 / config.get('TIMEFRAME_MINUTES', 1)
            df['realized_vol_20'] = (
                log_returns.rolling(window=20).std() * np.sqrt(periods_per_year)
            )
        
        # MA slope for trend filter (if not present)
        if 'sideways_ma_slope' not in df.columns:
            ma = df['close'].rolling(window=self.sideways_slope_window).mean()
            df['sideways_ma_slope'] = ma.pct_change(periods=self.sideways_slope_window)
        
        # Range features (if not present)
        if 'range_width' not in df.columns:
            rolling_high = df['high'].rolling(window=self.range_window).max()
            rolling_low = df['low'].rolling(window=self.range_window).min()
            df['range_width'] = (rolling_high - rolling_low) / df['close']
        
        # Trend flip rate (if not present)
        if 'trend_flip_rate' not in df.columns:
            short_trend = np.sign(df['close'].diff(periods=1))
            
            def count_flips(series):
                if len(series) < 2:
                    return 0
                return (series.diff() != 0).sum()
            
            trend_flips = short_trend.rolling(window=self.trend_sign_window).apply(
                count_flips, raw=False
            )
            df['trend_flip_rate'] = trend_flips / self.trend_sign_window
        
        return df
    
    def _apply_sideways_filters(self, df: pd.DataFrame) -> pd.Series:
        """
        Apply structural sideways filters.
        
        Returns:
            Boolean series indicating sideways segments
        """
        # Filter 1: Near-zero MA slope (trend filter)
        slope_filter = np.abs(df['sideways_ma_slope']) < self.sideways_slope_threshold
        
        # Filter 2: Price stays within volatility-scaled bands
        # Calculate drawup and drawdown from rolling anchor
        anchor = df['close'].rolling(window=self.range_window).mean()
        drawup = (df['close'] - anchor) / anchor
        drawdown = (anchor - df['close']) / anchor
        
        band_filter = (
            (drawup <= self.band_max_drawup) &
            (drawdown <= self.band_max_drawdown)
        )
        
        # Combine filters
        is_sideways = slope_filter & band_filter
        
        return is_sideways
    
    def _classify_sideways_subregime(self, df: pd.DataFrame) -> pd.Series:
        """
        Classify sideways segments into QUIET, VOLATILE, or CHOPPY.
        
        Args:
            df: DataFrame containing only sideways segments
            
        Returns:
            Series with sub-regime labels
        """
        # Use realized volatility for classification
        vol = df['realized_vol_20']
        
        # SIDEWAYS_QUIET: Low volatility and narrow bands
        quiet_conditions = (
            (vol < self.vol_quiet_max) &
            (df['range_width'] < self.band_width_quiet_max)
        )
        
        # SIDEWAYS_VOLATILE: High volatility but still range-bound
        volatile_conditions = (
            (vol > self.vol_volatile_min) &
            ~quiet_conditions
        )
        
        # CHOPPY: Frequent trend flips (even if technically sideways)
        choppy_conditions = (
            (df['trend_flip_rate'] > self.trend_flip_rate_min) &
            ~quiet_conditions &
            ~volatile_conditions
        )
        
        # Assign sub-regimes
        sub_regime = pd.Series('SIDEWAYS_QUIET', index=df.index)
        sub_regime[volatile_conditions] = 'SIDEWAYS_VOLATILE'
        sub_regime[choppy_conditions] = 'CHOPPY'
        
        return sub_regime
    
    def get_current_sideways_regime(self, df: pd.DataFrame) -> str:
        """
        Get the current sideways sub-regime.
        
        Args:
            df: DataFrame with sideways_regime column
            
        Returns:
            Current sideways regime or None
        """
        if 'sideways_regime' not in df.columns:
            df = self.detect_sideways_regime(df)
        
        return df['sideways_regime'].iloc[-1]
    
    def calculate_sideways_confidence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate confidence scores for sideways detection.
        
        Args:
            df: DataFrame with sideways features
            
        Returns:
            DataFrame with sideways_confidence column
        """
        df = df.copy()
        
        # Confidence based on how well conditions are met
        # Slope confidence (closer to zero = higher confidence)
        slope_conf = 1 - np.minimum(
            np.abs(df['sideways_ma_slope']) / self.sideways_slope_threshold,
            1.0
        )
        
        # Band confidence (staying within bands)
        anchor = df['close'].rolling(window=self.range_window).mean()
        drawup = np.abs((df['close'] - anchor) / anchor)
        drawdown = np.abs((anchor - df['close']) / anchor)
        
        band_conf = 1 - np.maximum(
            drawup / self.band_max_drawup,
            drawdown / self.band_max_drawdown
        ).clip(0, 1)
        
        # Combined confidence
        df['sideways_confidence'] = (slope_conf + band_conf) / 2
        
        return df
    
    def smooth_sideways_transitions(
        self,
        df: pd.DataFrame,
        min_duration: int = 20
    ) -> pd.DataFrame:
        """
        Smooth sideways sub-regime transitions.
        
        Args:
            df: DataFrame with sideways_regime column
            min_duration: Minimum bars to stay in a sub-regime
            
        Returns:
            DataFrame with smoothed sideways regimes
        """
        df = df.copy()
        
        if 'sideways_regime' not in df.columns:
            return df
        
        # Forward fill short regime periods
        regime_changes = df['sideways_regime'] != df['sideways_regime'].shift(1)
        regime_groups = regime_changes.cumsum()
        
        regime_counts = df.groupby(regime_groups)['sideways_regime'].transform('count')
        short_regimes = regime_counts < min_duration
        
        df.loc[short_regimes, 'sideways_regime'] = np.nan
        df['sideways_regime'] = df['sideways_regime'].fillna(method='ffill')
        
        return df
