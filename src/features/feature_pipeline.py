"""
Feature pipeline orchestrator.
Combines all feature engineering modules.
"""

import numpy as np
import pandas as pd
import logging

from src.config import config
from src.features.indicators import TechnicalIndicators
from src.features.sideways_features import SidewaysFeatures


logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Orchestrate feature engineering across all modules."""
    
    def __init__(self):
        """Initialize feature pipeline."""
        self.tech_indicators = TechnicalIndicators()
        self.sideways_features = SidewaysFeatures()
        
        logger.info("Initialized feature pipeline")
    
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features for a given OHLCV DataFrame.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with all features
        """
        if df.empty:
            logger.warning("Empty DataFrame provided to feature pipeline")
            return df
        
        df = df.copy()
        
        # Basic price features
        df = self.tech_indicators.add_price_features(df)
        
        # Technical indicators
        df = self.tech_indicators.compute_all(df)
        
        # Momentum features
        df = self.tech_indicators.add_momentum_features(df)
        
        # Sideways-specific features
        df = self.sideways_features.compute_all(df)
        
        # Trend flip features for choppy detection
        df = self.sideways_features.add_trend_flip_features(df)
        
        # Range breakout detection
        df = self.sideways_features.detect_range_breakout(df)
        
        # Drop rows with NaN values (from rolling calculations)
        initial_rows = len(df)
        df = df.dropna()
        dropped_rows = initial_rows - len(df)
        
        if dropped_rows > 0:
            logger.debug(f"Dropped {dropped_rows} rows with NaN values after feature computation")
        
        logger.info(f"Computed features: {len(df)} rows, {len(df.columns)} columns")
        
        return df
    
    def get_feature_names(self) -> list:
        """
        Get list of all feature column names.
        
        Returns:
            List of feature names
        """
        # Create a sample DataFrame to extract feature names
        sample_df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000, 1100, 1200]
        })
        
        featured_df = self.compute_features(sample_df)
        
        # Exclude OHLCV columns
        base_columns = ['open', 'high', 'low', 'close', 'volume']
        feature_columns = [col for col in featured_df.columns if col not in base_columns]
        
        return feature_columns
    
    def select_features_for_regime(self, df: pd.DataFrame, regime: str) -> pd.DataFrame:
        """
        Select relevant features for a specific regime.
        
        Args:
            df: DataFrame with all features
            regime: Regime name (BULL, BEAR, NEUTRAL_QUIET, etc.)
            
        Returns:
            DataFrame with selected features
        """
        # For now, use all features for all regimes
        # Can be customized per regime if needed
        return df
    
    def validate_features(self, df: pd.DataFrame) -> tuple:
        """
        Validate computed features.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Tuple of (is_valid, issues_list)
        """
        issues = []
        
        # Check for infinite values
        inf_cols = df.columns[df.isin([np.inf, -np.inf]).any()].tolist()
        if inf_cols:
            issues.append(f"Infinite values in columns: {inf_cols}")
        
        # Check for excessive NaN values
        nan_pct = (df.isna().sum() / len(df)) * 100
        high_nan_cols = nan_pct[nan_pct > 10].index.tolist()
        if high_nan_cols:
            issues.append(f"High NaN percentage in columns: {high_nan_cols}")
        
        # Check for constant columns
        constant_cols = df.columns[df.nunique() == 1].tolist()
        if constant_cols:
            issues.append(f"Constant value columns: {constant_cols}")
        
        is_valid = len(issues) == 0
        return is_valid, issues
