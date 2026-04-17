"""
Ensemble regime classifier.
Combines trend and sideways detection into unified regime labels.
"""

import pandas as pd
import logging

from src.config import config
from src.regime.trend_detector import TrendDetector
from src.regime.sideways_detector import SidewaysDetector


logger = logging.getLogger(__name__)


class RegimeClassifier:
    """
    Unified regime classification combining trend and sideways detection.
    
    Produces final regime labels:
    - BULL
    - BEAR
    - SIDEWAYS_QUIET
    - SIDEWAYS_VOLATILE
    - CHOPPY
    """
    
    def __init__(self):
        """Initialize regime classifier."""
        self.trend_detector = TrendDetector()
        self.sideways_detector = SidewaysDetector()
        
        logger.info("Initialized regime classifier")
    
    def classify_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify all regimes for a DataFrame.
        
        Args:
            df: DataFrame with OHLCV and features
            
        Returns:
            DataFrame with 'regime' column containing final regime labels
        """
        df = df.copy()
        
        # Step 1: Detect BULL/BEAR/NEUTRAL
        df = self.trend_detector.detect_regime(df)
        
        # Step 2: Decompose NEUTRAL into sideways sub-regimes
        df = self.sideways_detector.detect_sideways_regime(df)
        
        # Step 3: Create unified regime column
        df['regime'] = df['trend_regime']
        
        # Replace NEUTRAL with sideways sub-regimes
        neutral_mask = df['trend_regime'] == 'NEUTRAL'
        df.loc[neutral_mask, 'regime'] = df.loc[neutral_mask, 'sideways_regime']
        
        # Calculate regime confidence
        df = self._calculate_regime_confidence(df)
        
        # Log final regime distribution
        regime_counts = df['regime'].value_counts()
        logger.info(f"Final regime distribution: {regime_counts.to_dict()}")
        
        return df
    
    def _calculate_regime_confidence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate confidence scores for regime classification.
        
        Args:
            df: DataFrame with regime classifications
            
        Returns:
            DataFrame with regime_confidence column
        """
        df = df.copy()
        
        # Get trend strength for BULL/BEAR
        df = self.trend_detector.get_regime_strength(df)
        
        # Get sideways confidence for NEUTRAL sub-regimes
        df = self.sideways_detector.calculate_sideways_confidence(df)
        
        # Combine into single confidence score
        df['regime_confidence'] = 0.5  # Default medium confidence
        
        # BULL/BEAR confidence based on trend strength
        bull_bear_mask = df['regime'].isin(['BULL', 'BEAR'])
        if bull_bear_mask.any():
            # Normalize trend strength to 0-1 range
            trend_conf = df.loc[bull_bear_mask, 'trend_strength'].clip(0, 0.1) / 0.1
            df.loc[bull_bear_mask, 'regime_confidence'] = trend_conf
        
        # Sideways confidence
        sideways_mask = df['regime'].str.contains('SIDEWAYS', na=False)
        if sideways_mask.any() and 'sideways_confidence' in df.columns:
            df.loc[sideways_mask, 'regime_confidence'] = df.loc[
                sideways_mask, 'sideways_confidence'
            ]
        
        # CHOPPY gets lower confidence
        choppy_mask = df['regime'] == 'CHOPPY'
        df.loc[choppy_mask, 'regime_confidence'] = 0.3
        
        return df
    
    def get_current_regime(self, df: pd.DataFrame) -> tuple:
        """
        Get current regime and confidence.
        
        Args:
            df: DataFrame with regime classification
            
        Returns:
            Tuple of (regime, confidence)
        """
        if 'regime' not in df.columns:
            df = self.classify_regimes(df)
        
        current_regime = df['regime'].iloc[-1]
        current_confidence = df['regime_confidence'].iloc[-1]
        
        return current_regime, current_confidence
    
    def get_regime_statistics(self, df: pd.DataFrame) -> dict:
        """
        Get statistics about regime distribution.
        
        Args:
            df: DataFrame with regime classification
            
        Returns:
            Dictionary with regime statistics
        """
        if 'regime' not in df.columns:
            df = self.classify_regimes(df)
        
        total_bars = len(df)
        regime_counts = df['regime'].value_counts()
        
        stats = {
            'total_bars': total_bars,
            'regime_counts': regime_counts.to_dict(),
            'regime_percentages': (regime_counts / total_bars * 100).to_dict(),
            'current_regime': self.get_current_regime(df)[0],
            'current_confidence': self.get_current_regime(df)[1],
        }
        
        # Average confidence per regime
        avg_confidence = df.groupby('regime')['regime_confidence'].mean()
        stats['avg_confidence_by_regime'] = avg_confidence.to_dict()
        
        return stats
    
    def filter_by_regime(self, df: pd.DataFrame, regime: str) -> pd.DataFrame:
        """
        Filter DataFrame to only include specific regime.
        
        Args:
            df: DataFrame with regime classification
            regime: Regime to filter for
            
        Returns:
            Filtered DataFrame
        """
        if 'regime' not in df.columns:
            df = self.classify_regimes(df)
        
        return df[df['regime'] == regime].copy()
    
    def get_regime_transitions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify regime transition points.
        
        Args:
            df: DataFrame with regime classification
            
        Returns:
            DataFrame of transition points
        """
        if 'regime' not in df.columns:
            df = self.classify_regimes(df)
        
        # Find where regime changes
        regime_changes = df['regime'] != df['regime'].shift(1)
        transitions = df[regime_changes].copy()
        
        # Add previous regime
        transitions['previous_regime'] = df['regime'].shift(1)[regime_changes]
        
        logger.info(f"Found {len(transitions)} regime transitions")
        
        return transitions
