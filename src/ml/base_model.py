"""
Base model abstractions for ML models.
Provides common interface for all models.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import logging
from datetime import datetime

from src.config import config


logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all ML models."""
    
    def __init__(self, coin: str, regime: str):
        """
        Initialize base model.
        
        Args:
            coin: Trading symbol (e.g., 'BTC')
            regime: Regime this model is for (e.g., 'BULL', 'SIDEWAYS_QUIET')
        """
        self.coin = coin
        self.regime = regime
        self.model = None
        self.feature_names = None
        self.trained_at = None
        self.model_version = "1.0"
        
        self.model_dir = config.model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Train the model.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            Dictionary with training metrics
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predictions
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of prediction probabilities
        """
        pass
    
    def save(self, filepath: Path = None) -> bool:
        """
        Save model to disk.
        
        Args:
            filepath: Optional custom filepath
            
        Returns:
            True if successful
        """
        if self.model is None:
            logger.warning(f"No model to save for {self.coin} {self.regime}")
            return False
        
        if filepath is None:
            filepath = self.get_model_path()
        
        try:
            model_data = {
                'model': self.model,
                'feature_names': self.feature_names,
                'trained_at': self.trained_at,
                'model_version': self.model_version,
                'coin': self.coin,
                'regime': self.regime,
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Saved model to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load(self, filepath: Path = None) -> bool:
        """
        Load model from disk.
        
        Args:
            filepath: Optional custom filepath
            
        Returns:
            True if successful
        """
        if filepath is None:
            filepath = self.get_model_path()
        
        if not filepath.exists():
            logger.warning(f"Model file not found: {filepath}")
            return False
        
        try:
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.trained_at = model_data['trained_at']
            self.model_version = model_data.get('model_version', '1.0')
            
            logger.info(f"Loaded model from {filepath} (trained at {self.trained_at})")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def get_model_path(self) -> Path:
        """Get default model file path."""
        filename = f"{self.coin}_{self.regime}_model.pkl"
        return self.model_dir / filename
    
    def is_trained(self) -> bool:
        """Check if model is trained."""
        return self.model is not None
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance if available.
        
        Returns:
            DataFrame with feature importance or None
        """
        if not self.is_trained():
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            })
            importance_df = importance_df.sort_values('importance', ascending=False)
            return importance_df
        
        return None
    
    def validate_features(self, X: pd.DataFrame) -> bool:
        """
        Validate that features match training features.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            True if features are valid
        """
        if self.feature_names is None:
            return True
        
        missing_features = set(self.feature_names) - set(X.columns)
        extra_features = set(X.columns) - set(self.feature_names)
        
        if missing_features:
            logger.error(f"Missing features: {missing_features}")
            return False
        
        if extra_features:
            logger.warning(f"Extra features will be ignored: {extra_features}")
        
        return True
    
    def prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for prediction (select and order correctly).
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Prepared feature DataFrame
        """
        if self.feature_names is None:
            return X
        
        # Select only the features used during training, in the same order
        return X[self.feature_names]
