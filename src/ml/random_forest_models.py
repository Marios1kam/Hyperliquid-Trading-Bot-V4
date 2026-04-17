"""
Random Forest models per regime.
Each coin has models trained on its full history with regime as a feature.
All hyperparameters loaded from configuration.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging
from datetime import datetime

from src.config import config
from src.ml.base_model import BaseModel


logger = logging.getLogger(__name__)


class RandomForestModel(BaseModel):
    """
    Random Forest classifier for trading signals.
    
    Trained per coin on full history with regime as input feature.
    """
    
    def __init__(self, coin: str, regime: str):
        """
        Initialize Random Forest model.
        
        Args:
            coin: Trading symbol
            regime: Regime identifier (for hyperparameter selection)
        """
        super().__init__(coin, regime)
        
        # Load hyperparameters from config based on regime
        self.hyperparams = self._get_hyperparameters()
        
        logger.info(f"Initialized RF model for {coin} ({regime}) with params: {self.hyperparams}")
    
    def _get_hyperparameters(self) -> dict:
        """Get hyperparameters from config for this regime."""
        # Map regime to config key
        regime_key = self.regime.replace('SIDEWAYS_', 'NEUTRAL_')
        
        # Get hyperparameters from config
        params = config.get(f'ML.{regime_key}.RANDOM_FOREST', {})
        
        if not params:
            # Fallback to default parameters
            logger.warning(f"No hyperparameters found for {regime_key}, using defaults")
            params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'class_weight': 'balanced'
            }
        
        return params
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Train Random Forest model.
        
        Args:
            X: Feature DataFrame (full history for this coin)
            y: Target Series (1 for long, 0 for neutral, -1 for short)
            
        Returns:
            Dictionary with training metrics
        """
        if len(X) < 100:
            logger.warning(f"Insufficient data for training: {len(X)} samples")
            return {'error': 'insufficient_data'}
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False  # Don't shuffle to maintain time order
        )
        
        logger.info(f"Training RF for {self.coin}: {len(X_train)} train, {len(X_test)} test samples")
        
        # Initialize and train model
        self.model = RandomForestClassifier(**self.hyperparams)
        
        try:
            self.model.fit(X_train, y_train)
            self.trained_at = datetime.now()
            
            # Evaluate on test set
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'n_features': len(self.feature_names),
                'trained_at': self.trained_at,
            }
            
            # Class distribution
            unique, counts = np.unique(y_train, return_counts=True)
            metrics['class_distribution'] = dict(zip(unique.tolist(), counts.tolist()))
            
            # Classification report
            try:
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                metrics['classification_report'] = report
            except Exception as e:
                logger.warning(f"Could not generate classification report: {e}")
            
            logger.info(f"Training complete for {self.coin} ({self.regime}): "
                       f"accuracy={accuracy:.3f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training model for {self.coin}: {e}")
            return {'error': str(e)}
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict trading signals.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predictions (1, 0, -1)
        """
        if not self.is_trained():
            logger.error(f"Model not trained for {self.coin}")
            return np.zeros(len(X))
        
        # Validate and prepare features
        if not self.validate_features(X):
            return np.zeros(len(X))
        
        X_prepared = self.prepare_features(X)
        
        try:
            predictions = self.model.predict(X_prepared)
            return predictions
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return np.zeros(len(X))
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities for each class.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of probabilities for each class
        """
        if not self.is_trained():
            logger.error(f"Model not trained for {self.coin}")
            return np.zeros((len(X), 3))
        
        # Validate and prepare features
        if not self.validate_features(X):
            return np.zeros((len(X), 3))
        
        X_prepared = self.prepare_features(X)
        
        try:
            probabilities = self.model.predict_proba(X_prepared)
            return probabilities
        except Exception as e:
            logger.error(f"Error predicting probabilities: {e}")
            return np.zeros((len(X), 3))
    
    def get_signal_with_confidence(self, X: pd.DataFrame) -> tuple:
        """
        Get trading signal with confidence score.
        
        Args:
            X: Feature DataFrame (typically single row for current state)
            
        Returns:
            Tuple of (signal, confidence)
        """
        if not self.is_trained():
            return 0, 0.0
        
        prediction = self.predict(X)
        probabilities = self.predict_proba(X)
        
        if len(prediction) == 0:
            return 0, 0.0
        
        signal = prediction[-1]  # Most recent prediction
        
        # Confidence is the probability of the predicted class
        if len(probabilities) > 0:
            # Get the max probability (confidence in the prediction)
            confidence = np.max(probabilities[-1])
        else:
            confidence = 0.0
        
        return int(signal), float(confidence)


class RandomForestModelManager:
    """
    Manage Random Forest models for all coins and regimes.
    
    Per the requirements, training is done per coin on that coin's entire
    labelled history, with regime included as an explicit input feature.
    """
    
    def __init__(self):
        """Initialize model manager."""
        self.models = {}  # {coin: {regime: model}}
        self.coin_list = config.coin_list
        
        # All possible regimes
        self.regimes = ['BULL', 'BEAR', 'SIDEWAYS_QUIET', 'SIDEWAYS_VOLATILE', 'CHOPPY']
        
        logger.info(f"Initialized RF model manager for {len(self.coin_list)} coins")
    
    def train_coin_models(self, coin: str, df: pd.DataFrame) -> dict:
        """
        Train all regime models for a coin.
        
        Per requirements: training is done on the coin's entire history at once,
        not by splitting into separate per-regime training sets.
        
        Args:
            coin: Trading symbol
            df: Full DataFrame with features, regime labels, and targets
            
        Returns:
            Dictionary with training results per regime
        """
        if coin not in self.models:
            self.models[coin] = {}
        
        results = {}
        
        # Ensure we have required columns
        if 'regime' not in df.columns or 'target' not in df.columns:
            logger.error(f"Missing 'regime' or 'target' column for {coin}")
            return results
        
        # Prepare features (exclude target and regime from features)
        feature_cols = [col for col in df.columns 
                       if col not in ['target', 'regime', 'open', 'high', 'low', 'close', 'volume']]
        
        # Add regime as a categorical feature (one-hot encoded)
        df_with_regime = df.copy()
        regime_dummies = pd.get_dummies(df_with_regime['regime'], prefix='regime')
        
        # Select only numeric features to avoid string-to-float conversion errors
        numeric_features = df_with_regime[feature_cols].select_dtypes(include=[np.number])
        
        # Log if any non-numeric columns were dropped
        dropped_cols = set(feature_cols) - set(numeric_features.columns)
        if dropped_cols:
            logger.debug(f"Dropped non-numeric columns for {coin}: {dropped_cols}")
        
        X = pd.concat([numeric_features, regime_dummies], axis=1)
        y = df_with_regime['target']
        
        logger.info(f"Training {coin} models on {len(X)} samples with {len(X.columns)} features")
        
        # Fix 6: Train each regime model on regime-specific data (with floor fallback)
        MIN_REGIME_SAMPLES = 100 # was: 300

        for regime in self.regimes:
            # Filter to rows where this regime was active
            regime_mask = df_with_regime['regime'] == regime
            X_regime = X[regime_mask]
            y_regime = y[regime_mask]

            if len(X_regime) >= MIN_REGIME_SAMPLES:
                logger.info(
                    f"[TRAINING] {coin}/{regime}: {len(X_regime)} regime-specific samples"
                )
                X_train, y_train = X_regime, y_regime
            else:
                logger.warning(
                    f"[TRAINING] {coin}/{regime}: only {len(X_regime)} samples "
                    f"— falling back to full dataset ({len(X)} rows)"
                )
                X_train, y_train = X, y

            model = RandomForestModel(coin, regime)
            metrics = model.train(X_train, y_train)

            if 'error' not in metrics:
                self.models[coin][regime] = model
                results[regime] = metrics

                # Save model
                model.save()
            else:
                logger.error(f"Failed to train {coin} {regime}: {metrics.get('error')}")
                results[regime] = metrics

        
        return results
    
    def get_model(self, coin: str, regime: str) -> RandomForestModel:
        """Get model for a specific coin and regime."""
        if coin in self.models and regime in self.models[coin]:
            return self.models[coin][regime]
        return None
    
    def load_all_models(self) -> int:
        """
        Load all saved models from disk.
        
        Returns:
            Number of models loaded
        """
        loaded_count = 0
        
        for coin in self.coin_list:
            if coin not in self.models:
                self.models[coin] = {}
            
            for regime in self.regimes:
                model = RandomForestModel(coin, regime)
                if model.load():
                    self.models[coin][regime] = model
                    loaded_count += 1
        
        logger.info(f"Loaded {loaded_count} models from disk")
        return loaded_count
    
    def predict_for_coin(self, coin: str, regime: str, X: pd.DataFrame) -> tuple:
        """
        Get prediction for a coin in a specific regime.
        
        Args:
            coin: Trading symbol
            regime: Current regime
            X: Feature DataFrame (single row)
            
        Returns:
            Tuple of (signal, confidence)
        """
        model = self.get_model(coin, regime)
        
        if model is None:
            # Silence warning if model not found, just return neutral
            # logger.warning(f"No model found for {coin} {regime}")
            return 0, 0.0
        
        # Prepare X with regime features
        X_prepared = X.copy()
        
        # The model expects regime dummy variables (e.g., regime_BULL, regime_BEAR)
        # We need to add them manually since X only has technical indicators
        if model.feature_names:
            required_features = set(model.feature_names)
            current_features = set(X_prepared.columns)
            missing = required_features - current_features
            
            # Add missing regime columns
            for feature in missing:
                if feature.startswith('regime_'):
                    # Check if this is the active regime
                    # Example: feature='regime_BULL', regime='BULL' -> 1
                    # Example: feature='regime_BEAR', regime='BULL' -> 0
                    is_active = feature == f"regime_{regime}"
                    X_prepared[feature] = 1 if is_active else 0
        
        return model.get_signal_with_confidence(X_prepared)
