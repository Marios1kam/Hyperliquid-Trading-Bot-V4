"""
Model training pipeline.
Handles data preparation, target generation, and model training.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from pathlib import Path

from src.config import config
from src.data.data_manager import DataManager
from src.data.binance_collector import BinanceCollector
from src.features.feature_pipeline import FeaturePipeline
from src.regime.regime_classifier import RegimeClassifier
from src.ml.random_forest_models import RandomForestModelManager


logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Orchestrate the complete model training pipeline.
    
    Steps:
    1. Load/collect data
    2. Compute features
    3. Detect regimes
    4. Generate targets
    5. Train models per coin
    """
    
    def __init__(self):
        """Initialize trainer."""
        self.data_manager = DataManager()
        self.data_collector = BinanceCollector()
        self.feature_pipeline = FeaturePipeline()
        self.regime_classifier = RegimeClassifier()
        self.model_manager = RandomForestModelManager()
        
        # Get configuration
        self.look_ahead = config.look_ahead
        self.prediction_threshold = config.prediction_threshold
        
        logger.info(f"Initialized model trainer: look_ahead={self.look_ahead}, "
                   f"threshold={self.prediction_threshold}")
    
    def prepare_training_data(self, coin: str, update: bool = True) -> pd.DataFrame:
        """
        Prepare complete training dataset for a coin.
        
        Args:
            coin: Trading symbol
            update: Whether to update with latest data
            
        Returns:
            DataFrame with features, regime, and target
        """
        logger.info(f"Preparing training data for {coin}")
        
        # Load existing data
        df = self.data_manager.load_data(coin)
        
        # Update with latest data if requested
        if update:
            if df.empty:
                # Fetch initial data
                df = self.data_collector.fetch_ohlcv(coin)
            else:
                # Update with new candles
                df = self.data_collector.update_data(coin, df)
            
            # Save updated data
            self.data_manager.save_data(coin, df)
        
        if df.empty:
            logger.error(f"No data available for {coin}")
            return pd.DataFrame()
        
        # Compute features
        df = self.feature_pipeline.compute_features(df)
        
        if df.empty:
            logger.error(f"Feature computation failed for {coin}")
            return pd.DataFrame()
        
        # Detect regimes
        df = self.regime_classifier.classify_regimes(df)
        
        # Generate targets
        df = self.generate_targets(df)

        # Drop rows with NaN targets (last look_ahead rows have no future candle)
        df = df.dropna(subset=['target'])

        # Drop future_return now that target generation is complete
        df = df.drop(columns=['future_return'], errors='ignore')

        # Log label distribution
        counts = df['target'].value_counts()
        logger.info(
            f"[TRAINING LABELS] {coin}: "
            f"Long={counts.get(1, 0)} | "
            f"Short={counts.get(-1, 0)} | "
            f"Neutral={counts.get(0, 0)}"
        )

        logger.info(f"Prepared {len(df)} training samples for {coin}")

        return df
    
    def generate_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading targets based on future returns.
        
        Target labels:
        - 1: Long signal (future return > threshold)
        - 0: Neutral (future return within threshold)
        - -1: Short signal (future return < -threshold)
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with 'target' column added
        """
        df = df.copy()
        
        # Calculate future returns
        df['future_close'] = df['close'].shift(-self.look_ahead)
        df['future_return'] = (df['future_close'] - df['close']) / df['close']
        
        # Generate labels based on threshold
        df['target'] = 0  # Default neutral
        
        df.loc[df['future_return'] > self.prediction_threshold, 'target'] = 1  # Long
        df.loc[df['future_return'] < -self.prediction_threshold, 'target'] = -1  # Short
        
        # Log target distribution
        target_counts = df['target'].value_counts()
        logger.info(f"Target distribution: {target_counts.to_dict()}")
        
        # Drop future_close only — future_return is kept so prepare_training_data()
        # can filter out neutral rows before training
        df = df.drop(columns=['future_close'])
        
        return df
    
    def train_single_coin(self, coin: str, update_data: bool = True) -> Dict:
        """
        Train all models for a single coin.
        
        Args:
            coin: Trading symbol
            update_data: Whether to update data before training
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Training models for {coin}")
        
        # Prepare data
        df = self.prepare_training_data(coin, update=update_data)
        
        if df.empty:
            return {'error': 'no_data'}
        
        # Train models
        results = self.model_manager.train_coin_models(coin, df)
        
        logger.info(f"Completed training for {coin}: {len(results)} models trained")
        
        return results
    
    def train_all_coins(self, update_data: bool = True) -> Dict[str, Dict]:
        """
        Train models for all configured coins.
        
        Args:
            update_data: Whether to update data before training
            
        Returns:
            Dictionary mapping coin to training results
        """
        all_results = {}
        
        coin_list = config.coin_list
        logger.info(f"Training models for {len(coin_list)} coins")
        
        for i, coin in enumerate(coin_list, 1):
            logger.info(f"Processing {coin} ({i}/{len(coin_list)})")
            
            try:
                results = self.train_single_coin(coin, update_data=update_data)
                all_results[coin] = results
            except Exception as e:
                logger.error(f"Error training {coin}: {e}")
                all_results[coin] = {'error': str(e)}
        
        # Summary
        successful = sum(1 for r in all_results.values() if 'error' not in r)
        logger.info(f"Training complete: {successful}/{len(coin_list)} coins successful")
        
        return all_results
    
    def retrain_if_needed(self, coin: str, force: bool = False) -> bool:
        """
        Retrain models if they're outdated.
        
        Args:
            coin: Trading symbol
            force: Force retraining regardless of age
            
        Returns:
            True if retrained
        """
        if force:
            logger.info(f"Force retraining {coin}")
            self.train_single_coin(coin)
            return True
        
        # Check if models exist and their age
        retrain_interval_days = config.retrain_interval_days
        
        # For simplicity, check one model
        model = self.model_manager.get_model(coin, 'BULL')
        
        if model is None or not model.is_trained():
            logger.info(f"No trained model found for {coin}, training...")
            self.train_single_coin(coin)
            return True
        
        # Check model age
        if model.trained_at is not None:
            from datetime import datetime, timedelta
            age = datetime.now() - model.trained_at
            
            if age > timedelta(days=retrain_interval_days):
                logger.info(f"Model for {coin} is {age.days} days old, retraining...")
                self.train_single_coin(coin)
                return True
        
        logger.info(f"Model for {coin} is up to date")
        return False
    
    def get_training_summary(self) -> Dict:
        """
        Get summary of all trained models.
        
        Returns:
            Dictionary with training summary
        """
        summary = {
            'total_coins': len(config.coin_list),
            'trained_models': 0,
            'coins': {}
        }
        
        for coin in config.coin_list:
            coin_summary = {
                'regimes': {}
            }
            
            for regime in ['BULL', 'BEAR', 'SIDEWAYS_QUIET', 'SIDEWAYS_VOLATILE', 'CHOPPY']:
                model = self.model_manager.get_model(coin, regime)
                
                if model and model.is_trained():
                    coin_summary['regimes'][regime] = {
                        'trained': True,
                        'trained_at': str(model.trained_at),
                        'n_features': len(model.feature_names) if model.feature_names else 0
                    }
                    summary['trained_models'] += 1
                else:
                    coin_summary['regimes'][regime] = {'trained': False}
            
            summary['coins'][coin] = coin_summary
        
        return summary
