"""
Data manager for OHLCV storage and retrieval.
Handles CSV persistence with configurable paths.
"""

import os
import threading
import pandas as pd
from pathlib import Path
from typing import Optional, List
import logging

from src.config import config


logger = logging.getLogger(__name__)


class DataManager:
    """Manage OHLCV data storage and retrieval."""
    
    def __init__(self):
        """Initialize data manager."""
        self.data_dir = config.data_dir
        self.price_file_template = config.get('PRICE_FILE_TEMPLATE', '{coin}_1m.csv')
        
        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Per-symbol file locks to prevent concurrent write corruption
        self._file_locks = {}
        self._locks_mutex = threading.Lock()
        
        logger.info(f"Initialized data manager with directory: {self.data_dir}")
    
    def _get_lock(self, symbol: str) -> threading.Lock:
        """Get or create a per-symbol threading lock."""
        with self._locks_mutex:
            if symbol not in self._file_locks:
                self._file_locks[symbol] = threading.Lock()
            return self._file_locks[symbol]

    def get_file_path(self, symbol: str) -> Path:
        """
        Get file path for a symbol's data.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Path to CSV file
        """
        filename = self.price_file_template.format(coin=symbol)
        return self.data_dir / filename
    
    def save_data(self, symbol: str, df: pd.DataFrame) -> bool:
        """
        Save OHLCV data to CSV.
        
        Args:
            symbol: Trading symbol
            df: OHLCV DataFrame
            
        Returns:
            True if successful
        """
        with self._get_lock(symbol):
            try:
                file_path = self.get_file_path(symbol)
                tmp_path = str(file_path) + '.tmp'
                df.to_csv(tmp_path)
                os.replace(tmp_path, str(file_path))
                logger.info(f"Saved {len(df)} rows for {symbol} to {file_path}")
                return True
            except Exception as e:
                logger.error(f"Error saving data for {symbol}: {e}")
                return False
    
    def load_data(self, symbol: str) -> pd.DataFrame:
        """
        Load OHLCV data from CSV.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            OHLCV DataFrame or empty DataFrame if not found
        """
        with self._get_lock(symbol):
            file_path = self.get_file_path(symbol)
            
            if not file_path.exists():
                logger.warning(f"No data file found for {symbol} at {file_path}")
                return pd.DataFrame()
            
            try:
                df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
                logger.info(f"Loaded {len(df)} rows for {symbol}")
                return df
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {e}")
                raise ValueError(f"Corrupted or unreadable data file for {symbol}: {e}") from e
    
    def data_exists(self, symbol: str) -> bool:
        """
        Check if data file exists for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if data file exists
        """
        return self.get_file_path(symbol).exists()
    
    def get_latest_timestamp(self, symbol: str) -> Optional[pd.Timestamp]:
        """
        Get the latest timestamp for a symbol's data.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Latest timestamp or None
        """
        df = self.load_data(symbol)
        if not df.empty:
            return df.index[-1]
        return None
    
    def validate_data(self, df: pd.DataFrame) -> tuple:
        """
        Validate OHLCV data for gaps and anomalies.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Tuple of (is_valid, issues_list)
        """
        issues = []
        
        if df.empty:
            return False, ["DataFrame is empty"]
        
        # Check for required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues.append(f"Missing columns: {missing_columns}")
        
        # Check for NaN values
        nan_counts = df[required_columns].isna().sum()
        if nan_counts.any():
            issues.append(f"NaN values found: {nan_counts[nan_counts > 0].to_dict()}")
        
        # Check for negative values
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns and (df[col] < 0).any():
                issues.append(f"Negative values in {col}")
        
        # Check OHLC relationships
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            invalid_high = df['high'] < df[['open', 'close']].max(axis=1)
            invalid_low = df['low'] > df[['open', 'close']].min(axis=1)
            
            if invalid_high.any():
                issues.append(f"Invalid high values: {invalid_high.sum()} rows")
            if invalid_low.any():
                issues.append(f"Invalid low values: {invalid_low.sum()} rows")
        
        # Check for time gaps (assuming 1-minute data)
        if len(df) > 1:
            time_diffs = df.index.to_series().diff()
            expected_diff = pd.Timedelta(minutes=config.get('TIMEFRAME_MINUTES', 1))
            gaps = time_diffs[time_diffs > expected_diff * 1.5]
            
            if len(gaps) > 0:
                issues.append(f"Time gaps detected: {len(gaps)} gaps")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def get_all_symbols(self) -> List[str]:
        """
        Get list of all symbols with data files.
        
        Returns:
            List of symbol names
        """
        symbols = []
        
        for file_path in self.data_dir.glob('*.csv'):
            # Extract symbol from filename using template
            filename = file_path.stem
            # Simple extraction: remove suffix like '_1m'
            symbol = filename.split('_')[0]
            symbols.append(symbol)
        
        return sorted(symbols)
    
    def clean_old_data(self, symbol: str, keep_days: int = 30) -> bool:
        """
        Remove old data beyond specified days.
        
        Args:
            symbol: Trading symbol
            keep_days: Number of days to keep
            
        Returns:
            True if successful
        """
        try:
            df = self.load_data(symbol)
            if df.empty:
                return True
            
            cutoff_date = pd.Timestamp.utcnow().replace(tzinfo=None) - pd.Timedelta(days=keep_days)
            df_cleaned = df[df.index >= cutoff_date]
            
            self.save_data(symbol, df_cleaned)
            logger.info(f"Cleaned old data for {symbol}: kept {len(df_cleaned)}/{len(df)} rows")
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning data for {symbol}: {e}")
            return False
