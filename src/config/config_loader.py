"""
Centralized configuration loader.
All configuration parameters are loaded from YAML and environment variables.
No hardcoded values allowed in the codebase.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv


class ConfigLoader:
    """Centralized configuration management."""
    
    _instance = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._config:
            self.load_config()
    
    def load_config(self, config_path: Optional[str] = None):
        """Load configuration from YAML and environment variables."""
        # Determine project root
        project_root = Path(__file__).parent.parent.parent
        
        # Load environment variables from config/.env
        env_path = project_root / "config" / ".env"
        load_dotenv(env_path)
        
        # Determine config file path
        if config_path is None:
            # Default to config/settings.yaml relative to project root
            config_path = project_root / "config" / "settings.yaml"
        else:
            config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load YAML configuration
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
        
        # Override with environment variables where applicable
        self._override_from_env()
        
        # Validate configuration
        self._validate_config()
    
    def _override_from_env(self):
        """Override config values with environment variables."""
        env_mappings = {
            'APP_ENV': 'APP_ENV',
            'USE_TESTNET': 'USE_TESTNET',
            'ENABLE_LIVE_TRADING': 'ENABLE_LIVE_TRADING',
            'LOG_LEVEL': 'LOG_LEVEL',
            'HYPERLIQUID_RPC_URL': 'HYPERLIQUID_RPC_URL',
        }
        
        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert string to appropriate type
                if value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                self._config[config_key] = value
    
    def _validate_config(self):
        """Validate that all required configuration parameters are present."""
        required_keys = [
            'COIN_LIST', 'MAX_POSITIONS', 'MIN_CONFIDENCE',
            # 'LEVERAGE' removed - now using Hyperliquid default leverage
            'STOP_LOSS_PERCENT', 'TAKE_PROFIT_PERCENT',
            'DATA_DIR', 'MODEL_DIR', 'LOG_DIR',
            'MA_PERIODS', 'RSI_PERIOD', 'ATR_PERIOD',
            'REGIME', 'SIDEWAYS', 'ML',
            'LOOK_AHEAD', 'PREDICTION_THRESHOLD'
        ]
        
        missing_keys = [key for key in required_keys if key not in self._config]
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        
        return value
    
    def get_all(self) -> Dict[str, Any]:
        """Get entire configuration dictionary."""
        return self._config.copy()
    
    # Convenience methods for commonly accessed config
    
    @property
    def coin_list(self) -> List[str]:
        """Get list of trading coins."""
        return self.get('COIN_LIST', [])
    
    @property
    def max_positions(self) -> int:
        """Get maximum number of concurrent positions."""
        return self.get('MAX_POSITIONS', 5)
    
    @property
    def min_confidence(self) -> float:
        """Get minimum ML confidence threshold."""
        return self.get('MIN_CONFIDENCE', 0.6)
    
    @property
    def leverage(self) -> float:
        """Get position leverage."""
        return self.get('LEVERAGE', 1.0)
    
    @property
    def stop_loss_percent(self) -> float:
        """Get stop-loss percentage."""
        return self.get('STOP_LOSS_PERCENT', 2.0)
    
    @property
    def take_profit_percent(self) -> float:
        """Get take-profit percentage."""
        return self.get('TAKE_PROFIT_PERCENT', 4.0)
    
    @property
    def min_hold_minutes(self) -> int:
        """Get minimum hold time in minutes."""
        return self.get('MIN_HOLD_MINUTES', 30)
    
    @property
    def position_size_percent(self) -> float:
        """Get base position size percentage."""
        return self.get('POSITION_SIZE_PERCENT', 2.0)
    
    @property
    def data_dir(self) -> Path:
        """Get data directory path."""
        return Path(self.get('DATA_DIR', 'data'))
    
    @property
    def model_dir(self) -> Path:
        """Get model directory path."""
        return Path(self.get('MODEL_DIR', 'models'))
    
    @property
    def log_dir(self) -> Path:
        """Get log directory path."""
        return Path(self.get('LOG_DIR', 'logs'))
    
    @property
    def use_testnet(self) -> bool:
        """Check if testnet mode is enabled."""
        return self.get('USE_TESTNET', True)
    
    @property
    def enable_live_trading(self) -> bool:
        """Check if live trading is enabled."""
        return self.get('ENABLE_LIVE_TRADING', False)
    
    @property
    def hyperliquid_rpc_url(self) -> str:
        """Get Hyperliquid RPC URL."""
        return self.get('HYPERLIQUID_RPC_URL', 'https://api.hyperliquid.xyz')
    
    @property
    def retrain_interval_days(self) -> int:
        """Get model retraining interval in days."""
        return self.get('RETRAIN_INTERVAL_DAYS', 7)
    
    @property
    def look_ahead(self) -> int:
        """Get prediction look-ahead period."""
        return self.get('LOOK_AHEAD', 180)
    
    @property
    def prediction_threshold(self) -> float:
        """Get prediction threshold for labeling."""
        return self.get('PREDICTION_THRESHOLD', 0.015)


# Global config instance
config = ConfigLoader()


def get_config() -> ConfigLoader:
    """Get global configuration instance."""
    return config
