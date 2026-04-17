"""
Logging configuration.
All logging settings loaded from configuration.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime

from src.config import config


def setup_logging():
    """Configure logging based on config settings."""
    # Get logging configuration
    log_level = config.get('LOG_LEVEL', 'INFO')
    log_dir = config.log_dir
    
    # Create log directory
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log filename with timestamp
    log_filename = log_dir / f"trading_bot_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Capture everything at root level
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler - only show WARNING and above for most modules
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)  # Reduced verbosity for console
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation - log everything
    file_handler = RotatingFileHandler(
        log_filename,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)  # Log everything to file
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Set specific loggers to appropriate levels for console
    # Main bot logger - show INFO and above on console
    main_logger = logging.getLogger('__main__')
    main_logger.setLevel(logging.INFO)
    
    # Submodule loggers - only show WARNING and above on console
    for module in ['src.data', 'src.features', 'src.regime', 'src.ml', 
                   'src.trading_engine', 'src.strategies', 'src.exchange']:
        logging.getLogger(module).setLevel(logging.WARNING)
    
    logger.info(f"Logging initialized: level={log_level}, file={log_filename}")
    
    return logger
