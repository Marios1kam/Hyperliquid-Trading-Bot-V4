#!/usr/bin/env python3
"""
Initial data collection script.
Fetches historical OHLCV data for all configured coins.

Run this before first training or bot startup:
    python3 collect_initial_data.py
"""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import config
from src.utils.logging_setup import setup_logging
from src.data.binance_collector import BinanceCollector
from src.data.data_manager import DataManager


def main():
    """Collect initial data for all configured coins."""
    # Setup logging
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("Initial Data Collection")
    logger.info("=" * 60)
    
    # Initialize components
    collector = BinanceCollector()
    data_manager = DataManager()
    
    # Get coin list from config
    coin_list = config.coin_list
    lookback_periods = config.get('LOOKBACK_PERIODS', 200)
    
    logger.info(f"Collecting data for {len(coin_list)} coins")
    logger.info(f"Lookback periods: {lookback_periods}")
    logger.info(f"Timeframe: {config.get('TIMEFRAME_MINUTES', 1)} minutes")
    
    # Collect data for each coin
    successful = 0
    failed = []
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 5
    
    for i, coin in enumerate(coin_list, 1):
        logger.info(f"[{i}/{len(coin_list)}] Fetching data for {coin}...")
        
        try:
            # Check if data already exists
            existing_data = data_manager.load_data(coin)
            
            if not existing_data.empty:
                logger.info(f"  - Found existing data: {len(existing_data)} rows")
                logger.info(f"  - Latest: {existing_data.index[-1]}")
                
                # Update with latest data
                logger.info(f"  - Updating with latest candles...")
                updated_data = collector.update_data(coin, existing_data)
                if len(updated_data) > len(existing_data):
                    is_valid, issues = data_manager.validate_data(updated_data)
                    if not is_valid:
                        logger.warning(f"  - Validation issues for {coin}: {issues}")
                    
                    data_manager.save_data(coin, updated_data)
                    logger.info(f"  - Updated to {len(updated_data)} rows")
                else:
                    logger.info("  - No new data available")
            else:
                # Fetch fresh data (historical)
                start_date = datetime(2023, 1, 1)
                logger.info(f"  - No existing data, fetching all data since {start_date.strftime('%Y-%m-%d')}...")
                df = collector.fetch_all_data_since(coin, since_date=start_date)
                
                if df.empty:
                    logger.warning(f"  - Failed to fetch data for {coin}")
                    failed.append(coin)
                    continue
                
                # Save data
                is_valid, issues = data_manager.validate_data(df)
                if not is_valid:
                    logger.warning(f"  - Validation issues for {coin}: {issues}")
                    
                data_manager.save_data(coin, df)
                logger.info(f"  - Saved {len(df)} rows")
                logger.info(f"  - Date range: {df.index[0]} to {df.index[-1]}")
            
            successful += 1
            consecutive_failures = 0
            
        except Exception as e:
            logger.error(f"  - Error fetching data for {coin}: {e}")
            failed.append(coin)
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logger.critical(f"Aborting: {MAX_CONSECUTIVE_FAILURES} consecutive failures. Possible network or API issue.")
                break
    
    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Data Collection Summary")
    logger.info("=" * 60)
    logger.info(f"Successful: {successful}/{len(coin_list)}")
    
    if failed:
        logger.warning(f"Failed: {len(failed)} coins")
        logger.warning(f"Failed coins: {', '.join(failed)}")
    else:
        logger.info("All coins collected successfully!")
    
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Train models: python3 main.py --train-only")
    logger.info("2. Run bot: python3 main.py")
    logger.info("=" * 60)
    
    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
