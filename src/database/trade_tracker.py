import sqlite3
import pandas as pd
from datetime import datetime
import logging
import os
from typing import Optional, Dict

logger = logging.getLogger(__name__)

class TradeTracker:
    def __init__(self, db_path: str = "data/trades.db"):
        self.db_path = db_path
        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        entry_price REAL NOT NULL,
                        quantity REAL NOT NULL,
                        leverage INTEGER,
                        
                        -- Strategy Context (AI Data)
                        regime TEXT,
                        confidence REAL,
                        strategy_name TEXT,
                        
                        -- Risk Management
                        stop_loss_price REAL,
                        take_profit_price REAL,
                        
                        -- Trade Info
                        realized_pnl REAL,
                        order_id TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        
                        -- Status
                        is_backfilled BOOLEAN DEFAULT 0
                    )
                """)
                
                # Migration: Check if realized_pnl exists, if not add it
                cursor.execute("PRAGMA table_info(trades)")
                columns = [info[1] for info in cursor.fetchall()]
                if 'realized_pnl' not in columns:
                    logger.info("Migrating database: Adding realized_pnl column")
                    cursor.execute("ALTER TABLE trades ADD COLUMN realized_pnl REAL")
                
                conn.commit()
                # logger.info("Trade tracker database initialized")
        except Exception as e:
            logger.error(f"Failed to init database: {e}")

    def log_trade(self, trade_data: Dict):
        """
        Log a new trade with full AI context.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO trades (
                        symbol, side, entry_price, quantity, leverage,
                        regime, confidence, strategy_name,
                        stop_loss_price, take_profit_price,
                        realized_pnl,
                        order_id, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_data.get('symbol'),
                    trade_data.get('side'),
                    trade_data.get('entry_price'),
                    trade_data.get('quantity'),
                    trade_data.get('leverage'),
                    trade_data.get('regime'),
                    trade_data.get('confidence'),
                    trade_data.get('strategy_name'),
                    trade_data.get('stop_loss_price'),
                    trade_data.get('take_profit_price'),
                    trade_data.get('realized_pnl', 0.0),
                    trade_data.get('order_id'),
                    datetime.now()
                ))
                conn.commit()
                # logger.info(f"Logged trade for {trade_data.get('symbol')} to DB")
        except Exception as e:
            logger.error(f"Failed to log trade to DB: {e}")

    def get_all_trades(self, limit: int = 50) -> pd.DataFrame:
        """Get recent trades as DataFrame."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                return pd.read_sql_query(
                    "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?", 
                    conn, 
                    params=(limit,)
                )
        except Exception as e:
            logger.error(f"Failed to fetch trades: {e}")
            return pd.DataFrame()
            
    def backfill_trade(self, trade_data: Dict):
        """
        Backfill a past trade (AI columns will be None).
        Avoids duplicates based on order_id.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check duplicate
                cursor.execute("SELECT 1 FROM trades WHERE order_id = ?", (trade_data.get('order_id'),))
                if cursor.fetchone():
                    return  # Skip exists
                
                cursor.execute("""
                    INSERT INTO trades (
                        symbol, side, entry_price, quantity, leverage,
                        order_id, timestamp, is_backfilled,
                        regime, confidence, strategy_name, realized_pnl
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, 'Unknown', 0.0, 'Backfilled', ?)
                """, (
                    trade_data.get('symbol'),
                    trade_data.get('side'),
                    trade_data.get('entry_price'),
                    trade_data.get('quantity'),
                    trade_data.get('leverage'),
                    trade_data.get('order_id'),
                    trade_data.get('timestamp'),
                    trade_data.get('realized_pnl', 0.0)
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to backfill trade: {e}")
