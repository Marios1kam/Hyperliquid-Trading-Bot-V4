# Data Collection & Updates - Complete Guide

## Overview

The bot has a **comprehensive data management system** that ensures you always have fresh, up-to-date pricing data without manual intervention.

---

## ✅ What Was Added

### 1. **Initial Data Collection Script** ✨ NEW

**File**: `collect_initial_data.py`

**Purpose**: Fetch historical OHLCV data for all configured coins before first use.

**Usage**:

```bash
python3 collect_initial_data.py
```

**What It Does**:

- Fetches historical data from Binance for all coins in `COIN_LIST`
- Saves data to `data/` directory as CSV files (e.g., `BTC_1m.csv`)
- Checks for existing data and updates if found
- Provides detailed progress and summary
- Automatically called by `setup.sh`

**Output Example**:

```
[1/32] Fetching data for BTC...
  - No existing data, fetching 200 periods...
  - Saved 200 rows
  - Date range: 2024-01-20 to 2024-01-26
[2/32] Fetching data for ETH...
  - Found existing data: 150 rows
  - Updating with latest candles...
  - Updated to 200 rows
...
Data Collection Summary
Successful: 32/32
All coins collected successfully!
```

### 2. **Automatic Data Updates** ✅ ALREADY IMPLEMENTED

**Location**: `main.py` → `process_coin()` method (lines 160-171)

**How It Works**:

```python
def process_coin(self, coin: str):
    # Load existing data
    df = self.data_manager.load_data(coin)

    if df.empty:
        # First time - fetch full history
        df = self.data_collector.fetch_ohlcv(coin)
        self.data_manager.save_data(coin, df)
    else:
        # Update with latest candles
        df = self.data_collector.update_data(coin, df)
        self.data_manager.save_data(coin, df)
```

**Update Mechanism** (`src/data/binance_collector.py` lines 150-182):

```python
def update_data(self, symbol: str, existing_df: pd.DataFrame):
    # Get last timestamp from existing data
    last_timestamp = existing_df.index[-1]

    # Fetch new data since last timestamp
    new_df = self.fetch_ohlcv(
        symbol,
        since=last_timestamp + timedelta(minutes=1)
    )

    # Concatenate and remove duplicates
    updated_df = pd.concat([existing_df, new_df])
    updated_df = updated_df[~updated_df.index.duplicated(keep='last')]
    updated_df.sort_index(inplace=True)

    return updated_df
```

### 3. **Updated Setup Script** ✨ NEW

**File**: `setup.sh` (updated)

**Changes**:

- Now includes automatic data collection as step 5/5
- Runs `collect_initial_data.py` automatically
- Provides clear feedback about data collection status

**New Output**:

```
[5/5] Collecting initial data for all coins...
This may take a few minutes...
[Data collection output...]

Setup Complete!
Note: Initial data has been collected for all coins.
The bot will automatically update data during operation.
```

### 4. **Updated Documentation** ✨ NEW

**Files Updated**:

- `GETTING_STARTED.md` - Complete rewrite with data collection section
- `setup.sh` - Includes data collection step

**New Sections**:

- "Step 5: Verify Data Collection"
- "Understanding Data Updates"
- "Data Collection Summary" table
- Troubleshooting for data issues

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    INITIAL SETUP                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ./setup.sh  OR  python3 collect_initial_data.py       │
│       │                                                 │
│       ▼                                                 │
│  ┌──────────────────────────────────┐                  │
│  │  Fetch Historical Data           │                  │
│  │  - Binance API (CCXT)            │                  │
│  │  - 200 periods (configurable)    │                  │
│  │  - All coins in COIN_LIST        │                  │
│  └──────────────────────────────────┘                  │
│       │                                                 │
│       ▼                                                 │
│  ┌──────────────────────────────────┐                  │
│  │  Save to CSV Files               │                  │
│  │  data/BTC_1m.csv                 │                  │
│  │  data/ETH_1m.csv                 │                  │
│  │  data/SOL_1m.csv                 │                  │
│  │  ...                              │                  │
│  └──────────────────────────────────┘                  │
│                                                         │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                  BOT OPERATION                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  python3 main.py                                        │
│       │                                                 │
│       ▼                                                 │
│  ┌──────────────────────────────────┐                  │
│  │  Trading Loop (Every Iteration)  │                  │
│  │  - Every TIMEFRAME_MINUTES       │                  │
│  │  - Default: 1 minute             │                  │
│  └──────────────────────────────────┘                  │
│       │                                                 │
│       ▼                                                 │
│  ┌──────────────────────────────────┐                  │
│  │  For Each Coin:                  │                  │
│  │  1. Load existing data from CSV  │                  │
│  │  2. Get last timestamp           │                  │
│  │  3. Fetch new candles since then │                  │
│  │  4. Append new data              │                  │
│  │  5. Save updated CSV             │                  │
│  └──────────────────────────────────┘                  │
│       │                                                 │
│       ▼                                                 │
│  ┌──────────────────────────────────┐                  │
│  │  Compute Features                │                  │
│  │  Detect Regime                   │                  │
│  │  Generate Signal                 │                  │
│  │  Execute Trade (if enabled)      │                  │
│  └──────────────────────────────────┘                  │
│       │                                                 │
│       └──────────┐                                      │
│                  │                                      │
│       ┌──────────┘                                      │
│       │                                                 │
│       ▼                                                 │
│  (Repeat every iteration)                              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## When Data Is Collected/Updated

| Event                 | What Happens                       | Frequency                 | Manual?                                 |
| --------------------- | ---------------------------------- | ------------------------- | --------------------------------------- |
| **Setup**             | Initial historical data collection | Once                      | No (automatic in `setup.sh`)            |
| **Manual Collection** | Collect/update all coin data       | On demand                 | Yes (`python3 collect_initial_data.py`) |
| **Bot Startup**       | Load existing data from CSV        | Once per run              | No                                      |
| **Trading Iteration** | Update data with latest candles    | Every `TIMEFRAME_MINUTES` | No (automatic)                          |
| **New Coin Added**    | Need to collect data for new coin  | Once per new coin         | Yes (run `collect_initial_data.py`)     |

---

## Configuration

All data collection settings in `config/settings.yaml`:

```yaml
# Data collection settings
TIMEFRAME_MINUTES: 1 # Candle timeframe (1m, 5m, 15m, etc.)
LOOKBACK_PERIODS: 200 # Number of historical candles to fetch
DATA_DIR: data # Directory for CSV storage
PRICE_FILE_TEMPLATE: "{coin}_1m.csv" # CSV filename template

# Binance API
BINANCE_API_URL: https://api.binance.com/api
```

---

## File Structure

```
hyperliquid_tradingbot_v1/
├── collect_initial_data.py   # ✨ NEW - Initial data collection script
├── data/                      # Data storage directory
│   ├── BTC_1m.csv            # Historical OHLCV for BTC
│   ├── ETH_1m.csv            # Historical OHLCV for ETH
│   ├── SOL_1m.csv            # Historical OHLCV for SOL
│   └── ...                    # One file per coin
├── src/
│   └── data/
│       ├── binance_collector.py  # Fetches data from Binance
│       └── data_manager.py       # Saves/loads CSV files
└── main.py                    # Automatic updates in trading loop
```

---

## Data Update Examples

### Example 1: First Run (No Existing Data)

```python
# main.py - process_coin('BTC')
df = self.data_manager.load_data('BTC')  # Returns empty DataFrame

if df.empty:
    # Fetch full history
    df = self.data_collector.fetch_ohlcv('BTC', limit=200)
    # Returns 200 candles
    self.data_manager.save_data('BTC', df)
    # Saves to data/BTC_1m.csv
```

### Example 2: Subsequent Runs (Existing Data)

```python
# main.py - process_coin('BTC')
df = self.data_manager.load_data('BTC')  # Returns 200 rows

# Update with latest candles
df = self.data_collector.update_data('BTC', df)
# Fetches new candles since last timestamp
# Appends to existing data
# Returns 201+ rows (depending on time elapsed)

self.data_manager.save_data('BTC', df)
# Saves updated data to data/BTC_1m.csv
```

### Example 3: Manual Update

```bash
# User runs manual collection
python3 collect_initial_data.py

# Output:
# [1/32] Fetching data for BTC...
#   - Found existing data: 250 rows
#   - Latest: 2024-01-26 14:30:00
#   - Updating with latest candles...
#   - Updated to 300 rows
```

---

## Verification

### Check Data Files

```bash
# List all data files
ls -lh data/

# Output:
# -rw-r--r--  1 user  staff   45K Jan 26 14:30 BTC_1m.csv
# -rw-r--r--  1 user  staff   42K Jan 26 14:30 ETH_1m.csv
# -rw-r--r--  1 user  staff   38K Jan 26 14:30 SOL_1m.csv
```

### Check Data Content

```bash
# View first few rows
head -5 data/BTC_1m.csv

# View last few rows (most recent data)
tail -5 data/BTC_1m.csv

# Count total rows
wc -l data/BTC_1m.csv
```

### Check Data in Python

```python
from src.data.data_manager import DataManager

dm = DataManager()
df = dm.load_data('BTC')

print(f"Total rows: {len(df)}")
print(f"Date range: {df.index[0]} to {df.index[-1]}")
print(f"Latest price: ${df['close'].iloc[-1]:.2f}")
```

---

## Troubleshooting

### Problem: No data files in `data/` directory

**Solution**:

```bash
python3 collect_initial_data.py
```

### Problem: Data appears stale

**Check**:

1. Is bot running? (Data updates automatically during operation)
2. Check logs for update messages:
   ```bash
   tail -f logs/trading_bot_*.log | grep "Updated"
   ```

**Manual Update**:

```bash
python3 collect_initial_data.py
```

### Problem: "No data available for coin"

**Causes**:

- Internet connection issue
- Binance API down
- Coin not available on Binance spot market

**Test**:

```python
from src.data.binance_collector import BinanceCollector

collector = BinanceCollector()
df = collector.fetch_ohlcv('BTC', limit=10)
print(df)
```

### Problem: Data collection fails for some coins

**Check**:

- Coin symbol correct? (should be base symbol like 'BTC', not 'BTC/USDT')
- Coin available on Binance spot?
- Rate limiting? (script includes delays)

**Logs**:

```bash
# Check logs for specific error messages
grep "Error fetching" logs/trading_bot_*.log
```

---

## Best Practices

### 1. **Initial Setup**

- Run `./setup.sh` - handles everything automatically
- Verify data collected: `ls -lh data/`

### 2. **Adding New Coins**

- Edit `config/settings.yaml` → add to `COIN_LIST`
- Run `python3 collect_initial_data.py` to fetch data for new coin
- Retrain models: `python3 main.py --train-only`

### 3. **Changing Timeframe**

- Edit `config/settings.yaml` → change `TIMEFRAME_MINUTES`
- Delete old data: `rm data/*.csv`
- Re-collect: `python3 collect_initial_data.py`
- Retrain models: `python3 main.py --train-only`

### 4. **Regular Operation**

- Just run `python3 main.py`
- Data updates automatically - no manual intervention needed
- Monitor logs to verify updates happening

### 5. **Maintenance**

- Data files grow over time (normal)
- Consider periodic cleanup of very old data (optional)
- Backup `data/` directory periodically

---

## Summary

### ✅ **What You Get**

1. **Automatic Initial Collection**
   - Runs during setup
   - Fetches historical data for all coins
   - Saves to CSV files

2. **Automatic Updates During Operation**
   - Every trading iteration
   - Fetches latest candles
   - Appends to existing data
   - No stale data ever

3. **Manual Collection Available**
   - `python3 collect_initial_data.py`
   - Use when adding new coins
   - Use to refresh data if needed

4. **Complete Transparency**
   - All data in readable CSV files
   - Easy to inspect and verify
   - Detailed logging of all updates

### 🎯 **Key Points**

- ✅ Data collection is **automatic** during setup
- ✅ Data updates are **automatic** during bot operation
- ✅ No manual intervention needed for normal operation
- ✅ Fresh data guaranteed - updates every iteration
- ✅ Manual collection available when needed
- ✅ All settings configurable in YAML

**You never have to worry about stale data!** 🚀

---

## Files Modified/Created

| File                            | Status      | Purpose                                           |
| ------------------------------- | ----------- | ------------------------------------------------- |
| `collect_initial_data.py`       | ✨ NEW      | Manual data collection script                     |
| `setup.sh`                      | ✅ UPDATED  | Now includes automatic data collection            |
| `GETTING_STARTED.md`            | ✅ UPDATED  | Complete rewrite with data collection info        |
| `main.py`                       | ✅ EXISTING | Already had automatic updates (no changes needed) |
| `src/data/binance_collector.py` | ✅ EXISTING | Already had `update_data()` method                |
| `src/data/data_manager.py`      | ✅ EXISTING | Already had save/load methods                     |

---

**Everything is now documented and automated!** The bot handles data collection and updates seamlessly.
