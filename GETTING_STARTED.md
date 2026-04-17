# Getting Started with Hyperliquid ML Trading Bot

This guide will walk you through setting up and running the trading bot for the first time.

## Prerequisites

- macOS (tested on your system)
- Python 3.7+ (you have Python 3.13.7)
- Internet connection for data collection
- Hyperliquid account (for live trading)

## Step-by-Step Setup

### Step 1: Install Dependencies

Run the setup script to install all required Python packages:

```bash
./setup.sh
```

This will:

- Install all Python dependencies from `requirements.txt`
- Create necessary directories (`data/`, `models/`, `logs/`)
- Copy `.env.example` to `.env`
- **Automatically collect initial data for all coins** ✨

**Note**: If you get a permission error, run: `chmod +x setup.sh` first.

The setup script now includes automatic data collection, so you're ready to go!

### Step 2: Configure API Credentials (Optional for Testing)

Edit `config/.env` and add your Hyperliquid API credentials:

```bash
nano config/.env
```

Add:

```
HYPERLIQUID_PRIVATE_KEY=your_private_key_here
HYPERLIQUID_WALLET_ADDRESS=your_wallet_address_here
APP_ENV=testnet
```

**For initial testing, you can skip this step** - the bot will work in data collection and training mode without API keys.

### Step 3: Review Configuration

Open `config/settings.yaml` and review the settings. Key parameters:

```yaml
# Start with a smaller coin list for testing
COIN_LIST:
  - BTC
  - ETH
  - SOL

# Keep these safe for testing
ENABLE_LIVE_TRADING: false
USE_TESTNET: true
MAX_POSITIONS: 3
```

### Step 4: Verify Installation

Run the quick start verification:

```bash
python3 quickstart.py
```

This will test:

- Configuration loading
- Data collection from Binance
- Feature engineering
- Regime detection
- Model directory setup

Expected output:

```
[1/5] Testing configuration...
  [OK] Configuration loaded successfully
[2/5] Testing data collection...
  [OK] Data collection working
[3/5] Testing feature engineering...
  [OK] Feature engineering working
[4/5] Testing regime detection...
  [OK] Regime detection working
[5/5] Checking model directory...
  [OK] Model directory ready
```

### Step 5: Verify Data Collection

Check that data was collected:

```bash
ls -lh data/
```

You should see CSV files for each coin:

```
BTC_1m.csv
ETH_1m.csv
SOL_1m.csv
...
```

**Manual Data Collection** (if needed):

If you need to manually collect or update data:

```bash
python3 collect_initial_data.py
```

This will:

- Fetch historical OHLCV data from Binance for all configured coins
- Save data to `data/` directory as CSV files
- Check for existing data and update if needed
- Take 2-5 minutes depending on number of coins

### Step 6: Train Initial Models

Train ML models for all configured coins:

```bash
python3 main.py --config config/settings.yaml --train-only
```

This will:

1. Load historical data from `data/` directory
2. Compute 50+ technical indicators
3. Detect market regimes (BULL/BEAR/SIDEWAYS/CHOPPY)
4. Generate training labels based on future returns
5. Train Random Forest models for each coin
6. Save trained models to `models/` directory

**Expected time**: 5-15 minutes depending on number of coins.

### Step 7: Run the Bot (Dry Run)

Start the bot in monitoring mode (no trading):

```bash
python3 main.py --config config/settings.yaml
```

The bot will:

- Load trained models
- **Automatically fetch latest data every iteration** ✨
- Update existing data with new candles
- Compute features and detect regimes
- Generate trading signals with confidence scores
- Log everything to console and `logs/`

**Press Ctrl+C to stop gracefully.**

## Understanding Data Updates

### Automatic Data Updates ✨

The bot **automatically updates data** during operation:

1. **Every Trading Iteration**:
   - Loads existing data from CSV
   - Fetches new candles since last update
   - Appends new data to existing data
   - Saves updated data back to CSV

2. **No Manual Intervention Needed**:
   - Data stays fresh automatically
   - No stale data issues
   - Continuous updates as long as bot runs

3. **How It Works** (in `main.py`):

   ```python
   def process_coin(self, coin: str):
       # Load existing data
       df = self.data_manager.load_data(coin)

       if df.empty:
           # First time - fetch full history
           df = self.data_collector.fetch_ohlcv(coin)
       else:
           # Update with latest candles
           df = self.data_collector.update_data(coin, df)

       # Save updated data
       self.data_manager.save_data(coin, df)
   ```

4. **Update Frequency**:
   - Controlled by `TIMEFRAME_MINUTES` in config
   - Default: 1 minute (updates every minute)
   - Configurable: 1m, 5m, 15m, 1h, etc.

### Data Collection Summary

| When            | What Happens                          | Command                           |
| --------------- | ------------------------------------- | --------------------------------- |
| **Setup**       | Initial data collection for all coins | `./setup.sh` (automatic)          |
| **Manual**      | Collect/update data anytime           | `python3 collect_initial_data.py` |
| **Bot Running** | Automatic updates every iteration     | Built-in (no action needed)       |

## Understanding the Output

### Regime Detection

You'll see output like:

```
BTC: regime=BULL, signal=1, confidence=0.752, price=45234.50
ETH: regime=SIDEWAYS_QUIET, signal=0, confidence=0.423, price=2345.67
SOL: regime=BEAR, signal=-1, confidence=0.681, price=98.76
```

- **regime**: Current market state (BULL/BEAR/SIDEWAYS_QUIET/SIDEWAYS_VOLATILE/CHOPPY)
- **signal**: Trading signal (1=LONG, 0=NEUTRAL, -1=SHORT)
- **confidence**: ML model confidence (0-1)
- **price**: Current price (automatically updated)

### Signal Filtering

Only signals with `confidence >= MIN_CONFIDENCE` (default 0.6) are considered valid.

## Next Steps

### 1. Backtest (Recommended)

Before live trading, backtest the strategy:

```bash
python3 main.py --config config/settings.yaml --backtest
```

### 2. Paper Trading on Testnet

Once comfortable, enable testnet trading:

1. Edit `config/settings.yaml`:

   ```yaml
   ENABLE_LIVE_TRADING: true
   USE_TESTNET: true
   ```

2. Add testnet API credentials to `config/.env`

3. Run:
   ```bash
   python3 main.py
   ```

### 3. Live Trading (After Validation)

**⚠️ Only after thorough testing on testnet:**

1. Edit `config/settings.yaml`:

   ```yaml
   ENABLE_LIVE_TRADING: true
   USE_TESTNET: false
   ```

2. Add mainnet API credentials to `config/.env`

3. Start with small position sizes:

   ```yaml
   POSITION_SIZE_PERCENT: 1.0
   MAX_POSITIONS: 2
   ```

4. Run:
   ```bash
   python3 main.py
   ```

## Customization

### Adjust Coin Universe

Edit `config/settings.yaml`:

```yaml
COIN_LIST:
  - BTC
  - ETH
  - SOL
  # Add more coins as needed
```

**After adding coins**: Run `python3 collect_initial_data.py` to fetch data for new coins.

### Tune Risk Parameters

```yaml
LEVERAGE: 1.0 # Start with 1x leverage
STOP_LOSS_PERCENT: 2.0 # 2% stop-loss
TAKE_PROFIT_PERCENT: 3.0 # 3% take-profit
MAX_POSITIONS: 3 # Max 3 concurrent positions
```

### Adjust ML Confidence

```yaml
MIN_CONFIDENCE: 0.7 # Increase for more selective trading
```

### Change Timeframe

```yaml
TIMEFRAME_MINUTES: 5 # Use 5-minute candles instead of 1-minute
```

**Note**: After changing timeframe, re-collect data: `python3 collect_initial_data.py`

### Modify Feature Engineering

```yaml
MA_PERIODS: [10, 20, 50, 100] # Different MA periods
RSI_PERIOD: 21 # Longer RSI period
```

### Tune Regime Detection

```yaml
REGIME:
  BULL:
    UPTREND_THRESHOLD: 0.002 # Stricter bull detection
  BEAR:
    DOWNTREND_THRESHOLD: -0.002 # Stricter bear detection

SIDEWAYS:
  VOL_QUIET_MAX: 0.003 # Tighter quiet threshold
```

## Monitoring

### Logs

Check logs in `logs/` directory:

```bash
tail -f logs/trading_bot_*.log
```

### Data Status

Check data files:

```bash
# List all data files
ls -lh data/

# View latest data for BTC
tail -20 data/BTC_1m.csv

# Count rows in data file
wc -l data/BTC_1m.csv
```

### Model Performance

After running for a while, check model summary:

```python
from src.ml.trainer import ModelTrainer
trainer = ModelTrainer()
summary = trainer.get_training_summary()
print(summary)
```

### Regime Statistics

Check regime distribution for a coin:

```python
from src.regime.regime_classifier import RegimeClassifier
from src.data.data_manager import DataManager

dm = DataManager()
rc = RegimeClassifier()

df = dm.load_data('BTC')
df = rc.classify_regimes(df)
stats = rc.get_regime_statistics(df)
print(stats)
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'ccxt'"

Run: `pip3 install -r requirements.txt`

### "No data available for coin"

1. Check internet connection and Binance API status
2. Manually collect data: `python3 collect_initial_data.py`
3. Check data directory: `ls -lh data/`

Test data collection:

```python
from src.data.binance_collector import BinanceCollector
collector = BinanceCollector()
df = collector.fetch_ohlcv('BTC', limit=100)
print(df)
```

### "Model not trained"

Run training first: `python3 main.py --train-only`

### Data appears stale

The bot automatically updates data, but you can manually refresh:

```bash
python3 collect_initial_data.py
```

### High memory usage

Reduce number of coins in `COIN_LIST` or increase `RETRAIN_INTERVAL_DAYS`.

### No trading signals

- Check `MIN_CONFIDENCE` - may be too high
- Verify models are trained
- Check regime detection - may be in CHOPPY regime (no trading)
- Verify data is being updated (check logs)

## Safety Checklist

Before live trading:

- [ ] Tested on testnet for at least 1 week
- [ ] Reviewed all signals and understood regime detection
- [ ] Verified data is updating automatically (check logs)
- [ ] Set appropriate `STOP_LOSS_PERCENT` and `TAKE_PROFIT_PERCENT`
- [ ] Started with small `POSITION_SIZE_PERCENT` (1-2%)
- [ ] Limited `MAX_POSITIONS` (2-3)
- [ ] Set `DAILY_LOSS_LIMIT_PERCENT`
- [ ] Enabled `CLOSE_POSITIONS_ON_SHUTDOWN`
- [ ] Monitoring logs regularly
- [ ] Have emergency stop plan

## Support

- **Documentation**: See `README.md` and `PROJECT_SUMMARY.md`
- **Configuration**: All parameters documented in `config/settings.yaml`
- **Code**: Well-commented source in `src/`
- **Data Collection**: `collect_initial_data.py` for manual updates

## Important Notes

1. **This is experimental software** - use at your own risk
2. **Start small** - test thoroughly before scaling
3. **Monitor closely** - especially in first weeks
4. **Data updates automatically** - no manual intervention needed during bot operation
5. **Cryptocurrency trading is risky** - never risk more than you can afford to lose
6. **Past performance ≠ future results** - ML models can fail

---

**Happy Trading! 🚀**

Remember: The best trade is often no trade. The bot respects this with confidence filtering and regime-aware risk management.

**Data is automatically kept fresh** - the bot updates pricing every iteration, so you never have stale data!
