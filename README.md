# ML Trading Bot on Hyperliquid

A production-grade, self-improving ML trading bot for Hyperliquid perpetuals with regime detection, sideways market analysis, and full configuration externalization.

## Features

- **Multi-Regime Detection**: Automatically classifies markets into BULL, BEAR, SIDEWAYS_QUIET, SIDEWAYS_VOLATILE, and CHOPPY regimes
- **Per-Coin ML Models**: Random Forest models trained on each coin's full history with regime as a feature
- **Sideways Market Specialization**: Structural decomposition of neutral/sideways markets with dedicated strategies
- **Full Configuration**: All parameters externalized to YAML/environment variables - zero hardcoded values
- **Self-Learning**: Automatic model retraining on configurable schedules
- **24/7 Operation**: Designed for continuous trading with graceful shutdown
- **Testnet Support**: Full testnet mode for safe validation

## Quick Start

### 1. Installation

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configuration

Copy the environment template and add your API credentials:

```bash
cp config/.env.example config/.env
# Edit config/.env and add your Hyperliquid credentials
```

All trading parameters are in `config/settings.yaml`. Customize as needed:

- Coin universe (COIN_LIST)
- Risk parameters (leverage, stop-loss, take-profit)
- Feature engineering settings (MA periods, RSI, MACD, etc.)
- Regime detection thresholds
- ML hyperparameters per regime
- And much more...

### 3. Initial Training

Train models for all coins:

```bash
python main.py --config config/settings.yaml --train-only
```

### 4. Run the Bot

**Testnet mode (recommended first):**

```bash
# Ensure USE_TESTNET: true in config/settings.yaml
python main.py --config config/settings.yaml
```

**Live trading (after validation):**

```bash
# Set USE_TESTNET: false and ENABLE_LIVE_TRADING: true in config
python main.py --config config/settings.yaml
```

## Architecture

```
hyperliquid_tradingbot_v1/
├── main.py                    # Entry point
├── config/
│   ├── settings.yaml          # All tuneable parameters
│   └── .env                   # API credentials (not in git)
├── src/
│   ├── config/                # Configuration loader
│   ├── data/                  # Binance data collection
│   ├── features/              # Technical indicators & sideways features
│   ├── regime/                # Regime detection (trend + sideways)
│   ├── ml/                    # ML models and training
│   ├── strategies/            # Trading strategies
│   ├── risk/                  # Risk management
│   ├── exchange/              # Hyperliquid execution
│   ├── learning/              # Self-learning & metrics
│   ├── diagnostics/           # Failure analysis
│   └── utils/                 # Logging and utilities
├── data/                      # OHLCV CSV storage
├── models/                    # Trained model persistence
└── logs/                      # Log files
```

## Configuration

All behavior is controlled via `config/settings.yaml`. Key sections:

### Trading Parameters

- `COIN_LIST`: Universe of tradeable coins
- `MAX_POSITIONS`: Maximum concurrent positions
- `MIN_CONFIDENCE`: Minimum ML confidence for entry
- `POSITION_SIZE_PERCENT`: Base position size

### Risk Management

- `LEVERAGE`: Position leverage
- `STOP_LOSS_PERCENT`: Stop-loss percentage
- `TAKE_PROFIT_PERCENT`: Take-profit percentage
- `DAILY_LOSS_LIMIT_PERCENT`: Daily loss limit

### Feature Engineering

- `MA_PERIODS`: Moving average periods
- `RSI_PERIOD`: RSI calculation period
- `MACD_*`: MACD parameters
- `BB_*`: Bollinger Bands settings
- `VOLUME_*`: Volume indicator settings
- `SUPPORT_RESIST_LOOKBACK`: Support/resistance detection window
- `REALIZED_VOL_WINDOWS`: Volatility calculation windows

### Regime Detection

- `REGIME.BULL.*`: Bull market detection parameters
- `REGIME.BEAR.*`: Bear market detection parameters
- `SIDEWAYS.*`: Sideways market detection and sub-regime classification

### Machine Learning

- `ML.BULL.RANDOM_FOREST`: Hyperparameters for bull regime models
- `ML.BEAR.RANDOM_FOREST`: Hyperparameters for bear regime models
- `ML.NEUTRAL_QUIET.RANDOM_FOREST`: Hyperparameters for quiet sideways models
- `ML.NEUTRAL_VOLATILE.RANDOM_FOREST`: Hyperparameters for volatile sideways models
- `ML.NEUTRAL_CHOPPY.RANDOM_FOREST`: Hyperparameters for choppy market models

### Prediction

- `LOOK_AHEAD`: Prediction horizon (bars)
- `PREDICTION_THRESHOLD`: Minimum return threshold for labeling

## Regime Classification

The system uses a hierarchical regime classification:

1. **Primary Regimes** (Trend Detection):
   - **BULL**: Strong uptrend (price > MAs, positive slopes)
   - **BEAR**: Strong downtrend (price < MAs, negative slopes)
   - **NEUTRAL**: Neither bull nor bear

2. **Sideways Sub-Regimes** (Neutral Decomposition):
   - **SIDEWAYS_QUIET**: Low volatility, narrow range
   - **SIDEWAYS_VOLATILE**: High volatility but range-bound
   - **CHOPPY**: Frequent trend flips, no stable pattern

Each regime has dedicated ML models with regime-specific hyperparameters.

## Model Training

Training is done **per coin** on the coin's **entire history** with regime as an input feature:

```python
# For BTC: train on all BTC data (BULL + BEAR + all NEUTRAL sub-regimes)
# For ETH: train on all ETH data (BULL + BEAR + all NEUTRAL sub-regimes)
# etc.
```

This approach (per requirements) ensures models learn regime-conditional patterns while having access to the full dataset.

## Data Flow

1. **Data Collection**: Fetch OHLCV from Binance (spot) via CCXT
2. **Feature Engineering**: Compute 50+ technical indicators and sideways features
3. **Regime Detection**: Classify into BULL/BEAR/SIDEWAYS_QUIET/SIDEWAYS_VOLATILE/CHOPPY
4. **Target Generation**: Label based on future returns (look_ahead periods)
5. **Model Training**: Train Random Forest per coin on full history
6. **Prediction**: Generate signals with confidence scores
7. **Execution**: (To be implemented) Execute via Hyperliquid API

## Logging

Logs are written to both console and file:

- Console: Configurable level (INFO default)
- File: `logs/trading_bot_YYYYMMDD.log` with rotation

Log level controlled via `LOG_LEVEL` in config.

## Safety Features

- **Testnet Mode**: Test all logic without real funds
- **Configuration Validation**: Startup checks for required parameters
- **Graceful Shutdown**: SIGINT/SIGTERM handlers for clean exit
- **Position Closing**: Optional auto-close on shutdown
- **Confidence Filtering**: Only trade signals above MIN_CONFIDENCE
- **Max Positions**: Limit concurrent positions

## Roadmap

### Phase 1: Core Infrastructure ✅

- Configuration system
- Data collection
- Feature engineering
- Regime detection
- ML model training

### Phase 2: Trading Execution (In Progress)

- [ ] Hyperliquid API client
- [ ] Order management
- [ ] Position tracking
- [ ] Risk management layer

### Phase 3: Advanced Features

- [ ] Sideways-specific strategies
- [ ] Volatility forecasting
- [ ] Statistical testing framework
- [ ] Performance metrics
- [ ] Automated retraining
- [ ] Failure diagnostics

## Development

### Running Tests

```bash
python -m pytest tests/ -v
```

### Training Single Coin

```python
from src.ml.trainer import ModelTrainer

trainer = ModelTrainer()
results = trainer.train_single_coin('BTC', update_data=True)
```

### Checking Regime Distribution

```python
from src.regime.regime_classifier import RegimeClassifier
from src.data.data_manager import DataManager

dm = DataManager()
rc = RegimeClassifier()

df = dm.load_data('BTC')
df = rc.classify_regimes(df)
print(rc.get_regime_statistics(df))
```

## Warning

⚠️ **This is an experimental code for fun & educational purposes.**

- Cryptocurrency trading carries significant risk
- Past performance does not guarantee future results
- Start with testnet and paper trading
- Never risk more than you can afford to lose

## Support

For issues and questions, please open a GitHub issue.

---

**Built with:** Python, scikit-learn, CCXT, pandas, numpy
**Exchange:** Hyperliquid (testnet & mainnet)
**Data Source:** Binance (spot market via CCXT)
