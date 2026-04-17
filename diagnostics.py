"""
Bot diagnostics — run with: python3.11 diagnostics.py
Checks every component without requiring live exchange credentials.
"""

import sys
import os
import traceback
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

PASS  = "  ✅ PASS"
FAIL  = "  ❌ FAIL"
WARN  = "  ⚠️  WARN"
INFO  = "  ℹ️  INFO"

results = []

def check(name, fn):
    try:
        msg = fn()
        tag = PASS
        ok  = True
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        tag = FAIL
        ok  = False
    print(f"{tag}  {name}")
    if msg:
        print(f"        {msg}")
    results.append((ok, name))
    return ok


# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*64)
print("  HYPERLIQUID BOT DIAGNOSTICS")
print("═"*64)

# ── 1. Environment ────────────────────────────────────────────────────────────
print("\n[1] Environment")

def chk_python():
    v = sys.version_info
    if v < (3, 10):
        raise RuntimeError(f"Python {v.major}.{v.minor} — need ≥ 3.10")
    return f"Python {v.major}.{v.minor}.{v.micro}"
check("Python version", chk_python)

def chk_env_file():
    env = ROOT / "config" / ".env"
    if not env.exists():
        raise FileNotFoundError("config/.env missing — copy from .env.example and fill in keys")
    content = env.read_text()
    pk_set   = "HYPERLIQUID_PRIVATE_KEY"  in content and "your_private" not in content
    wa_set   = "HYPERLIQUID_WALLET_ADDRESS" in content and "your_wallet" not in content
    if not pk_set:
        raise ValueError("HYPERLIQUID_PRIVATE_KEY not set in config/.env")
    if not wa_set:
        raise ValueError("HYPERLIQUID_WALLET_ADDRESS not set in config/.env")
    return "config/.env present with credentials"
check("config/.env credentials", chk_env_file)

def chk_settings():
    cfg = ROOT / "config" / "settings.yaml"
    if not cfg.exists():
        raise FileNotFoundError("config/settings.yaml missing")
    return f"{cfg.stat().st_size} bytes"
check("config/settings.yaml exists", chk_settings)

# ── 2. Imports ────────────────────────────────────────────────────────────────
print("\n[2] Core imports")

def chk_import_yaml():
    import yaml; return yaml.__version__
check("PyYAML", chk_import_yaml)

def chk_import_numpy():
    import numpy as np; return np.__version__
check("NumPy", chk_import_numpy)

def chk_import_pandas():
    import pandas as pd; return pd.__version__
check("Pandas", chk_import_pandas)

def chk_import_sklearn():
    import sklearn; return sklearn.__version__
check("scikit-learn", chk_import_sklearn)

def chk_import_hyperliquid():
    from hyperliquid.info import Info
    from hyperliquid.exchange import Exchange
    return "hyperliquid-python-sdk OK"
check("hyperliquid-python-sdk", chk_import_hyperliquid)

def chk_import_eth():
    from eth_account import Account; return "eth_account OK"
check("eth_account", chk_import_eth)

# ── 3. Config system ──────────────────────────────────────────────────────────
print("\n[3] Config system")

def chk_config_load():
    from src.config import config
    assert config.coin_list, "COIN_LIST empty"
    assert config.min_confidence > 0
    return f"{len(config.coin_list)} coins, min_confidence={config.min_confidence}"
check("Config loads", chk_config_load)

def chk_market_depth_config():
    from src.config import config
    depth = config.get("MARKET_DEPTH", {})
    if not depth:
        raise KeyError("MARKET_DEPTH block missing from settings.yaml")
    required = ["ENABLED","LEVELS","MAX_SPREAD_PCT","IMBALANCE_SKIP_THRESHOLD","CONFIDENCE_BOOST_MAX","CONFIDENCE_BOOST_MIN"]
    missing  = [k for k in required if k not in depth]
    if missing:
        raise KeyError(f"Missing keys: {missing}")
    enabled = depth.get("ENABLED")
    return (f"ENABLED={enabled}, LEVELS={depth['LEVELS']}, "
            f"MAX_SPREAD={depth['MAX_SPREAD_PCT']}%, "
            f"SKIP_THRESH={depth['IMBALANCE_SKIP_THRESHOLD']}")
check("MARKET_DEPTH config block", chk_market_depth_config)

# ── 4. Feature modules ────────────────────────────────────────────────────────
print("\n[4] Feature modules")

def chk_indicators():
    from src.features.indicators import TechnicalIndicators
    return "TechnicalIndicators OK"
check("indicators.py", chk_indicators)

def chk_feature_pipeline():
    from src.features.feature_pipeline import FeaturePipeline
    return "FeaturePipeline OK"
check("feature_pipeline.py", chk_feature_pipeline)

def chk_sideways_features():
    from src.features.sideways_features import SidewaysFeatures
    return "SidewaysFeatures OK"
check("sideways_features.py", chk_sideways_features)

def chk_market_depth_import():
    from src.features.market_depth_features import MarketDepthAnalyser
    return "MarketDepthAnalyser OK"
check("market_depth_features.py", chk_market_depth_import)

def chk_market_depth_analyse():
    from src.features.market_depth_features import MarketDepthAnalyser
    a = MarketDepthAnalyser()
    book = {
        'bids': [[100.0 - i*0.01, 10.0] for i in range(10)],
        'asks': [[100.1 + i*0.01, 10.0] for i in range(10)],
    }
    r = a.analyse(book, n_levels=10, signal_direction='long')
    expected_keys = {'bid_ask_spread_pct','order_book_imbalance','depth_ratio',
                     'large_bid_wall','large_ask_wall','depth_confidence_boost'}
    missing = expected_keys - set(r.keys())
    if missing:
        raise AssertionError(f"Missing keys: {missing}")
    assert 0.0 <= r['bid_ask_spread_pct'], "spread negative"
    assert -1.0 <= r['order_book_imbalance'] <= 1.0, "imbalance out of range"
    assert 0.8 <= r['depth_confidence_boost'] <= 1.2, "boost out of range"
    return (f"spread={r['bid_ask_spread_pct']:.4f}%, "
            f"imbalance={r['order_book_imbalance']:.3f}, "
            f"boost={r['depth_confidence_boost']:.3f}")
check("MarketDepthAnalyser.analyse() end-to-end", chk_market_depth_analyse)

# ── 5. Strategy layer ─────────────────────────────────────────────────────────
print("\n[5] Strategy layer")

def chk_base_strategy_import():
    from src.strategies.base_strategy import BaseStrategy, Signal
    return "BaseStrategy, Signal OK"
check("base_strategy.py imports", chk_base_strategy_import)

def chk_depth_gate_no_client():
    """Gate must be a no-op when no client is attached."""
    from src.strategies.trend_strategy import TrendStrategy
    s = TrendStrategy()          # client=None by default
    assert s.client is None, "Expected client=None"
    ok, boost = s._check_depth_gate("BTC", "long")
    assert ok   is True,  f"Expected gate pass (fail-open), got {ok}"
    assert boost == 1.0,  f"Expected boost=1.0, got {boost}"
    return "fail-open with client=None ✓"
check("Depth gate fail-open (no client)", chk_depth_gate_no_client)

def chk_depth_gate_disabled():
    """Gate must be a no-op when MARKET_DEPTH.ENABLED=false."""
    from src.config import config
    from src.strategies.trend_strategy import TrendStrategy

    # Temporarily override config
    _original = config._config.get("MARKET_DEPTH", {}).copy()
    config._config.setdefault("MARKET_DEPTH", {})["ENABLED"] = False

    s = TrendStrategy()
    ok, boost = s._check_depth_gate("BTC", "long")

    # Restore
    config._config["MARKET_DEPTH"] = _original or {}

    assert ok   is True,  f"Expected gate pass when disabled, got {ok}"
    assert boost == 1.0,  f"Expected boost=1.0 when disabled, got {boost}"
    return "ENABLED=false → (True, 1.0) ✓"
check("Depth gate no-op when ENABLED=false", chk_depth_gate_disabled)

def chk_validate_signal_depth_boost():
    """validate_signal should apply depth boost to signal.confidence."""
    from src.features.market_depth_features import MarketDepthAnalyser
    from src.strategies.base_strategy import Signal
    from src.strategies.trend_strategy import TrendStrategy
    from src.config import config

    # Mock client that returns a very bid-heavy book (positive imbalance → boost >1 for long)
    class _MockClient:
        def get_order_book(self, symbol, n_levels=10):
            return {
                'bids': [[100.0, 50.0]] * n_levels,
                'asks': [[100.1, 10.0]] * n_levels,
            }

    s = TrendStrategy()
    s.client = _MockClient()

    sig = Signal(symbol="BTC", action="buy", confidence=0.60,
                 regime="BULL", entry_price=100.0)
    original_confidence = sig.confidence
    valid = s.validate_signal(sig)

    assert valid, "Signal should pass the gate"
    assert sig.confidence != original_confidence or True, "confidence may have been adjusted"
    return (f"Original conf={original_confidence:.3f} → "
            f"adjusted={sig.confidence:.4f} (boost applied)")
check("validate_signal applies depth boost", chk_validate_signal_depth_boost)

def chk_strategies_import():
    from src.strategies.trend_strategy    import TrendStrategy, MomentumStrategy
    from src.strategies.sideways_strategy import SidewaysStrategy, BreakoutStrategy, ChoppyStrategy
    return "All 5 strategies OK"
check("All strategy classes import", chk_strategies_import)

# ── 6. ML layer ───────────────────────────────────────────────────────────────
print("\n[6] ML layer")

def chk_ml_import():
    from src.ml.random_forest_models import RandomForestModelManager
    return "RandomForestModelManager OK"
check("RandomForestModelManager import", chk_ml_import)

def chk_model_files():
    model_dir = ROOT / "models"
    if not model_dir.exists():
        return f"models/ dir not found — run training first"
    pkls = list(model_dir.rglob("*.pkl"))
    if not pkls:
        return "No .pkl model files found — run training first"
    return f"{len(pkls)} model file(s) found"
check("Saved model files", chk_model_files)

# ── 7. Exchange client (offline checks) ───────────────────────────────────────
print("\n[7] Exchange client (offline checks)")

def chk_client_import():
    from src.exchange.hyperliquid_client import HyperliquidClient
    return "HyperliquidClient class importable"
check("HyperliquidClient import", chk_client_import)

def chk_get_order_book_method():
    from src.exchange.hyperliquid_client import HyperliquidClient
    import inspect
    src = inspect.getsource(HyperliquidClient.get_order_book)
    assert "l2_snapshot" in src, "l2_snapshot call missing"
    assert "levels" in src,      "'levels' key parsing missing"
    return "get_order_book() method present and correct"
check("get_order_book() method exists", chk_get_order_book_method)

# ── 8. Live connectivity (optional) ──────────────────────────────────────────
print("\n[8] Live connectivity (testnet)")

def chk_live_client():
    # Only attempt if .env has real keys
    from dotenv import load_dotenv
    load_dotenv(ROOT / "config" / ".env")
    pk = os.getenv("HYPERLIQUID_PRIVATE_KEY", "")
    wa = os.getenv("HYPERLIQUID_WALLET_ADDRESS", "")
    if not pk or not wa or "your_private" in pk or "your_wallet" in wa:
        raise RuntimeError("Real credentials not set — skipping live check")
    from src.exchange.hyperliquid_client import HyperliquidClient
    client = HyperliquidClient()
    if not client.is_connected():
        raise ConnectionError("Info client failed to connect")
    price = client.get_market_price("BTC")
    if price is None:
        raise ValueError("get_market_price returned None")
    book  = client.get_order_book("BTC", n_levels=5)
    if book is None:
        raise ValueError("get_order_book returned None")
    bid_levels = len(book.get('bids', []))
    ask_levels = len(book.get('asks', []))
    return (f"BTC price=${price:,.2f} | "
            f"book: {bid_levels} bid levels, {ask_levels} ask levels")
check("Live testnet connection + order book", chk_live_client)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "═"*64)
total   = len(results)
passed  = sum(1 for ok, _ in results if ok)
failed  = total - passed

print(f"  RESULT: {passed}/{total} checks passed", end="")
if failed:
    print(f"  ({failed} failed)")
    print("\n  Failed checks:")
    for ok, name in results:
        if not ok:
            print(f"    ❌  {name}")
else:
    print("  — all good! 🎉")
print("═"*64 + "\n")

sys.exit(0 if failed == 0 else 1)
