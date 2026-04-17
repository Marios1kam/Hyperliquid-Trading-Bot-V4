"""Test configuration loading."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config


def test_config_loading():
    """Test that configuration loads correctly."""
    print("Testing configuration loading...")
    
    # Test basic parameters
    assert config.coin_list is not None, "COIN_LIST not loaded"
    assert len(config.coin_list) > 0, "COIN_LIST is empty"
    print(f"[OK] Loaded {len(config.coin_list)} coins")
    
    assert config.max_positions > 0, "MAX_POSITIONS invalid"
    print(f"[OK] MAX_POSITIONS: {config.max_positions}")
    
    assert 0 <= config.min_confidence <= 1, "MIN_CONFIDENCE out of range"
    print(f"[OK] MIN_CONFIDENCE: {config.min_confidence}")
    
    # Test feature parameters
    ma_periods = config.get('MA_PERIODS')
    assert ma_periods is not None, "MA_PERIODS not loaded"
    print(f"[OK] MA_PERIODS: {ma_periods}")
    
    # Test regime parameters
    bull_config = config.get('REGIME.BULL')
    assert bull_config is not None, "REGIME.BULL not loaded"
    print(f"[OK] REGIME.BULL loaded")
    
    sideways_config = config.get('SIDEWAYS')
    assert sideways_config is not None, "SIDEWAYS config not loaded"
    print(f"[OK] SIDEWAYS config loaded")
    
    # Test ML parameters
    ml_bull = config.get('ML.BULL.RANDOM_FOREST')
    assert ml_bull is not None, "ML.BULL.RANDOM_FOREST not loaded"
    print(f"[OK] ML.BULL.RANDOM_FOREST: {ml_bull}")
    
    # Test prediction parameters
    assert config.look_ahead > 0, "LOOK_AHEAD invalid"
    print(f"[OK] LOOK_AHEAD: {config.look_ahead}")
    
    print("\n[SUCCESS] All configuration tests passed!")
    return True


if __name__ == '__main__':
    try:
        test_config_loading()
    except AssertionError as e:
        print(f"\n[FAILED] Configuration test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        sys.exit(1)
