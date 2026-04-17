"""
Portfolio-level risk controls.
Manages overall portfolio risk and limits.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from src.config import config


logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Portfolio risk metrics."""
    total_equity: float
    total_margin_used: float
    margin_usage_pct: float
    total_pnl: float
    daily_pnl: float
    daily_pnl_pct: float
    position_count: int
    max_positions: int
    leverage_ratio: float
    at_risk_positions: List[str]


class PortfolioRisk:
    """Manage portfolio-level risk controls."""
    
    def __init__(self):
        """Initialize portfolio risk manager."""
        # Load risk limits from config
        self.max_positions = config.max_positions
        self.max_position_size_pct = config.get('MAX_POSITION_SIZE_PERCENT', 5.0)
        self.daily_loss_limit_pct = config.get('DAILY_LOSS_LIMIT_PERCENT', 5.0)
        self.leverage = config.leverage
        
        # Track daily PnL
        self.daily_pnl_tracker = {}  # date -> pnl
        self.start_of_day_equity = None
        self.last_reset_date = datetime.now().date()
        
        # Emergency stop flag
        self.emergency_stop = False
        self.stop_reason = None
        
        logger.info(f"Initialized portfolio risk: max_positions={self.max_positions}, "
                   f"daily_loss_limit={self.daily_loss_limit_pct}%")
    
    def calculate_metrics(
        self,
        account_equity: float,
        margin_used: float,
        total_pnl: float,
        position_count: int,
        at_risk_positions: List[str]
    ) -> RiskMetrics:
        """
        Calculate portfolio risk metrics.
        
        Args:
            account_equity: Total account equity
            margin_used: Total margin used
            total_pnl: Total unrealized PnL
            position_count: Number of open positions
            at_risk_positions: Positions at liquidation risk
            
        Returns:
            RiskMetrics object
        """
        # Reset daily tracking if new day
        self._check_daily_reset(account_equity)
        
        # Calculate margin usage
        margin_usage_pct = (margin_used / account_equity * 100) if account_equity > 0 else 0
        
        # Calculate daily PnL
        daily_pnl = self._calculate_daily_pnl(account_equity)
        daily_pnl_pct = (daily_pnl / self.start_of_day_equity * 100) if self.start_of_day_equity else 0
        
        # Calculate effective leverage
        total_notional = margin_used * self.leverage
        leverage_ratio = (total_notional / account_equity) if account_equity > 0 else 0
        
        metrics = RiskMetrics(
            total_equity=account_equity,
            total_margin_used=margin_used,
            margin_usage_pct=margin_usage_pct,
            total_pnl=total_pnl,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            position_count=position_count,
            max_positions=self.max_positions,
            leverage_ratio=leverage_ratio,
            at_risk_positions=at_risk_positions
        )
        
        return metrics
    
    def check_risk_limits(self, metrics: RiskMetrics) -> Dict[str, bool]:
        """
        Check if risk limits are breached.
        
        Args:
            metrics: Risk metrics
            
        Returns:
            Dictionary of limit checks
        """
        checks = {
            'max_positions_ok': metrics.position_count < metrics.max_positions,
            'daily_loss_ok': metrics.daily_pnl_pct > -self.daily_loss_limit_pct,
            'margin_ok': metrics.margin_usage_pct < 90,  # 90% margin usage limit
            'no_liquidation_risk': len(metrics.at_risk_positions) == 0,
            'emergency_stop_ok': not self.emergency_stop,
        }
        
        # Log warnings for breached limits
        if not checks['max_positions_ok']:
            logger.warning(f"Max positions limit reached: {metrics.position_count}/{metrics.max_positions}")
        
        if not checks['daily_loss_ok']:
            logger.warning(f"Daily loss limit breached: {metrics.daily_pnl_pct:.2f}% "
                         f"(limit: -{self.daily_loss_limit_pct}%)")
        
        if not checks['margin_ok']:
            logger.warning(f"High margin usage: {metrics.margin_usage_pct:.2f}%")
        
        if not checks['no_liquidation_risk']:
            logger.warning(f"Positions at liquidation risk: {metrics.at_risk_positions}")
        
        return checks
    
    def can_open_position(self, metrics: RiskMetrics) -> tuple[bool, Optional[str]]:
        """
        Check if a new position can be opened.
        
        Args:
            metrics: Current risk metrics
            
        Returns:
            Tuple of (can_open, reason_if_not)
        """
        checks = self.check_risk_limits(metrics)
        
        if not checks['emergency_stop_ok']:
            return False, f"Emergency stop active: {self.stop_reason}"
        
        if not checks['max_positions_ok']:
            return False, "Maximum positions reached"
        
        if not checks['daily_loss_ok']:
            return False, "Daily loss limit exceeded"
        
        if not checks['margin_ok']:
            return False, "Margin usage too high"
        
        if not checks['no_liquidation_risk']:
            return False, "Existing positions at liquidation risk"
        
        return True, None
    
    def should_reduce_risk(self, metrics: RiskMetrics) -> tuple[bool, List[str]]:
        """
        Check if risk should be reduced.
        
        Args:
            metrics: Current risk metrics
            
        Returns:
            Tuple of (should_reduce, reasons)
        """
        reasons = []
        
        # Check daily loss approaching limit
        if metrics.daily_pnl_pct < -(self.daily_loss_limit_pct * 0.8):
            reasons.append(f"Daily loss at {metrics.daily_pnl_pct:.2f}% "
                          f"(80% of limit)")
        
        # Check high margin usage
        if metrics.margin_usage_pct > 80:
            reasons.append(f"High margin usage: {metrics.margin_usage_pct:.2f}%")
        
        # Check liquidation risk
        if len(metrics.at_risk_positions) > 0:
            reasons.append(f"Liquidation risk: {metrics.at_risk_positions}")
        
        # Check too many positions
        if metrics.position_count >= metrics.max_positions * 0.9:
            reasons.append(f"Near max positions: {metrics.position_count}/{metrics.max_positions}")
        
        return len(reasons) > 0, reasons
    
    def trigger_emergency_stop(self, reason: str):
        """
        Trigger emergency stop.
        
        Args:
            reason: Reason for emergency stop
        """
        self.emergency_stop = True
        self.stop_reason = reason
        logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")
    
    def reset_emergency_stop(self):
        """Reset emergency stop."""
        self.emergency_stop = False
        self.stop_reason = None
        logger.info("Emergency stop reset")
    
    def _check_daily_reset(self, current_equity: float):
        """Check if we need to reset daily tracking."""
        today = datetime.now().date()
        
        if today != self.last_reset_date:
            # New day - reset tracking
            self.last_reset_date = today
            self.start_of_day_equity = current_equity
            logger.info(f"Daily reset: start_of_day_equity=${current_equity:.2f}")
        elif self.start_of_day_equity is None:
            # First time initialization
            self.start_of_day_equity = current_equity
    
    def _calculate_daily_pnl(self, current_equity: float) -> float:
        """Calculate daily PnL."""
        if self.start_of_day_equity is None:
            return 0.0
        
        return current_equity - self.start_of_day_equity
    
    def get_risk_summary(self, metrics: RiskMetrics) -> Dict:
        """
        Get risk summary.
        
        Args:
            metrics: Risk metrics
            
        Returns:
            Summary dictionary
        """
        checks = self.check_risk_limits(metrics)
        should_reduce, reduce_reasons = self.should_reduce_risk(metrics)
        
        return {
            'metrics': {
                'equity': metrics.total_equity,
                'margin_used': metrics.total_margin_used,
                'margin_usage_pct': metrics.margin_usage_pct,
                'total_pnl': metrics.total_pnl,
                'daily_pnl': metrics.daily_pnl,
                'daily_pnl_pct': metrics.daily_pnl_pct,
                'position_count': metrics.position_count,
                'leverage_ratio': metrics.leverage_ratio,
            },
            'limits': {
                'max_positions': self.max_positions,
                'daily_loss_limit_pct': self.daily_loss_limit_pct,
                'max_leverage': self.leverage,
            },
            'checks': checks,
            'all_limits_ok': all(checks.values()),
            'should_reduce_risk': should_reduce,
            'reduce_reasons': reduce_reasons,
            'emergency_stop': self.emergency_stop,
            'stop_reason': self.stop_reason,
        }
