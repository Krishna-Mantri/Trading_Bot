"""
Risk Management System for the Nifty 50 Trading Bot
Handles position sizing, stop losses, and portfolio risk management
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from config import RISK_CONFIG, TRADING_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    quantity: int
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    current_price: Optional[float] = None
    
    @property
    def market_value(self) -> float:
        """Calculate current market value of the position"""
        if self.current_price is None:
            return self.entry_price * self.quantity
        return self.current_price * self.quantity
    
    @property
    def pnl(self) -> float:
        """Calculate unrealized P&L"""
        if self.current_price is None:
            return 0.0
        
        if self.side == 'long':
            return (self.current_price - self.entry_price) * self.quantity
        else:  # short
            return (self.entry_price - self.current_price) * self.quantity
    
    @property
    def pnl_percentage(self) -> float:
        """Calculate P&L as percentage"""
        initial_value = self.entry_price * self.quantity
        return (self.pnl / initial_value) * 100 if initial_value != 0 else 0


@dataclass
class Trade:
    """Represents a completed trade"""
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: int
    entry_time: datetime
    exit_time: datetime
    pnl: float
    commission: float = 0.0
    
    @property
    def net_pnl(self) -> float:
        """Calculate net P&L after commission"""
        return self.pnl - self.commission
    
    @property
    def return_percentage(self) -> float:
        """Calculate return as percentage"""
        initial_value = self.entry_price * self.quantity
        return (self.net_pnl / initial_value) * 100 if initial_value != 0 else 0


class RiskManager:
    """Comprehensive risk management system"""
    
    def __init__(self):
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.daily_pnl = 0.0
        self.total_capital = RISK_CONFIG['max_position_size'] * 10  # Assume 10x capital
        self.available_capital = self.total_capital
        self.max_daily_loss = RISK_CONFIG['max_daily_loss']
        self.max_position_size = RISK_CONFIG['max_position_size']
        self.max_open_positions = RISK_CONFIG['max_open_positions']
        self.risk_per_trade = RISK_CONFIG['risk_per_trade']
        
        logger.info("Risk Manager initialized")
    
    def calculate_position_size(self, entry_price: float, stop_loss_price: float, 
                              risk_amount: Optional[float] = None) -> int:
        """Calculate optimal position size based on risk management rules"""
        try:
            if risk_amount is None:
                risk_amount = self.total_capital * (self.risk_per_trade / 100)
            
            # Calculate risk per share
            risk_per_share = abs(entry_price - stop_loss_price)
            
            if risk_per_share == 0:
                logger.warning("Risk per share is zero, using minimum position size")
                return 1
            
            # Calculate position size based on risk
            position_size = int(risk_amount / risk_per_share)
            
            # Apply maximum position size limit
            max_shares = int(self.max_position_size / entry_price)
            position_size = min(position_size, max_shares)
            
            # Ensure minimum position size
            position_size = max(position_size, 1)
            
            logger.info(f"Calculated position size: {position_size} shares")
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 1
    
    def calculate_stop_loss(self, entry_price: float, side: str) -> float:
        """Calculate stop loss price"""
        try:
            stop_loss_pct = RISK_CONFIG['stop_loss_pct'] / 100
            
            if side == 'long':
                stop_loss = entry_price * (1 - stop_loss_pct)
            else:  # short
                stop_loss = entry_price * (1 + stop_loss_pct)
            
            return stop_loss
            
        except Exception as e:
            logger.error(f"Error calculating stop loss: {e}")
            return entry_price
    
    def calculate_take_profit(self, entry_price: float, side: str) -> float:
        """Calculate take profit price"""
        try:
            take_profit_pct = RISK_CONFIG['take_profit_pct'] / 100
            
            if side == 'long':
                take_profit = entry_price * (1 + take_profit_pct)
            else:  # short
                take_profit = entry_price * (1 - take_profit_pct)
            
            return take_profit
            
        except Exception as e:
            logger.error(f"Error calculating take profit: {e}")
            return entry_price
    
    def can_open_position(self, symbol: str, position_value: float) -> Tuple[bool, str]:
        """Check if a new position can be opened"""
        try:
            # Check maximum number of open positions
            if len(self.positions) >= self.max_open_positions:
                return False, f"Maximum open positions ({self.max_open_positions}) reached"
            
            # Check if position already exists for this symbol
            if symbol in self.positions:
                return False, f"Position already exists for {symbol}"
            
            # Check available capital
            if position_value > self.available_capital:
                return False, f"Insufficient capital. Required: {position_value}, Available: {self.available_capital}"
            
            # Check daily loss limit
            if self.daily_pnl <= -self.max_daily_loss:
                return False, f"Daily loss limit ({self.max_daily_loss}) exceeded"
            
            # Check position size limit
            if position_value > self.max_position_size:
                return False, f"Position size exceeds maximum allowed ({self.max_position_size})"
            
            return True, "Position can be opened"
            
        except Exception as e:
            logger.error(f"Error checking position constraints: {e}")
            return False, "Error checking constraints"
    
    def open_position(self, symbol: str, side: str, entry_price: float, 
                     quantity: int, stop_loss: Optional[float] = None,
                     take_profit: Optional[float] = None) -> bool:
        """Open a new position"""
        try:
            position_value = entry_price * quantity
            
            # Check if position can be opened
            can_open, reason = self.can_open_position(symbol, position_value)
            if not can_open:
                logger.warning(f"Cannot open position for {symbol}: {reason}")
                return False
            
            # Calculate stop loss and take profit if not provided
            if stop_loss is None:
                stop_loss = self.calculate_stop_loss(entry_price, side)
            
            if take_profit is None:
                take_profit = self.calculate_take_profit(entry_price, side)
            
            # Create position
            position = Position(
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                quantity=quantity,
                entry_time=datetime.now(),
                stop_loss=stop_loss,
                take_profit=take_profit,
                current_price=entry_price
            )
            
            # Add position to portfolio
            self.positions[symbol] = position
            self.available_capital -= position_value
            
            logger.info(f"Opened {side} position for {symbol}: {quantity} shares at {entry_price}")
            return True
            
        except Exception as e:
            logger.error(f"Error opening position for {symbol}: {e}")
            return False
    
    def close_position(self, symbol: str, exit_price: float, 
                      reason: str = "Manual close") -> bool:
        """Close an existing position"""
        try:
            if symbol not in self.positions:
                logger.warning(f"No position found for {symbol}")
                return False
            
            position = self.positions[symbol]
            
            # Calculate P&L
            if position.side == 'long':
                pnl = (exit_price - position.entry_price) * position.quantity
            else:  # short
                pnl = (position.entry_price - exit_price) * position.quantity
            
            # Create trade record
            trade = Trade(
                symbol=symbol,
                side=position.side,
                entry_price=position.entry_price,
                exit_price=exit_price,
                quantity=position.quantity,
                entry_time=position.entry_time,
                exit_time=datetime.now(),
                pnl=pnl,
                commission=self._calculate_commission(position.entry_price * position.quantity)
            )
            
            # Update portfolio
            self.trades.append(trade)
            self.daily_pnl += trade.net_pnl
            self.available_capital += exit_price * position.quantity
            
            # Remove position
            del self.positions[symbol]
            
            logger.info(f"Closed {position.side} position for {symbol}: P&L = {trade.net_pnl:.2f} ({reason})")
            return True
            
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
            return False
    
    def update_positions(self, market_data: Dict[str, float]):
        """Update current prices and check for stop loss/take profit triggers"""
        try:
            positions_to_close = []
            
            for symbol, position in self.positions.items():
                if symbol in market_data:
                    current_price = market_data[symbol]
                    position.current_price = current_price
                    
                    # Check stop loss
                    if self._should_trigger_stop_loss(position, current_price):
                        positions_to_close.append((symbol, current_price, "Stop Loss"))
                    
                    # Check take profit
                    elif self._should_trigger_take_profit(position, current_price):
                        positions_to_close.append((symbol, current_price, "Take Profit"))
            
            # Close triggered positions
            for symbol, price, reason in positions_to_close:
                self.close_position(symbol, price, reason)
            
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    def _should_trigger_stop_loss(self, position: Position, current_price: float) -> bool:
        """Check if stop loss should be triggered"""
        if position.stop_loss is None:
            return False
        
        if position.side == 'long':
            return current_price <= position.stop_loss
        else:  # short
            return current_price >= position.stop_loss
    
    def _should_trigger_take_profit(self, position: Position, current_price: float) -> bool:
        """Check if take profit should be triggered"""
        if position.take_profit is None:
            return False
        
        if position.side == 'long':
            return current_price >= position.take_profit
        else:  # short
            return current_price <= position.take_profit
    
    def _calculate_commission(self, position_value: float) -> float:
        """Calculate trading commission"""
        # Simple commission calculation (can be customized)
        return position_value * 0.001  # 0.1% commission
    
    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary"""
        try:
            total_positions_value = sum(pos.market_value for pos in self.positions.values())
            total_pnl = sum(pos.pnl for pos in self.positions.values())
            
            summary = {
                'total_capital': self.total_capital,
                'available_capital': self.available_capital,
                'invested_capital': total_positions_value,
                'open_positions': len(self.positions),
                'unrealized_pnl': total_pnl,
                'daily_pnl': self.daily_pnl,
                'total_trades': len(self.trades),
                'portfolio_utilization': (total_positions_value / self.total_capital) * 100
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating portfolio summary: {e}")
            return {}
    
    def get_risk_metrics(self) -> Dict:
        """Calculate risk metrics"""
        try:
            if not self.trades:
                return {}
            
            # Calculate metrics from completed trades
            trade_returns = [trade.return_percentage for trade in self.trades]
            trade_pnls = [trade.net_pnl for trade in self.trades]
            
            winning_trades = [t for t in self.trades if t.net_pnl > 0]
            losing_trades = [t for t in self.trades if t.net_pnl < 0]
            
            metrics = {
                'total_trades': len(self.trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': (len(winning_trades) / len(self.trades)) * 100 if self.trades else 0,
                'average_win': np.mean([t.net_pnl for t in winning_trades]) if winning_trades else 0,
                'average_loss': np.mean([t.net_pnl for t in losing_trades]) if losing_trades else 0,
                'largest_win': max(trade_pnls) if trade_pnls else 0,
                'largest_loss': min(trade_pnls) if trade_pnls else 0,
                'total_pnl': sum(trade_pnls),
                'average_return': np.mean(trade_returns) if trade_returns else 0,
                'return_volatility': np.std(trade_returns) if trade_returns else 0,
            }
            
            # Calculate profit factor
            total_profit = sum(t.net_pnl for t in winning_trades) if winning_trades else 0
            total_loss = abs(sum(t.net_pnl for t in losing_trades)) if losing_trades else 1
            metrics['profit_factor'] = total_profit / total_loss if total_loss > 0 else 0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def reset_daily_pnl(self):
        """Reset daily P&L (should be called at market open)"""
        self.daily_pnl = 0.0
        logger.info("Daily P&L reset")
    
    def validate_trade_signal(self, symbol: str, signal: int, current_price: float) -> Tuple[bool, str]:
        """Validate if a trade signal should be executed"""
        try:
            # Check daily loss limit
            if self.daily_pnl <= -self.max_daily_loss:
                return False, "Daily loss limit exceeded"
            
            # Check if position already exists
            if symbol in self.positions:
                existing_side = self.positions[symbol].side
                new_side = 'long' if signal > 0 else 'short'
                
                if existing_side == new_side:
                    return False, "Position already exists in same direction"
            
            # Check maximum open positions
            if len(self.positions) >= self.max_open_positions and symbol not in self.positions:
                return False, "Maximum open positions reached"
            
            # Calculate position size and check capital
            stop_loss = self.calculate_stop_loss(current_price, 'long' if signal > 0 else 'short')
            position_size = self.calculate_position_size(current_price, stop_loss)
            position_value = current_price * position_size
            
            if position_value > self.available_capital:
                return False, "Insufficient available capital"
            
            return True, "Signal validated"
            
        except Exception as e:
            logger.error(f"Error validating trade signal: {e}")
            return False, "Validation error"


class PortfolioTracker:
    """Tracks portfolio performance over time"""
    
    def __init__(self):
        self.daily_snapshots = []
        self.performance_history = pd.DataFrame()
    
    def take_snapshot(self, risk_manager: RiskManager):
        """Take a daily portfolio snapshot"""
        try:
            summary = risk_manager.get_portfolio_summary()
            risk_metrics = risk_manager.get_risk_metrics()
            
            snapshot = {
                'date': datetime.now().date(),
                'total_capital': summary.get('total_capital', 0),
                'portfolio_value': summary.get('total_capital', 0) + summary.get('unrealized_pnl', 0),
                'unrealized_pnl': summary.get('unrealized_pnl', 0),
                'daily_pnl': summary.get('daily_pnl', 0),
                'open_positions': summary.get('open_positions', 0),
                'total_trades': risk_metrics.get('total_trades', 0),
                'win_rate': risk_metrics.get('win_rate', 0)
            }
            
            self.daily_snapshots.append(snapshot)
            logger.info("Portfolio snapshot taken")
            
        except Exception as e:
            logger.error(f"Error taking portfolio snapshot: {e}")
    
    def get_performance_chart_data(self) -> pd.DataFrame:
        """Get data for performance charts"""
        try:
            if not self.daily_snapshots:
                return pd.DataFrame()
            
            df = pd.DataFrame(self.daily_snapshots)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Calculate cumulative returns
            df['portfolio_return'] = df['portfolio_value'].pct_change()
            df['cumulative_return'] = (1 + df['portfolio_return']).cumprod()
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating performance chart data: {e}")
            return pd.DataFrame()


if __name__ == "__main__":
    # Test the risk manager
    rm = RiskManager()
    
    # Test position sizing
    entry_price = 18000
    stop_loss = 17640  # 2% stop loss
    position_size = rm.calculate_position_size(entry_price, stop_loss)
    print(f"Position size: {position_size}")
    
    # Test opening a position
    success = rm.open_position('^NSEI', 'long', entry_price, position_size)
    print(f"Position opened: {success}")
    
    # Test portfolio summary
    summary = rm.get_portfolio_summary()
    print("Portfolio Summary:", summary)