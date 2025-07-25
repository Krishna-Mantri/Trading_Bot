"""
Main Trading Bot for Nifty 50 Algorithmic Trading
Orchestrates all components and provides the main execution loop
"""

import pandas as pd
import numpy as np
import logging
import time
import schedule
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import sqlite3

from config import (
    TRADING_CONFIG, RISK_CONFIG, ALGO_CONFIG, 
    NOTIFICATION_CONFIG, LOGGING_CONFIG, DATABASE_CONFIG
)
from data_manager import DataManager
from trading_strategies import StrategyManager, SignalType
from risk_manager import RiskManager, PortfolioTracker
from backtester import Backtester

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG['level']),
    format=LOGGING_CONFIG['format'],
    handlers=[
        logging.FileHandler(LOGGING_CONFIG['file']),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TradingBot:
    """Main Trading Bot class"""
    
    def __init__(self, strategy_name: str = 'combined'):
        """Initialize the trading bot"""
        try:
            # Initialize components
            self.data_manager = DataManager()
            self.strategy_manager = StrategyManager()
            self.risk_manager = RiskManager()
            self.portfolio_tracker = PortfolioTracker()
            self.backtester = Backtester()
            
            # Configuration
            self.strategy_name = strategy_name
            self.primary_symbol = TRADING_CONFIG['symbol']
            self.symbols = TRADING_CONFIG['index_symbols']
            self.running = False
            self.paper_trading = True  # Start in paper trading mode
            
            # Get the selected strategy
            self.strategy = self.strategy_manager.get_strategy(strategy_name)
            if self.strategy is None:
                raise ValueError(f"Strategy '{strategy_name}' not found")
            
            # Trading state
            self.current_signals = {}
            self.last_update = None
            self.market_data = {}
            
            # Performance tracking
            self.daily_stats = {
                'date': datetime.now().date(),
                'trades': 0,
                'pnl': 0.0,
                'signals_generated': 0,
                'signals_executed': 0
            }
            
            logger.info(f"Trading Bot initialized with strategy: {strategy_name}")
            
        except Exception as e:
            logger.error(f"Error initializing trading bot: {e}")
            raise
    
    def start(self, paper_trading: bool = True):
        """Start the trading bot"""
        try:
            self.paper_trading = paper_trading
            self.running = True
            
            mode = "PAPER TRADING" if paper_trading else "LIVE TRADING"
            logger.info(f"Starting Trading Bot in {mode} mode")
            
            # Schedule tasks
            self._schedule_tasks()
            
            # Initial data update
            self._update_market_data()
            
            # Start main loop
            self._run_main_loop()
            
        except Exception as e:
            logger.error(f"Error starting trading bot: {e}")
            self.stop()
    
    def stop(self):
        """Stop the trading bot"""
        try:
            self.running = False
            
            # Close all open positions in paper trading
            if self.paper_trading:
                self._close_all_positions("Bot stopped")
            
            # Save final portfolio snapshot
            self.portfolio_tracker.take_snapshot(self.risk_manager)
            
            logger.info("Trading Bot stopped")
            
        except Exception as e:
            logger.error(f"Error stopping trading bot: {e}")
    
    def _schedule_tasks(self):
        """Schedule recurring tasks"""
        try:
            # Market data updates (every 15 minutes during trading hours)
            schedule.every(15).minutes.do(self._update_market_data)
            
            # Strategy execution (every 5 minutes during trading hours)
            schedule.every(5).minutes.do(self._execute_trading_strategy)
            
            # Risk management check (every minute)
            schedule.every(1).minutes.do(self._check_risk_management)
            
            # Daily tasks
            schedule.every().day.at("09:00").do(self._daily_startup)
            schedule.every().day.at("15:45").do(self._daily_shutdown)
            
            # Portfolio snapshot (end of day)
            schedule.every().day.at("16:00").do(self._take_daily_snapshot)
            
            logger.info("Tasks scheduled successfully")
            
        except Exception as e:
            logger.error(f"Error scheduling tasks: {e}")
    
    def _run_main_loop(self):
        """Main execution loop"""
        try:
            while self.running:
                # Check if market is open
                if self.data_manager.is_market_open():
                    # Run scheduled tasks
                    schedule.run_pending()
                    
                    # Sleep for a short interval
                    time.sleep(30)  # Check every 30 seconds
                else:
                    # Market is closed, sleep longer
                    logger.debug("Market is closed, sleeping...")
                    time.sleep(300)  # Check every 5 minutes when market is closed
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, stopping bot...")
            self.stop()
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            self.stop()
    
    def _update_market_data(self):
        """Update market data for all symbols"""
        try:
            logger.info("Updating market data...")
            
            # Update data for all symbols
            for symbol in self.symbols:
                # Get latest data
                latest_data = self.data_manager.get_latest_data(symbol, 100)
                
                if not latest_data.empty:
                    current_price = latest_data['close'].iloc[-1]
                    self.market_data[symbol] = current_price
                    
                    logger.debug(f"Updated {symbol}: {current_price}")
            
            self.last_update = datetime.now()
            logger.info(f"Market data updated at {self.last_update}")
            
        except Exception as e:
            logger.error(f"Error updating market data: {e}")
    
    def _execute_trading_strategy(self):
        """Execute the trading strategy"""
        try:
            if not self.data_manager.is_market_open():
                return
            
            logger.info("Executing trading strategy...")
            
            for symbol in self.symbols:
                # Get latest data with indicators
                data = self.data_manager.get_latest_data(symbol, 100)
                
                if data.empty:
                    logger.warning(f"No data available for {symbol}")
                    continue
                
                # Generate signals
                signals = self.strategy.generate_signals(data)
                
                if signals.empty:
                    logger.warning(f"No signals generated for {symbol}")
                    continue
                
                # Get latest signal
                latest_signal = signals['signal'].iloc[-1]
                current_price = data['close'].iloc[-1]
                
                # Store signal
                self.current_signals[symbol] = {
                    'signal': latest_signal,
                    'price': current_price,
                    'timestamp': datetime.now()
                }
                
                self.daily_stats['signals_generated'] += 1
                
                # Process signal
                self._process_signal(symbol, latest_signal, current_price)
            
            logger.info("Strategy execution completed")
            
        except Exception as e:
            logger.error(f"Error executing trading strategy: {e}")
    
    def _process_signal(self, symbol: str, signal: int, current_price: float):
        """Process a trading signal"""
        try:
            if signal == 0:  # No signal
                return
            
            # Validate signal
            is_valid, reason = self.risk_manager.validate_trade_signal(symbol, signal, current_price)
            
            if not is_valid:
                logger.info(f"Signal rejected for {symbol}: {reason}")
                return
            
            # Determine action
            side = 'long' if signal > 0 else 'short'
            
            # Check if we already have a position
            if symbol in self.risk_manager.positions:
                existing_position = self.risk_manager.positions[symbol]
                
                # If signal is opposite to existing position, close it
                if (existing_position.side == 'long' and signal < 0) or \
                   (existing_position.side == 'short' and signal > 0):
                    self._close_position(symbol, current_price, "Signal reversal")
                    return
                else:
                    # Same direction, ignore
                    return
            
            # Calculate position size
            stop_loss = self.risk_manager.calculate_stop_loss(current_price, side)
            position_size = self.risk_manager.calculate_position_size(current_price, stop_loss)
            
            # Execute trade
            if self.paper_trading:
                success = self._execute_paper_trade(symbol, side, current_price, position_size)
            else:
                success = self._execute_live_trade(symbol, side, current_price, position_size)
            
            if success:
                self.daily_stats['signals_executed'] += 1
                self.daily_stats['trades'] += 1
                logger.info(f"Trade executed: {side} {position_size} shares of {symbol} at {current_price}")
            
        except Exception as e:
            logger.error(f"Error processing signal for {symbol}: {e}")
    
    def _execute_paper_trade(self, symbol: str, side: str, price: float, quantity: int) -> bool:
        """Execute a paper trade (simulation)"""
        try:
            success = self.risk_manager.open_position(symbol, side, price, quantity)
            
            if success:
                logger.info(f"Paper trade executed: {side} {quantity} {symbol} at {price}")
                return True
            else:
                logger.warning(f"Paper trade failed for {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing paper trade: {e}")
            return False
    
    def _execute_live_trade(self, symbol: str, side: str, price: float, quantity: int) -> bool:
        """Execute a live trade (placeholder for broker integration)"""
        try:
            # This would integrate with a real broker API
            # For now, we'll simulate it
            logger.warning("Live trading not implemented yet, executing as paper trade")
            return self._execute_paper_trade(symbol, side, price, quantity)
            
        except Exception as e:
            logger.error(f"Error executing live trade: {e}")
            return False
    
    def _close_position(self, symbol: str, price: float, reason: str):
        """Close a position"""
        try:
            success = self.risk_manager.close_position(symbol, price, reason)
            
            if success:
                self.daily_stats['trades'] += 1
                logger.info(f"Position closed for {symbol} at {price}: {reason}")
            
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
    
    def _close_all_positions(self, reason: str):
        """Close all open positions"""
        try:
            for symbol in list(self.risk_manager.positions.keys()):
                if symbol in self.market_data:
                    current_price = self.market_data[symbol]
                    self._close_position(symbol, current_price, reason)
            
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
    
    def _check_risk_management(self):
        """Check risk management rules"""
        try:
            if not self.market_data:
                return
            
            # Update positions with current prices
            self.risk_manager.update_positions(self.market_data)
            
            # Update daily P&L
            portfolio_summary = self.risk_manager.get_portfolio_summary()
            self.daily_stats['pnl'] = portfolio_summary.get('daily_pnl', 0)
            
            # Log current status
            open_positions = portfolio_summary.get('open_positions', 0)
            unrealized_pnl = portfolio_summary.get('unrealized_pnl', 0)
            
            logger.debug(f"Open positions: {open_positions}, Unrealized P&L: {unrealized_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error in risk management check: {e}")
    
    def _daily_startup(self):
        """Daily startup routine"""
        try:
            logger.info("Executing daily startup routine...")
            
            # Reset daily P&L
            self.risk_manager.reset_daily_pnl()
            
            # Reset daily stats
            self.daily_stats = {
                'date': datetime.now().date(),
                'trades': 0,
                'pnl': 0.0,
                'signals_generated': 0,
                'signals_executed': 0
            }
            
            # Update all market data
            self.data_manager.update_all_data()
            
            logger.info("Daily startup completed")
            
        except Exception as e:
            logger.error(f"Error in daily startup: {e}")
    
    def _daily_shutdown(self):
        """Daily shutdown routine"""
        try:
            logger.info("Executing daily shutdown routine...")
            
            # Close all positions (if configured to do so)
            # self._close_all_positions("End of day")
            
            # Log daily performance
            self._log_daily_performance()
            
            logger.info("Daily shutdown completed")
            
        except Exception as e:
            logger.error(f"Error in daily shutdown: {e}")
    
    def _take_daily_snapshot(self):
        """Take daily portfolio snapshot"""
        try:
            self.portfolio_tracker.take_snapshot(self.risk_manager)
            logger.info("Daily portfolio snapshot taken")
            
        except Exception as e:
            logger.error(f"Error taking daily snapshot: {e}")
    
    def _log_daily_performance(self):
        """Log daily performance statistics"""
        try:
            portfolio_summary = self.risk_manager.get_portfolio_summary()
            risk_metrics = self.risk_manager.get_risk_metrics()
            
            logger.info("=== DAILY PERFORMANCE SUMMARY ===")
            logger.info(f"Date: {self.daily_stats['date']}")
            logger.info(f"Trades executed: {self.daily_stats['trades']}")
            logger.info(f"Signals generated: {self.daily_stats['signals_generated']}")
            logger.info(f"Signals executed: {self.daily_stats['signals_executed']}")
            logger.info(f"Daily P&L: ₹{self.daily_stats['pnl']:.2f}")
            logger.info(f"Open positions: {portfolio_summary.get('open_positions', 0)}")
            logger.info(f"Portfolio utilization: {portfolio_summary.get('portfolio_utilization', 0):.1f}%")
            logger.info(f"Total trades: {risk_metrics.get('total_trades', 0)}")
            logger.info(f"Win rate: {risk_metrics.get('win_rate', 0):.1f}%")
            logger.info("==================================")
            
        except Exception as e:
            logger.error(f"Error logging daily performance: {e}")
    
    def get_status(self) -> Dict:
        """Get current bot status"""
        try:
            portfolio_summary = self.risk_manager.get_portfolio_summary()
            risk_metrics = self.risk_manager.get_risk_metrics()
            
            status = {
                'running': self.running,
                'paper_trading': self.paper_trading,
                'strategy': self.strategy_name,
                'last_update': self.last_update.isoformat() if self.last_update else None,
                'market_open': self.data_manager.is_market_open(),
                'daily_stats': self.daily_stats,
                'portfolio': portfolio_summary,
                'risk_metrics': risk_metrics,
                'current_signals': self.current_signals,
                'positions': {symbol: {
                    'side': pos.side,
                    'quantity': pos.quantity,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'pnl': pos.pnl,
                    'pnl_percentage': pos.pnl_percentage
                } for symbol, pos in self.risk_manager.positions.items()}
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting bot status: {e}")
            return {}
    
    def run_backtest(self, days: int = 252) -> Dict:
        """Run a quick backtest"""
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            result = self.backtester.run_backtest(
                self.strategy_name, 
                self.primary_symbol, 
                start_date, 
                end_date
            )
            
            if result.metrics:
                logger.info(f"Backtest completed for {self.strategy_name}")
                return result.metrics
            else:
                logger.warning("Backtest returned no results")
                return {}
                
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return {}


def main():
    """Main function to run the trading bot"""
    try:
        # Create and start the trading bot
        bot = TradingBot(strategy_name='combined')
        
        # Run a quick backtest first
        logger.info("Running initial backtest...")
        backtest_results = bot.run_backtest(days=90)
        
        if backtest_results:
            logger.info("Backtest Results:")
            logger.info(f"Total Return: {backtest_results.get('total_return', 0):.2f}%")
            logger.info(f"Sharpe Ratio: {backtest_results.get('sharpe_ratio', 0):.2f}")
            logger.info(f"Max Drawdown: {backtest_results.get('max_drawdown', 0):.2f}%")
            logger.info(f"Win Rate: {backtest_results.get('win_rate', 0):.2f}%")
        
        # Start the trading bot
        bot.start(paper_trading=True)
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        logger.info("Trading bot terminated")


if __name__ == "__main__":
    main()