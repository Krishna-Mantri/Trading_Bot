"""
Backtesting Engine for the Nifty 50 Trading Bot
Tests trading strategies on historical data and provides performance analytics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from config import BACKTEST_CONFIG, RISK_CONFIG, ALGO_CONFIG
from data_manager import DataManager
from trading_strategies import StrategyManager, TradingStrategy
from risk_manager import RiskManager, Position, Trade

logger = logging.getLogger(__name__)


class BacktestResult:
    """Container for backtest results"""
    
    def __init__(self):
        self.trades = []
        self.portfolio_values = []
        self.daily_returns = []
        self.positions = []
        self.signals = pd.DataFrame()
        self.metrics = {}
        self.equity_curve = pd.DataFrame()
        
    def add_trade(self, trade: Trade):
        """Add a completed trade to results"""
        self.trades.append(trade)
    
    def calculate_metrics(self):
        """Calculate comprehensive performance metrics"""
        try:
            if not self.trades:
                return {}
            
            # Basic trade statistics
            trade_pnls = [trade.net_pnl for trade in self.trades]
            trade_returns = [trade.return_percentage for trade in self.trades]
            
            winning_trades = [t for t in self.trades if t.net_pnl > 0]
            losing_trades = [t for t in self.trades if t.net_pnl < 0]
            
            # Portfolio returns
            portfolio_returns = pd.Series(self.daily_returns).dropna()
            initial_capital = BACKTEST_CONFIG['initial_capital']
            final_value = self.portfolio_values[-1] if self.portfolio_values else initial_capital
            
            # Calculate metrics
            self.metrics = {
                # Basic metrics
                'initial_capital': initial_capital,
                'final_value': final_value,
                'total_return': ((final_value / initial_capital) - 1) * 100,
                'total_pnl': sum(trade_pnls),
                
                # Trade statistics
                'total_trades': len(self.trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': (len(winning_trades) / len(self.trades)) * 100 if self.trades else 0,
                
                # P&L statistics
                'average_win': np.mean([t.net_pnl for t in winning_trades]) if winning_trades else 0,
                'average_loss': np.mean([t.net_pnl for t in losing_trades]) if losing_trades else 0,
                'largest_win': max(trade_pnls) if trade_pnls else 0,
                'largest_loss': min(trade_pnls) if trade_pnls else 0,
                'profit_factor': (sum(t.net_pnl for t in winning_trades) / 
                                abs(sum(t.net_pnl for t in losing_trades))) if losing_trades else float('inf'),
                
                # Risk metrics
                'sharpe_ratio': self._calculate_sharpe_ratio(portfolio_returns),
                'max_drawdown': self._calculate_max_drawdown(),
                'volatility': portfolio_returns.std() * np.sqrt(252) * 100 if len(portfolio_returns) > 1 else 0,
                
                # Time-based metrics
                'avg_trade_duration': self._calculate_avg_trade_duration(),
                'trades_per_month': len(self.trades) / max(1, len(portfolio_returns) / 21),
                
                # Risk-adjusted returns
                'calmar_ratio': self._calculate_calmar_ratio(),
                'sortino_ratio': self._calculate_sortino_ratio(portfolio_returns),
            }
            
            return self.metrics
            
        except Exception as e:
            logger.error(f"Error calculating backtest metrics: {e}")
            return {}
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.06) -> float:
        """Calculate Sharpe ratio"""
        try:
            if len(returns) < 2:
                return 0.0
            
            excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
            return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() != 0 else 0
        except:
            return 0.0
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        try:
            if not self.portfolio_values:
                return 0.0
            
            portfolio_series = pd.Series(self.portfolio_values)
            rolling_max = portfolio_series.expanding().max()
            drawdown = (portfolio_series - rolling_max) / rolling_max
            return drawdown.min() * 100
        except:
            return 0.0
    
    def _calculate_calmar_ratio(self) -> float:
        """Calculate Calmar ratio (Annual return / Max drawdown)"""
        try:
            annual_return = self.metrics.get('total_return', 0) * (252 / len(self.portfolio_values)) if self.portfolio_values else 0
            max_dd = abs(self.metrics.get('max_drawdown', 1))
            return annual_return / max_dd if max_dd != 0 else 0
        except:
            return 0.0
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.06) -> float:
        """Calculate Sortino ratio"""
        try:
            if len(returns) < 2:
                return 0.0
            
            excess_returns = returns - (risk_free_rate / 252)
            downside_returns = returns[returns < 0]
            downside_deviation = downside_returns.std()
            
            return (excess_returns.mean() / downside_deviation) * np.sqrt(252) if downside_deviation != 0 else 0
        except:
            return 0.0
    
    def _calculate_avg_trade_duration(self) -> float:
        """Calculate average trade duration in days"""
        try:
            if not self.trades:
                return 0.0
            
            durations = [(trade.exit_time - trade.entry_time).days for trade in self.trades]
            return np.mean(durations)
        except:
            return 0.0


class Backtester:
    """Main backtesting engine"""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.strategy_manager = StrategyManager()
        self.commission = BACKTEST_CONFIG['commission']
        self.initial_capital = BACKTEST_CONFIG['initial_capital']
        
    def run_backtest(self, strategy_name: str, symbol: str, 
                    start_date: str = None, end_date: str = None) -> BacktestResult:
        """Run backtest for a specific strategy"""
        try:
            # Get strategy
            strategy = self.strategy_manager.get_strategy(strategy_name)
            if strategy is None:
                raise ValueError(f"Strategy '{strategy_name}' not found")
            
            # Get historical data
            if start_date is None:
                start_date = BACKTEST_CONFIG['start_date']
            if end_date is None:
                end_date = BACKTEST_CONFIG['end_date']
            
            data = self.data_manager.get_market_data(symbol, start_date, end_date)
            if data.empty:
                logger.warning(f"No data available for {symbol} between {start_date} and {end_date}")
                return BacktestResult()
            
            # Calculate technical indicators
            data = self.data_manager.calculate_technical_indicators(data)
            
            # Generate signals
            signals = strategy.generate_signals(data)
            
            # Run simulation
            result = self._simulate_trading(data, signals, symbol, strategy_name)
            result.signals = signals
            
            # Calculate metrics
            result.calculate_metrics()
            
            logger.info(f"Backtest completed for {strategy_name} on {symbol}")
            return result
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return BacktestResult()
    
    def _simulate_trading(self, data: pd.DataFrame, signals: pd.DataFrame, 
                         symbol: str, strategy_name: str) -> BacktestResult:
        """Simulate trading based on signals"""
        try:
            result = BacktestResult()
            cash = self.initial_capital
            position_size = 0
            position_entry_price = 0
            position_entry_time = None
            
            portfolio_values = []
            daily_returns = []
            
            for i, (timestamp, row) in enumerate(data.iterrows()):
                current_price = row['close']
                signal = signals.loc[timestamp, 'signal'] if timestamp in signals.index else 0
                
                # Calculate current portfolio value
                portfolio_value = cash + (position_size * current_price)
                portfolio_values.append(portfolio_value)
                
                # Calculate daily return
                if i > 0:
                    prev_value = portfolio_values[i-1]
                    daily_return = (portfolio_value - prev_value) / prev_value
                    daily_returns.append(daily_return)
                
                # Process signals
                if signal != 0 and position_size == 0:  # Open position
                    # Calculate position size based on available cash
                    shares_to_buy = int(cash * 0.95 / current_price)  # Use 95% of cash
                    
                    if shares_to_buy > 0:
                        position_size = shares_to_buy if signal > 0 else -shares_to_buy
                        position_entry_price = current_price
                        position_entry_time = timestamp
                        
                        # Calculate commission
                        commission_cost = abs(position_size * current_price * self.commission)
                        cash -= abs(position_size * current_price) + commission_cost
                        
                        logger.debug(f"Opened position: {position_size} shares at {current_price}")
                
                elif signal != 0 and position_size != 0:  # Close position
                    if (signal > 0 and position_size < 0) or (signal < 0 and position_size > 0):
                        # Exit position
                        exit_price = current_price
                        exit_time = timestamp
                        
                        # Calculate P&L
                        if position_size > 0:  # Long position
                            pnl = (exit_price - position_entry_price) * position_size
                        else:  # Short position
                            pnl = (position_entry_price - exit_price) * abs(position_size)
                        
                        # Calculate commission
                        commission_cost = abs(position_size * exit_price * self.commission)
                        cash += abs(position_size * exit_price) - commission_cost
                        
                        # Create trade record
                        trade = Trade(
                            symbol=symbol,
                            side='long' if position_size > 0 else 'short',
                            entry_price=position_entry_price,
                            exit_price=exit_price,
                            quantity=abs(position_size),
                            entry_time=position_entry_time,
                            exit_time=exit_time,
                            pnl=pnl,
                            commission=commission_cost * 2  # Entry + exit commission
                        )
                        
                        result.add_trade(trade)
                        
                        logger.debug(f"Closed position: P&L = {trade.net_pnl:.2f}")
                        
                        # Reset position
                        position_size = 0
                        position_entry_price = 0
                        position_entry_time = None
            
            # Close any remaining position at the end
            if position_size != 0:
                exit_price = data['close'].iloc[-1]
                exit_time = data.index[-1]
                
                if position_size > 0:  # Long position
                    pnl = (exit_price - position_entry_price) * position_size
                else:  # Short position
                    pnl = (position_entry_price - exit_price) * abs(position_size)
                
                commission_cost = abs(position_size * exit_price * self.commission)
                
                trade = Trade(
                    symbol=symbol,
                    side='long' if position_size > 0 else 'short',
                    entry_price=position_entry_price,
                    exit_price=exit_price,
                    quantity=abs(position_size),
                    entry_time=position_entry_time,
                    exit_time=exit_time,
                    pnl=pnl,
                    commission=commission_cost * 2
                )
                
                result.add_trade(trade)
            
            result.portfolio_values = portfolio_values
            result.daily_returns = daily_returns
            
            return result
            
        except Exception as e:
            logger.error(f"Error in trading simulation: {e}")
            return BacktestResult()
    
    def compare_strategies(self, strategies: List[str], symbol: str, 
                          start_date: str = None, end_date: str = None) -> Dict[str, BacktestResult]:
        """Compare multiple strategies"""
        try:
            results = {}
            
            for strategy_name in strategies:
                logger.info(f"Running backtest for {strategy_name}")
                result = self.run_backtest(strategy_name, symbol, start_date, end_date)
                results[strategy_name] = result
            
            return results
            
        except Exception as e:
            logger.error(f"Error comparing strategies: {e}")
            return {}
    
    def optimize_strategy(self, strategy_name: str, symbol: str, 
                         parameter_ranges: Dict[str, List]) -> Dict:
        """Optimize strategy parameters"""
        try:
            best_params = {}
            best_score = -float('inf')
            best_result = None
            
            # Generate parameter combinations
            import itertools
            param_names = list(parameter_ranges.keys())
            param_values = list(parameter_ranges.values())
            
            for params in itertools.product(*param_values):
                # Update strategy parameters
                param_dict = dict(zip(param_names, params))
                
                # Create strategy with new parameters
                # This would need to be implemented based on strategy type
                # For now, we'll use the default strategy
                
                result = self.run_backtest(strategy_name, symbol)
                
                # Score based on Sharpe ratio or total return
                score = result.metrics.get('sharpe_ratio', 0)
                
                if score > best_score:
                    best_score = score
                    best_params = param_dict
                    best_result = result
            
            return {
                'best_parameters': best_params,
                'best_score': best_score,
                'best_result': best_result
            }
            
        except Exception as e:
            logger.error(f"Error optimizing strategy: {e}")
            return {}
    
    def generate_report(self, result: BacktestResult, strategy_name: str) -> str:
        """Generate a comprehensive backtest report"""
        try:
            report = f"""
=== BACKTEST REPORT: {strategy_name.upper()} ===

PERFORMANCE SUMMARY:
- Initial Capital: ₹{result.metrics.get('initial_capital', 0):,.2f}
- Final Value: ₹{result.metrics.get('final_value', 0):,.2f}
- Total Return: {result.metrics.get('total_return', 0):.2f}%
- Total P&L: ₹{result.metrics.get('total_pnl', 0):,.2f}

TRADE STATISTICS:
- Total Trades: {result.metrics.get('total_trades', 0)}
- Winning Trades: {result.metrics.get('winning_trades', 0)}
- Losing Trades: {result.metrics.get('losing_trades', 0)}
- Win Rate: {result.metrics.get('win_rate', 0):.2f}%
- Average Win: ₹{result.metrics.get('average_win', 0):,.2f}
- Average Loss: ₹{result.metrics.get('average_loss', 0):,.2f}
- Largest Win: ₹{result.metrics.get('largest_win', 0):,.2f}
- Largest Loss: ₹{result.metrics.get('largest_loss', 0):,.2f}
- Profit Factor: {result.metrics.get('profit_factor', 0):.2f}

RISK METRICS:
- Sharpe Ratio: {result.metrics.get('sharpe_ratio', 0):.2f}
- Sortino Ratio: {result.metrics.get('sortino_ratio', 0):.2f}
- Calmar Ratio: {result.metrics.get('calmar_ratio', 0):.2f}
- Maximum Drawdown: {result.metrics.get('max_drawdown', 0):.2f}%
- Volatility: {result.metrics.get('volatility', 0):.2f}%

TRADING METRICS:
- Average Trade Duration: {result.metrics.get('avg_trade_duration', 0):.1f} days
- Trades per Month: {result.metrics.get('trades_per_month', 0):.1f}
"""
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return "Error generating report"
    
    def plot_results(self, results: Dict[str, BacktestResult], symbol: str):
        """Plot backtest results"""
        try:
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Portfolio value comparison
            ax1 = axes[0, 0]
            for strategy_name, result in results.items():
                if result.portfolio_values:
                    dates = pd.date_range(start=BACKTEST_CONFIG['start_date'], 
                                        periods=len(result.portfolio_values), freq='D')
                    ax1.plot(dates, result.portfolio_values, label=strategy_name, linewidth=2)
            
            ax1.set_title('Portfolio Value Over Time')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Portfolio Value (₹)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Drawdown chart
            ax2 = axes[0, 1]
            for strategy_name, result in results.items():
                if result.portfolio_values:
                    portfolio_series = pd.Series(result.portfolio_values)
                    rolling_max = portfolio_series.expanding().max()
                    drawdown = (portfolio_series - rolling_max) / rolling_max * 100
                    
                    dates = pd.date_range(start=BACKTEST_CONFIG['start_date'], 
                                        periods=len(drawdown), freq='D')
                    ax2.fill_between(dates, drawdown, 0, alpha=0.3, label=strategy_name)
            
            ax2.set_title('Drawdown')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Drawdown (%)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Performance metrics comparison
            ax3 = axes[1, 0]
            metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
            strategy_names = list(results.keys())
            
            x = np.arange(len(strategy_names))
            width = 0.2
            
            for i, metric in enumerate(metrics):
                values = [results[strategy].metrics.get(metric, 0) for strategy in strategy_names]
                ax3.bar(x + i * width, values, width, label=metric)
            
            ax3.set_title('Performance Metrics Comparison')
            ax3.set_xlabel('Strategy')
            ax3.set_ylabel('Value')
            ax3.set_xticks(x + width * 1.5)
            ax3.set_xticklabels(strategy_names)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Trade distribution
            ax4 = axes[1, 1]
            for strategy_name, result in results.items():
                if result.trades:
                    pnls = [trade.net_pnl for trade in result.trades]
                    ax4.hist(pnls, alpha=0.6, label=strategy_name, bins=20)
            
            ax4.set_title('Trade P&L Distribution')
            ax4.set_xlabel('P&L (₹)')
            ax4.set_ylabel('Frequency')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'backtest_results_{symbol}_{datetime.now().strftime("%Y%m%d")}.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting results: {e}")


if __name__ == "__main__":
    # Run backtests
    backtester = Backtester()
    
    # Test strategies
    strategies = ['ma_crossover', 'rsi_mean_reversion', 'macd_momentum', 'combined']
    symbol = '^NSEI'
    
    results = backtester.compare_strategies(strategies, symbol)
    
    # Print reports
    for strategy_name, result in results.items():
        if result.metrics:
            print(backtester.generate_report(result, strategy_name))
            print("\n" + "="*80 + "\n")
    
    # Plot results
    if results:
        backtester.plot_results(results, symbol)