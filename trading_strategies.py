"""
Trading Strategies for the Nifty 50 Trading Bot
Contains various algorithmic trading strategies
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod
from enum import Enum

from config import ALGO_CONFIG, RISK_CONFIG

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Trading signal types"""
    BUY = 1
    SELL = -1
    HOLD = 0


class TradingStrategy(ABC):
    """Abstract base class for trading strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.signals = []
        self.performance_metrics = {}
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on the strategy"""
        pass
    
    def calculate_returns(self, data: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        """Calculate strategy returns"""
        try:
            returns = data['close'].pct_change()
            strategy_returns = returns * signals['signal'].shift(1)
            cumulative_returns = (1 + strategy_returns).cumprod()
            
            result = pd.DataFrame({
                'returns': returns,
                'strategy_returns': strategy_returns,
                'cumulative_returns': cumulative_returns,
                'signals': signals['signal']
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating returns for {self.name}: {e}")
            return pd.DataFrame()


class MovingAverageCrossoverStrategy(TradingStrategy):
    """Simple Moving Average Crossover Strategy"""
    
    def __init__(self, short_window: int = 20, long_window: int = 50):
        super().__init__("MA_Crossover")
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on moving average crossover"""
        try:
            signals = pd.DataFrame(index=data.index)
            signals['signal'] = 0
            
            # Calculate moving averages
            signals['short_ma'] = data['close'].rolling(window=self.short_window).mean()
            signals['long_ma'] = data['close'].rolling(window=self.long_window).mean()
            
            # Generate signals
            signals['signal'][self.short_window:] = np.where(
                signals['short_ma'][self.short_window:] > signals['long_ma'][self.short_window:], 1, 0
            )
            
            # Create positions (1 for buy, -1 for sell, 0 for hold)
            signals['positions'] = signals['signal'].diff()
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating MA crossover signals: {e}")
            return pd.DataFrame()


class RSIMeanReversionStrategy(TradingStrategy):
    """RSI Mean Reversion Strategy"""
    
    def __init__(self, rsi_period: int = 14, oversold: int = 30, overbought: int = 70):
        super().__init__("RSI_MeanReversion")
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on RSI mean reversion"""
        try:
            signals = pd.DataFrame(index=data.index)
            signals['signal'] = 0
            
            # Use RSI from data if available, otherwise calculate
            if 'rsi_14' in data.columns:
                rsi = data['rsi_14']
            else:
                rsi = self._calculate_rsi(data['close'], self.rsi_period)
            
            signals['rsi'] = rsi
            
            # Generate signals
            signals['signal'] = np.where(rsi < self.oversold, 1,  # Buy when oversold
                                np.where(rsi > self.overbought, -1, 0))  # Sell when overbought
            
            # Create positions
            signals['positions'] = signals['signal'].diff()
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating RSI signals: {e}")
            return pd.DataFrame()
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class MACDMomentumStrategy(TradingStrategy):
    """MACD Momentum Strategy"""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        super().__init__("MACD_Momentum")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on MACD momentum"""
        try:
            signals = pd.DataFrame(index=data.index)
            signals['signal'] = 0
            
            # Use MACD from data if available, otherwise calculate
            if 'macd' in data.columns and 'macd_signal' in data.columns:
                macd_line = data['macd']
                signal_line = data['macd_signal']
            else:
                macd_line, signal_line = self._calculate_macd(data['close'])
            
            signals['macd'] = macd_line
            signals['macd_signal'] = signal_line
            signals['macd_histogram'] = macd_line - signal_line
            
            # Generate signals based on MACD crossover
            signals['signal'] = np.where(macd_line > signal_line, 1, -1)
            
            # Additional filter: only trade when MACD is above/below zero line
            signals['signal'] = np.where(
                (signals['signal'] == 1) & (macd_line > 0), 1,
                np.where((signals['signal'] == -1) & (macd_line < 0), -1, 0)
            )
            
            # Create positions
            signals['positions'] = signals['signal'].diff()
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating MACD signals: {e}")
            return pd.DataFrame()
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=self.fast_period).mean()
        ema_slow = prices.ewm(span=self.slow_period).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal_period).mean()
        return macd_line, signal_line


class BollingerBandsStrategy(TradingStrategy):
    """Bollinger Bands Mean Reversion Strategy"""
    
    def __init__(self, period: int = 20, std_dev: float = 2):
        super().__init__("BollingerBands")
        self.period = period
        self.std_dev = std_dev
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on Bollinger Bands"""
        try:
            signals = pd.DataFrame(index=data.index)
            signals['signal'] = 0
            
            # Use BB from data if available, otherwise calculate
            if all(col in data.columns for col in ['bb_upper', 'bb_lower', 'bb_middle']):
                upper_band = data['bb_upper']
                lower_band = data['bb_lower']
                middle_band = data['bb_middle']
            else:
                upper_band, lower_band, middle_band = self._calculate_bollinger_bands(data['close'])
            
            signals['bb_upper'] = upper_band
            signals['bb_lower'] = lower_band
            signals['bb_middle'] = middle_band
            signals['price'] = data['close']
            
            # Generate signals
            # Buy when price touches lower band, sell when price touches upper band
            signals['signal'] = np.where(data['close'] <= lower_band, 1,
                                np.where(data['close'] >= upper_band, -1, 0))
            
            # Exit positions when price returns to middle band
            signals['signal'] = np.where(
                (signals['signal'].shift(1) == 1) & (data['close'] >= middle_band), 0,
                np.where((signals['signal'].shift(1) == -1) & (data['close'] <= middle_band), 0,
                        signals['signal'])
            )
            
            # Create positions
            signals['positions'] = signals['signal'].diff()
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating Bollinger Bands signals: {e}")
            return pd.DataFrame()
    
    def _calculate_bollinger_bands(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle_band = prices.rolling(window=self.period).mean()
        std = prices.rolling(window=self.period).std()
        upper_band = middle_band + (std * self.std_dev)
        lower_band = middle_band - (std * self.std_dev)
        return upper_band, lower_band, middle_band


class CombinedStrategy(TradingStrategy):
    """Combined strategy using multiple indicators"""
    
    def __init__(self):
        super().__init__("Combined_Strategy")
        self.ma_strategy = MovingAverageCrossoverStrategy()
        self.rsi_strategy = RSIMeanReversionStrategy()
        self.macd_strategy = MACDMomentumStrategy()
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on multiple indicators"""
        try:
            signals = pd.DataFrame(index=data.index)
            signals['signal'] = 0
            
            # Get signals from individual strategies
            ma_signals = self.ma_strategy.generate_signals(data)
            rsi_signals = self.rsi_strategy.generate_signals(data)
            macd_signals = self.macd_strategy.generate_signals(data)
            
            # Combine signals with different weights
            signals['ma_signal'] = ma_signals['signal']
            signals['rsi_signal'] = rsi_signals['signal']
            signals['macd_signal'] = macd_signals['signal']
            
            # Weighted combination
            signals['combined_score'] = (
                0.4 * ma_signals['signal'] +
                0.3 * rsi_signals['signal'] +
                0.3 * macd_signals['signal']
            )
            
            # Generate final signals based on threshold
            signals['signal'] = np.where(signals['combined_score'] > 0.5, 1,
                                np.where(signals['combined_score'] < -0.5, -1, 0))
            
            # Additional filters
            if 'adx' in data.columns:
                # Only trade when ADX > 25 (trending market)
                signals['signal'] = np.where(data['adx'] > 25, signals['signal'], 0)
            
            if 'volume' in data.columns:
                # Volume confirmation
                volume_ma = data['volume'].rolling(window=20).mean()
                signals['signal'] = np.where(data['volume'] > volume_ma, signals['signal'], 0)
            
            # Create positions
            signals['positions'] = signals['signal'].diff()
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating combined signals: {e}")
            return pd.DataFrame()


class StrategyManager:
    """Manages multiple trading strategies"""
    
    def __init__(self):
        self.strategies = {
            'ma_crossover': MovingAverageCrossoverStrategy(
                ALGO_CONFIG['sma_short'], ALGO_CONFIG['sma_long']
            ),
            'rsi_mean_reversion': RSIMeanReversionStrategy(
                ALGO_CONFIG['rsi_period'], ALGO_CONFIG['rsi_oversold'], ALGO_CONFIG['rsi_overbought']
            ),
            'macd_momentum': MACDMomentumStrategy(
                ALGO_CONFIG['macd_fast'], ALGO_CONFIG['macd_slow'], ALGO_CONFIG['macd_signal']
            ),
            'bollinger_bands': BollingerBandsStrategy(
                ALGO_CONFIG['bb_period'], ALGO_CONFIG['bb_std']
            ),
            'combined': CombinedStrategy()
        }
    
    def get_strategy(self, strategy_name: str) -> TradingStrategy:
        """Get a specific strategy"""
        return self.strategies.get(strategy_name)
    
    def get_all_signals(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Get signals from all strategies"""
        all_signals = {}
        
        for name, strategy in self.strategies.items():
            try:
                signals = strategy.generate_signals(data)
                all_signals[name] = signals
                logger.info(f"Generated signals for {name}")
            except Exception as e:
                logger.error(f"Error generating signals for {name}: {e}")
        
        return all_signals
    
    def evaluate_strategies(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """Evaluate performance of all strategies"""
        performance = {}
        
        for name, strategy in self.strategies.items():
            try:
                signals = strategy.generate_signals(data)
                returns = strategy.calculate_returns(data, signals)
                
                if not returns.empty:
                    metrics = self._calculate_performance_metrics(returns)
                    performance[name] = metrics
                    logger.info(f"Evaluated performance for {name}")
                
            except Exception as e:
                logger.error(f"Error evaluating {name}: {e}")
        
        return performance
    
    def _calculate_performance_metrics(self, returns: pd.DataFrame) -> Dict:
        """Calculate performance metrics for a strategy"""
        try:
            strategy_returns = returns['strategy_returns'].dropna()
            cumulative_returns = returns['cumulative_returns'].dropna()
            
            # Basic metrics
            total_return = (cumulative_returns.iloc[-1] - 1) * 100
            annual_return = (cumulative_returns.iloc[-1] ** (252 / len(strategy_returns)) - 1) * 100
            volatility = strategy_returns.std() * np.sqrt(252) * 100
            
            # Sharpe ratio
            sharpe_ratio = (annual_return / volatility) if volatility != 0 else 0
            
            # Max drawdown
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min() * 100
            
            # Win rate
            winning_trades = strategy_returns[strategy_returns > 0]
            total_trades = strategy_returns[strategy_returns != 0]
            win_rate = (len(winning_trades) / len(total_trades) * 100) if len(total_trades) > 0 else 0
            
            metrics = {
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'total_trades': len(total_trades)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}


if __name__ == "__main__":
    # Test strategies
    from data_manager import DataManager
    
    dm = DataManager()
    data = dm.get_latest_data('^NSEI', 100)
    
    if not data.empty:
        sm = StrategyManager()
        performance = sm.evaluate_strategies(data)
        
        for strategy_name, metrics in performance.items():
            print(f"\n{strategy_name.upper()} Strategy Performance:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.2f}")
    else:
        print("No data available for testing")