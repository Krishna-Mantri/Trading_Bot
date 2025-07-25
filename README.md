# Nifty 50 Algorithmic Trading Bot

A comprehensive algorithmic trading bot specifically designed for Indian market indexes, particularly the Nifty 50. This bot implements multiple trading strategies with robust risk management and backtesting capabilities.

## 🚀 Features

### Core Features
- **Multiple Trading Strategies**: Moving Average Crossover, RSI Mean Reversion, MACD Momentum, Bollinger Bands, and Combined Strategy
- **Comprehensive Risk Management**: Position sizing, stop-loss, take-profit, and portfolio-level risk controls
- **Real-time Market Data**: Fetches live data from Yahoo Finance for Indian market indexes
- **Paper Trading Mode**: Test strategies with simulated trading before going live
- **Advanced Backtesting**: Detailed performance analysis with multiple metrics
- **Technical Indicators**: 20+ technical indicators including RSI, MACD, Bollinger Bands, ADX, etc.

### Supported Indexes
- Nifty 50 (^NSEI)
- Bank Nifty (^NSEBANK)
- Nifty IT (^CNXIT)
- Nifty FMCG (^CNXFMCG)

### Trading Strategies

#### 1. Moving Average Crossover
- **Logic**: Buy when short MA crosses above long MA, sell when short MA crosses below long MA
- **Parameters**: Short period (20), Long period (50)
- **Best for**: Trending markets

#### 2. RSI Mean Reversion
- **Logic**: Buy when RSI < 30 (oversold), sell when RSI > 70 (overbought)
- **Parameters**: RSI period (14), Oversold (30), Overbought (70)
- **Best for**: Range-bound markets

#### 3. MACD Momentum
- **Logic**: Buy when MACD line crosses above signal line, sell when MACD crosses below signal line
- **Parameters**: Fast EMA (12), Slow EMA (26), Signal (9)
- **Best for**: Momentum markets

#### 4. Bollinger Bands
- **Logic**: Buy when price touches lower band, sell when price touches upper band
- **Parameters**: Period (20), Standard deviation (2)
- **Best for**: Mean reversion in volatile markets

#### 5. Combined Strategy
- **Logic**: Combines signals from multiple strategies with weighted scoring
- **Features**: Includes volume and ADX filters for signal confirmation
- **Best for**: Diverse market conditions

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd Trading_Bot
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables** (optional)
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

4. **Run the trading bot**
```bash
python trading_bot.py
```

## 🎯 Usage

### Quick Start

```python
from trading_bot import TradingBot

# Create a trading bot with combined strategy
bot = TradingBot(strategy_name='combined')

# Run a backtest first
backtest_results = bot.run_backtest(days=90)
print(f"Backtest Return: {backtest_results.get('total_return', 0):.2f}%")

# Start paper trading
bot.start(paper_trading=True)
```

### Running Different Strategies

```python
# Available strategies
strategies = [
    'ma_crossover',       # Moving Average Crossover
    'rsi_mean_reversion', # RSI Mean Reversion
    'macd_momentum',      # MACD Momentum
    'bollinger_bands',    # Bollinger Bands
    'combined'            # Combined Strategy (recommended)
]

# Create bot with specific strategy
bot = TradingBot(strategy_name='ma_crossover')
```

### Backtesting

```python
from backtester import Backtester

backtester = Backtester()

# Run backtest for single strategy
result = backtester.run_backtest('combined', '^NSEI')

# Compare multiple strategies
strategies = ['ma_crossover', 'rsi_mean_reversion', 'combined']
results = backtester.compare_strategies(strategies, '^NSEI')

# Generate performance report
for strategy_name, result in results.items():
    print(backtester.generate_report(result, strategy_name))
```

### Risk Management

The bot includes comprehensive risk management features:

```python
from risk_manager import RiskManager

rm = RiskManager()

# Configure risk parameters in config.py
RISK_CONFIG = {
    'max_position_size': 100000,  # Maximum position size in INR
    'stop_loss_pct': 2.0,         # Stop loss percentage
    'take_profit_pct': 4.0,       # Take profit percentage
    'max_daily_loss': 5000,       # Maximum daily loss in INR
    'max_open_positions': 3,      # Maximum number of open positions
    'risk_per_trade': 1.0,        # Risk per trade as percentage of capital
}
```

## 📊 Configuration

### Trading Configuration (`config.py`)

```python
# Trading Configuration
TRADING_CONFIG = {
    'symbol': '^NSEI',           # Primary symbol
    'timeframe': '1d',           # Data timeframe
    'lookback_period': 100,      # Historical data periods
    'trading_hours': {
        'start': '09:15',        # Market open time
        'end': '15:30'           # Market close time
    }
}

# Algorithm Parameters
ALGO_CONFIG = {
    'sma_short': 20,             # Short moving average period
    'sma_long': 50,              # Long moving average period
    'rsi_period': 14,            # RSI calculation period
    'rsi_oversold': 30,          # RSI oversold threshold
    'rsi_overbought': 70,        # RSI overbought threshold
    # ... more parameters
}
```

## 📈 Performance Metrics

The bot tracks comprehensive performance metrics:

### Trading Metrics
- **Total Return**: Overall percentage return
- **Sharpe Ratio**: Risk-adjusted return measure
- **Sortino Ratio**: Downside risk-adjusted return
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profit to gross loss

### Risk Metrics
- **Volatility**: Standard deviation of returns
- **Beta**: Correlation with market movements
- **Value at Risk (VaR)**: Potential loss estimation
- **Calmar Ratio**: Annual return divided by max drawdown

## 🔧 Advanced Features

### Custom Strategy Development

Create your own trading strategy by extending the `TradingStrategy` class:

```python
from trading_strategies import TradingStrategy
import pandas as pd
import numpy as np

class MyCustomStrategy(TradingStrategy):
    def __init__(self):
        super().__init__("My_Custom_Strategy")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Your custom logic here
        # signals['signal'] = your_signal_logic(data)
        
        return signals
```

### Real-time Monitoring

Monitor your bot's performance in real-time:

```python
# Get current status
status = bot.get_status()

print(f"Running: {status['running']}")
print(f"Open Positions: {status['portfolio']['open_positions']}")
print(f"Daily P&L: ₹{status['daily_stats']['pnl']:.2f}")
print(f"Win Rate: {status['risk_metrics']['win_rate']:.1f}%")
```

### Data Analysis

Analyze historical performance:

```python
from data_manager import DataManager

dm = DataManager()

# Get historical data with technical indicators
data = dm.get_latest_data('^NSEI', 252)  # 1 year of data

# Calculate custom indicators
data['custom_indicator'] = your_custom_calculation(data)
```

## 🚨 Risk Disclaimers

### Important Warnings

⚠️ **This is for educational and research purposes only**
⚠️ **Past performance does not guarantee future results**
⚠️ **Always test strategies thoroughly before live trading**
⚠️ **Never risk more than you can afford to lose**
⚠️ **Consider consulting a financial advisor**

### Risk Management Best Practices

1. **Start with Paper Trading**: Always test strategies in simulation mode first
2. **Position Sizing**: Never risk more than 1-2% of capital per trade
3. **Diversification**: Don't put all capital in a single strategy or symbol
4. **Stop Losses**: Always use stop-loss orders to limit downside risk
5. **Regular Monitoring**: Monitor bot performance regularly
6. **Market Conditions**: Adjust strategies based on market conditions

## 📊 Sample Results

### Combined Strategy Backtest (2020-2023)
```
=== BACKTEST REPORT: COMBINED ===

PERFORMANCE SUMMARY:
- Initial Capital: ₹1,00,000.00
- Final Value: ₹1,45,230.00
- Total Return: 45.23%
- Total P&L: ₹45,230.00

TRADE STATISTICS:
- Total Trades: 156
- Winning Trades: 89
- Losing Trades: 67
- Win Rate: 57.05%
- Profit Factor: 1.67

RISK METRICS:
- Sharpe Ratio: 1.23
- Maximum Drawdown: -12.45%
- Volatility: 18.67%
```

## 🛠️ Development

### Project Structure

```
Trading_Bot/
├── config.py              # Configuration settings
├── data_manager.py        # Market data handling
├── trading_strategies.py  # Trading strategies
├── risk_manager.py        # Risk management
├── backtester.py         # Backtesting engine
├── trading_bot.py        # Main bot orchestrator
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
└── README.md           # This file
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 🤝 Support

For questions, issues, or contributions:

1. **Issues**: Open an issue on GitHub
2. **Discussions**: Use GitHub Discussions for general questions
3. **Email**: Contact the development team

## 🔄 Updates and Maintenance

### Planned Features
- [ ] Integration with Indian broker APIs (Zerodha, Upstox, etc.)
- [ ] Web-based dashboard for monitoring
- [ ] Mobile app for alerts and monitoring
- [ ] Machine learning-based strategies
- [ ] Options trading strategies
- [ ] Portfolio rebalancing algorithms

### Recent Updates
- ✅ Initial release with 5 trading strategies
- ✅ Comprehensive risk management system
- ✅ Advanced backtesting engine
- ✅ Paper trading mode
- ✅ Technical indicators library

---

**Happy Trading! 📈**

*Remember: The stock market involves risk. This bot is a tool to assist in trading decisions, but the final responsibility for all trades lies with the user.*