"""
Demo script for the Nifty 50 Trading Bot
Showcases the bot's capabilities with examples and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import bot components
from config import TRADING_CONFIG, RISK_CONFIG, ALGO_CONFIG
from data_manager import DataManager
from trading_strategies import StrategyManager
from risk_manager import RiskManager
from backtester import Backtester
from trading_bot import TradingBot

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def demo_data_fetching():
    """Demonstrate data fetching and technical indicators"""
    print("=" * 60)
    print("DEMO 1: DATA FETCHING AND TECHNICAL INDICATORS")
    print("=" * 60)
    
    dm = DataManager()
    
    # Fetch data for Nifty 50
    print("Fetching Nifty 50 data...")
    data = dm.fetch_market_data('^NSEI', period='6mo')
    
    if not data.empty:
        print(f"✅ Fetched {len(data)} records")
        print(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
        print(f"Price range: ₹{data['close'].min():.2f} - ₹{data['close'].max():.2f}")
        
        # Calculate technical indicators
        data_with_indicators = dm.calculate_technical_indicators(data)
        
        # Display current values
        latest = data_with_indicators.iloc[-1]
        print(f"\nLatest Values (as of {latest['timestamp']}):")
        print(f"Close Price: ₹{latest['close']:,.2f}")
        print(f"SMA 20: ₹{latest['sma_20']:,.2f}")
        print(f"SMA 50: ₹{latest['sma_50']:,.2f}")
        print(f"RSI: {latest['rsi_14']:.2f}")
        print(f"MACD: {latest['macd']:.4f}")
        
        # Plot price and indicators
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Price chart with moving averages
        ax1 = axes[0]
        ax1.plot(data_with_indicators['timestamp'], data_with_indicators['close'], 
                label='Close Price', linewidth=2, color='blue')
        ax1.plot(data_with_indicators['timestamp'], data_with_indicators['sma_20'], 
                label='SMA 20', alpha=0.7, color='orange')
        ax1.plot(data_with_indicators['timestamp'], data_with_indicators['sma_50'], 
                label='SMA 50', alpha=0.7, color='green')
        ax1.set_title('Nifty 50 Price Chart with Moving Averages')
        ax1.set_ylabel('Price (₹)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # RSI
        ax2 = axes[1]
        ax2.plot(data_with_indicators['timestamp'], data_with_indicators['rsi_14'], 
                color='purple', linewidth=2)
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought (70)')
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold (30)')
        ax2.set_title('RSI (14)')
        ax2.set_ylabel('RSI')
        ax2.set_ylim(0, 100)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # MACD
        ax3 = axes[2]
        ax3.plot(data_with_indicators['timestamp'], data_with_indicators['macd'], 
                label='MACD', color='blue', linewidth=2)
        ax3.plot(data_with_indicators['timestamp'], data_with_indicators['macd_signal'], 
                label='Signal', color='red', linewidth=2)
        ax3.bar(data_with_indicators['timestamp'], data_with_indicators['macd_histogram'], 
               alpha=0.3, color='gray', label='Histogram')
        ax3.set_title('MACD')
        ax3.set_ylabel('MACD')
        ax3.set_xlabel('Date')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('demo_technical_indicators.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    else:
        print("❌ Failed to fetch data")


def demo_strategy_comparison():
    """Demonstrate strategy comparison with backtesting"""
    print("\n" + "=" * 60)
    print("DEMO 2: STRATEGY COMPARISON")
    print("=" * 60)
    
    backtester = Backtester()
    
    # Strategies to compare
    strategies = ['ma_crossover', 'rsi_mean_reversion', 'macd_momentum', 'combined']
    symbol = '^NSEI'
    
    print(f"Running backtests for {len(strategies)} strategies on {symbol}...")
    print("This may take a moment...\n")
    
    # Run backtests
    results = {}
    for strategy in strategies:
        print(f"Testing {strategy}...")
        result = backtester.run_backtest(strategy, symbol)
        if result.metrics:
            results[strategy] = result
            print(f"✅ {strategy}: {result.metrics.get('total_return', 0):.2f}% return")
        else:
            print(f"❌ {strategy}: No results")
    
    if results:
        print(f"\n📊 STRATEGY PERFORMANCE COMPARISON")
        print("-" * 60)
        
        # Create comparison table
        comparison_data = []
        for strategy_name, result in results.items():
            comparison_data.append({
                'Strategy': strategy_name.replace('_', ' ').title(),
                'Total Return (%)': result.metrics.get('total_return', 0),
                'Sharpe Ratio': result.metrics.get('sharpe_ratio', 0),
                'Max Drawdown (%)': result.metrics.get('max_drawdown', 0),
                'Win Rate (%)': result.metrics.get('win_rate', 0),
                'Total Trades': result.metrics.get('total_trades', 0),
                'Profit Factor': result.metrics.get('profit_factor', 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False, float_format='%.2f'))
        
        # Find best strategy
        best_strategy = comparison_df.loc[comparison_df['Sharpe Ratio'].idxmax()]
        print(f"\n🏆 Best Strategy (by Sharpe Ratio): {best_strategy['Strategy']}")
        print(f"   Return: {best_strategy['Total Return (%)']}%, Sharpe: {best_strategy['Sharpe Ratio']:.2f}")
        
        # Plot comparison
        backtester.plot_results(results, symbol)


def demo_risk_management():
    """Demonstrate risk management features"""
    print("\n" + "=" * 60)
    print("DEMO 3: RISK MANAGEMENT SYSTEM")
    print("=" * 60)
    
    rm = RiskManager()
    
    # Simulate some trading scenarios
    print("Simulating trading scenarios...")
    
    # Example 1: Position sizing
    entry_price = 18000
    stop_loss = 17640  # 2% stop loss
    position_size = rm.calculate_position_size(entry_price, stop_loss)
    
    print(f"\n📈 POSITION SIZING EXAMPLE")
    print(f"Entry Price: ₹{entry_price:,}")
    print(f"Stop Loss: ₹{stop_loss:,} ({((stop_loss/entry_price-1)*100):+.1f}%)")
    print(f"Calculated Position Size: {position_size} shares")
    print(f"Position Value: ₹{entry_price * position_size:,}")
    print(f"Risk Amount: ₹{abs((entry_price - stop_loss) * position_size):,}")
    
    # Example 2: Open some positions
    print(f"\n🎯 PORTFOLIO SIMULATION")
    
    # Open positions
    positions = [
        ('^NSEI', 'long', 18000, 50),
        ('^NSEBANK', 'long', 45000, 20),
        ('^CNXIT', 'short', 35000, 25)
    ]
    
    for symbol, side, price, quantity in positions:
        success = rm.open_position(symbol, side, price, quantity)
        if success:
            print(f"✅ Opened {side} position: {quantity} shares of {symbol} at ₹{price}")
    
    # Show portfolio summary
    portfolio = rm.get_portfolio_summary()
    print(f"\n📊 PORTFOLIO SUMMARY")
    print(f"Total Capital: ₹{portfolio['total_capital']:,}")
    print(f"Available Capital: ₹{portfolio['available_capital']:,}")
    print(f"Invested Capital: ₹{portfolio['invested_capital']:,}")
    print(f"Open Positions: {portfolio['open_positions']}")
    print(f"Portfolio Utilization: {portfolio['portfolio_utilization']:.1f}%")
    
    # Simulate price changes and risk management
    print(f"\n⚠️ RISK MANAGEMENT IN ACTION")
    market_data = {
        '^NSEI': 17500,   # Down 2.8% - should trigger stop loss
        '^NSEBANK': 46000,  # Up 2.2%
        '^CNXIT': 36000    # Up 2.9% - should trigger stop loss for short position
    }
    
    print("Simulating market price changes...")
    rm.update_positions(market_data)
    
    # Show updated portfolio
    updated_portfolio = rm.get_portfolio_summary()
    print(f"Updated Open Positions: {updated_portfolio['open_positions']}")
    print(f"Daily P&L: ₹{updated_portfolio['daily_pnl']:,}")
    
    # Show risk metrics
    risk_metrics = rm.get_risk_metrics()
    if risk_metrics:
        print(f"\n📈 TRADING PERFORMANCE")
        print(f"Total Trades: {risk_metrics['total_trades']}")
        print(f"Win Rate: {risk_metrics['win_rate']:.1f}%")
        print(f"Average Win: ₹{risk_metrics['average_win']:,}")
        print(f"Average Loss: ₹{risk_metrics['average_loss']:,}")
        print(f"Profit Factor: {risk_metrics['profit_factor']:.2f}")


def demo_live_simulation():
    """Demonstrate live trading simulation"""
    print("\n" + "=" * 60)
    print("DEMO 4: LIVE TRADING SIMULATION")
    print("=" * 60)
    
    # Create trading bot
    bot = TradingBot(strategy_name='combined')
    
    print(f"🤖 Trading Bot initialized with '{bot.strategy_name}' strategy")
    
    # Run a quick backtest
    print("\n📊 Running preliminary backtest...")
    backtest_results = bot.run_backtest(days=30)
    
    if backtest_results:
        print(f"Backtest Results (30 days):")
        print(f"  Total Return: {backtest_results.get('total_return', 0):.2f}%")
        print(f"  Sharpe Ratio: {backtest_results.get('sharpe_ratio', 0):.2f}")
        print(f"  Max Drawdown: {backtest_results.get('max_drawdown', 0):.2f}%")
        print(f"  Win Rate: {backtest_results.get('win_rate', 0):.2f}%")
        print(f"  Total Trades: {backtest_results.get('total_trades', 0)}")
    
    # Simulate some market data updates
    print(f"\n📈 Simulating market data updates...")
    bot._update_market_data()
    
    if bot.market_data:
        print("Current market prices:")
        for symbol, price in bot.market_data.items():
            print(f"  {symbol}: ₹{price:,.2f}")
    
    # Simulate strategy execution
    print(f"\n🎯 Simulating strategy execution...")
    bot._execute_trading_strategy()
    
    # Show current signals
    if bot.current_signals:
        print("Generated signals:")
        for symbol, signal_info in bot.current_signals.items():
            signal = signal_info['signal']
            price = signal_info['price']
            signal_type = 'BUY' if signal > 0 else 'SELL' if signal < 0 else 'HOLD'
            print(f"  {symbol}: {signal_type} at ₹{price:,.2f}")
    
    # Get bot status
    status = bot.get_status()
    print(f"\n🤖 BOT STATUS")
    print(f"Running: {status.get('running', False)}")
    print(f"Paper Trading: {status.get('paper_trading', True)}")
    print(f"Market Open: {status.get('market_open', False)}")
    print(f"Signals Generated Today: {status.get('daily_stats', {}).get('signals_generated', 0)}")
    print(f"Trades Executed Today: {status.get('daily_stats', {}).get('trades', 0)}")
    
    # Show portfolio if any positions
    portfolio = status.get('portfolio', {})
    if portfolio.get('open_positions', 0) > 0:
        print(f"\nOpen Positions: {portfolio['open_positions']}")
        print(f"Portfolio Value: ₹{portfolio.get('invested_capital', 0):,}")
        print(f"Unrealized P&L: ₹{portfolio.get('unrealized_pnl', 0):,}")


def demo_custom_analysis():
    """Demonstrate custom analysis capabilities"""
    print("\n" + "=" * 60)
    print("DEMO 5: CUSTOM ANALYSIS")
    print("=" * 60)
    
    dm = DataManager()
    
    # Get data for multiple symbols
    symbols = ['^NSEI', '^NSEBANK', '^CNXIT']
    data_dict = {}
    
    print("Analyzing multiple Indian market indexes...")
    
    for symbol in symbols:
        data = dm.get_latest_data(symbol, 100)
        if not data.empty:
            data_dict[symbol] = data
            
    if data_dict:
        # Calculate correlations
        returns_data = {}
        for symbol, data in data_dict.items():
            returns_data[symbol] = data['close'].pct_change().dropna()
        
        returns_df = pd.DataFrame(returns_data)
        correlation_matrix = returns_df.corr()
        
        print("\n📊 CORRELATION ANALYSIS")
        print("Daily return correlations between indexes:")
        print(correlation_matrix.round(3))
        
        # Plot correlation heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, fmt='.3f')
        plt.title('Correlation Matrix - Indian Market Indexes')
        plt.tight_layout()
        plt.savefig('demo_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Performance comparison
        print(f"\n📈 PERFORMANCE COMPARISON (Last 100 days)")
        print("-" * 50)
        
        for symbol, data in data_dict.items():
            start_price = data['close'].iloc[0]
            end_price = data['close'].iloc[-1]
            total_return = ((end_price / start_price) - 1) * 100
            volatility = returns_data[symbol].std() * np.sqrt(252) * 100
            
            symbol_name = {
                '^NSEI': 'Nifty 50',
                '^NSEBANK': 'Bank Nifty', 
                '^CNXIT': 'Nifty IT'
            }.get(symbol, symbol)
            
            print(f"{symbol_name:12} | Return: {total_return:+6.2f}% | Volatility: {volatility:5.1f}%")


def main():
    """Run all demos"""
    print("🚀 NIFTY 50 TRADING BOT - COMPREHENSIVE DEMO")
    print("This demo showcases the bot's capabilities with real market data")
    print("Please wait while we fetch data and run analysis...\n")
    
    try:
        # Run all demos
        demo_data_fetching()
        demo_strategy_comparison()
        demo_risk_management()
        demo_live_simulation()
        demo_custom_analysis()
        
        print("\n" + "=" * 60)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Key takeaways:")
        print("✅ Market data fetching and technical analysis working")
        print("✅ Multiple trading strategies implemented and tested")
        print("✅ Comprehensive risk management system active")
        print("✅ Live trading simulation ready")
        print("✅ Custom analysis tools available")
        print("\nThe bot is ready for paper trading!")
        print("Check the generated images for visual analysis.")
        
    except Exception as e:
        print(f"❌ Demo encountered an error: {e}")
        print("This might be due to network issues or data availability.")
        print("Please check your internet connection and try again.")


if __name__ == "__main__":
    main()