"""
Configuration file for the Nifty 50 Trading Bot
Contains all trading parameters, risk management settings, and API configurations
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Trading Configuration
TRADING_CONFIG = {
    'symbol': '^NSEI',  # Nifty 50 symbol on Yahoo Finance
    'index_symbols': [
        '^NSEI',    # Nifty 50
        '^NSEBANK', # Bank Nifty
        '^CNXIT',   # Nifty IT
        '^CNXFMCG', # Nifty FMCG
    ],
    'timeframe': '1d',  # 1d, 1h, 15m, 5m
    'lookback_period': 100,  # Number of historical candles
    'trading_hours': {
        'start': '09:15',
        'end': '15:30'
    }
}

# Risk Management
RISK_CONFIG = {
    'max_position_size': 100000,  # Maximum position size in INR
    'stop_loss_pct': 2.0,  # Stop loss percentage
    'take_profit_pct': 4.0,  # Take profit percentage
    'max_daily_loss': 5000,  # Maximum daily loss in INR
    'max_open_positions': 3,  # Maximum number of open positions
    'risk_per_trade': 1.0,  # Risk per trade as percentage of capital
}

# Algorithm Parameters
ALGO_CONFIG = {
    'sma_short': 20,
    'sma_long': 50,
    'ema_short': 12,
    'ema_long': 26,
    'rsi_period': 14,
    'rsi_oversold': 30,
    'rsi_overbought': 70,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'bb_period': 20,
    'bb_std': 2,
    'adx_period': 14,
    'volume_threshold': 1.5,  # Volume threshold multiplier
}

# Backtesting Configuration
BACKTEST_CONFIG = {
    'start_date': '2020-01-01',
    'end_date': '2023-12-31',
    'initial_capital': 100000,
    'commission': 0.001,  # 0.1% commission per trade
}

# Database Configuration
DATABASE_CONFIG = {
    'db_path': 'trading_bot.db',
    'backup_interval': 24,  # hours
}

# Notification Configuration
NOTIFICATION_CONFIG = {
    'email_enabled': False,
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'email_user': os.getenv('EMAIL_USER'),
    'email_password': os.getenv('EMAIL_PASSWORD'),
    'email_recipients': ['your_email@gmail.com'],
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'trading_bot.log',
    'max_bytes': 10485760,  # 10MB
    'backup_count': 5,
}

# API Keys (to be set in .env file)
API_CONFIG = {
    'alpha_vantage_key': os.getenv('ALPHA_VANTAGE_KEY'),
    'zerodha_api_key': os.getenv('ZERODHA_API_KEY'),
    'zerodha_secret': os.getenv('ZERODHA_SECRET'),
}