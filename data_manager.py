"""
Data Manager for the Nifty 50 Trading Bot
Handles data fetching, storage, and preprocessing
"""

import yfinance as yf
import pandas as pd
import numpy as np
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import ta

from config import TRADING_CONFIG, DATABASE_CONFIG, LOGGING_CONFIG

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


class DataManager:
    """Manages market data fetching, storage, and preprocessing"""
    
    def __init__(self):
        self.db_path = DATABASE_CONFIG['db_path']
        self.symbols = TRADING_CONFIG['index_symbols']
        self.timeframe = TRADING_CONFIG['timeframe']
        self.lookback_period = TRADING_CONFIG['lookback_period']
        self._setup_database()
    
    def _setup_database(self):
        """Initialize the database tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create market data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    timeframe TEXT NOT NULL,
                    UNIQUE(symbol, timestamp, timeframe)
                )
            """)
            
            # Create technical indicators table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS technical_indicators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    sma_20 REAL,
                    sma_50 REAL,
                    ema_12 REAL,
                    ema_26 REAL,
                    rsi_14 REAL,
                    macd REAL,
                    macd_signal REAL,
                    macd_histogram REAL,
                    bb_upper REAL,
                    bb_middle REAL,
                    bb_lower REAL,
                    adx REAL,
                    volume_sma REAL,
                    UNIQUE(symbol, timestamp)
                )
            """)
            
            conn.commit()
            conn.close()
            logger.info("Database setup completed successfully")
            
        except Exception as e:
            logger.error(f"Error setting up database: {e}")
            raise
    
    def fetch_market_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Fetch market data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=self.timeframe)
            
            if data.empty:
                logger.warning(f"No data fetched for {symbol}")
                return pd.DataFrame()
            
            # Clean and format data
            data = data.reset_index()
            data.columns = [col.lower() if col != 'Date' else 'timestamp' for col in data.columns]
            data['symbol'] = symbol
            data['timeframe'] = self.timeframe
            
            logger.info(f"Fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def save_market_data(self, data: pd.DataFrame):
        """Save market data to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Prepare data for insertion
            data_to_save = data[['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'timeframe']].copy()
            
            # Insert or replace data
            data_to_save.to_sql('market_data', conn, if_exists='append', index=False, method='ignore')
            
            conn.close()
            logger.info(f"Saved {len(data)} market data records")
            
        except Exception as e:
            logger.error(f"Error saving market data: {e}")
    
    def get_market_data(self, symbol: str, start_date: Optional[str] = None, 
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """Retrieve market data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = """
                SELECT * FROM market_data 
                WHERE symbol = ? AND timeframe = ?
            """
            params = [symbol, self.timeframe]
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
            
            query += " ORDER BY timestamp"
            
            data = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            if not data.empty:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data.set_index('timestamp', inplace=True)
            
            return data
            
        except Exception as e:
            logger.error(f"Error retrieving market data: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the given data"""
        try:
            if data.empty:
                return data
            
            df = data.copy()
            
            # Moving Averages
            df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
            df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
            df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
            
            # RSI
            df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_histogram'] = macd.macd_diff()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'])
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_lower'] = bb.bollinger_lband()
            
            # ADX
            df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
            
            # Volume indicators
            df['volume_sma'] = ta.volume.volume_sma(df['close'], df['volume'], window=20)
            
            logger.info("Technical indicators calculated successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return data
    
    def save_technical_indicators(self, data: pd.DataFrame, symbol: str):
        """Save technical indicators to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Prepare indicators data
            indicators_data = data.reset_index()[['timestamp', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
                                                'rsi_14', 'macd', 'macd_signal', 'macd_histogram',
                                                'bb_upper', 'bb_middle', 'bb_lower', 'adx', 'volume_sma']].copy()
            indicators_data['symbol'] = symbol
            
            # Remove rows with NaN values
            indicators_data = indicators_data.dropna()
            
            # Insert or replace data
            indicators_data.to_sql('technical_indicators', conn, if_exists='append', index=False, method='ignore')
            
            conn.close()
            logger.info(f"Saved {len(indicators_data)} technical indicator records")
            
        except Exception as e:
            logger.error(f"Error saving technical indicators: {e}")
    
    def update_all_data(self):
        """Update market data and technical indicators for all symbols"""
        try:
            for symbol in self.symbols:
                logger.info(f"Updating data for {symbol}")
                
                # Fetch new data
                market_data = self.fetch_market_data(symbol)
                
                if not market_data.empty:
                    # Save market data
                    self.save_market_data(market_data)
                    
                    # Calculate and save technical indicators
                    data_with_indicators = self.calculate_technical_indicators(market_data)
                    self.save_technical_indicators(data_with_indicators, symbol)
                
                logger.info(f"Data update completed for {symbol}")
            
            logger.info("All data updates completed successfully")
            
        except Exception as e:
            logger.error(f"Error updating data: {e}")
    
    def get_latest_data(self, symbol: str, periods: int = None) -> pd.DataFrame:
        """Get the latest market data with technical indicators"""
        try:
            if periods is None:
                periods = self.lookback_period
            
            # Get market data
            market_data = self.get_market_data(symbol)
            
            if market_data.empty:
                return pd.DataFrame()
            
            # Get latest periods
            latest_data = market_data.tail(periods).copy()
            
            # Calculate technical indicators
            latest_data = self.calculate_technical_indicators(latest_data)
            
            return latest_data
            
        except Exception as e:
            logger.error(f"Error getting latest data: {e}")
            return pd.DataFrame()
    
    def is_market_open(self) -> bool:
        """Check if the market is currently open"""
        now = datetime.now()
        
        # Check if it's a weekday (Monday = 0, Sunday = 6)
        if now.weekday() >= 5:  # Saturday or Sunday
            return False
        
        # Check trading hours (IST)
        start_time = datetime.strptime(TRADING_CONFIG['trading_hours']['start'], '%H:%M').time()
        end_time = datetime.strptime(TRADING_CONFIG['trading_hours']['end'], '%H:%M').time()
        
        current_time = now.time()
        return start_time <= current_time <= end_time


if __name__ == "__main__":
    # Test the DataManager
    dm = DataManager()
    dm.update_all_data()
    
    # Get latest data for Nifty 50
    latest_data = dm.get_latest_data('^NSEI')
    print(f"Latest data shape: {latest_data.shape}")
    print(latest_data.tail())