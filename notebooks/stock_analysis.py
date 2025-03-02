"""
# Stock Price Analysis and Visualization

This file provides exploratory data analysis of stock price data fetched from stocks.wajipk.com API. 
We'll visualize patterns, analyze technical indicators, and look for potential swing trading opportunities.
"""

# Import necessary libraries
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Add project root to path to import project modules
sys.path.append('..')
from src.data_collection import fetch_stock_data, save_stock_data
from src.preprocessing import add_technical_indicators, load_stock_data

# Set up visualization settings
plt.style.use('fivethirtyeight')
sns.set_theme(style='darkgrid')

"""
## Fetch Stock Data

Let's fetch historical data for a specific stock symbol. If you've already run the data collection step, 
you can load the data from the saved CSV file.
"""

# Set the stock symbol and time period
symbol = 'AAPL'  # Change to your desired stock symbol
days = 365  # One year of historical data

# Check if data already exists
file_path = f"../data/{symbol}_historical_data.csv"
if os.path.exists(file_path):
    print(f"Loading existing data for {symbol}...")
    df = load_stock_data(symbol)
else:
    print(f"Fetching new data for {symbol}...")
    df = fetch_stock_data(symbol, days=days)
    save_stock_data(df, symbol)

# Display the first few rows
print(df.head())

"""
## Exploratory Data Analysis

Let's explore the basic statistics and characteristics of the stock data.
"""

# Basic information about the dataset
print(f"Data shape: {df.shape}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print("\nData types:")
print(df.dtypes)
print("\nBasic statistics:")
print(df.describe())

"""
## Visualize Price History

Let's create some basic visualizations of the stock price history.
"""

# Plot price history
plt.figure(figsize=(14, 7))
plt.plot(df['date'], df['close'], label='Close Price')
plt.title(f'{symbol} Price History')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.savefig(f'../models/{symbol}_price_history.png')
plt.close()

# Daily returns
df['daily_return'] = df['close'].pct_change() * 100

plt.figure(figsize=(14, 7))
plt.plot(df['date'], df['daily_return'])
plt.title(f'{symbol} Daily Returns (%)')
plt.xlabel('Date')
plt.ylabel('Return (%)')
plt.axhline(y=0, color='r', linestyle='--')
plt.savefig(f'../models/{symbol}_daily_returns.png')
plt.close()

# Distribution of daily returns
plt.figure(figsize=(10, 6))
sns.histplot(df['daily_return'].dropna(), kde=True)
plt.title(f'{symbol} Distribution of Daily Returns')
plt.xlabel('Daily Return (%)')
plt.axvline(x=0, color='r', linestyle='--')
plt.savefig(f'../models/{symbol}_returns_distribution.png')
plt.close()

"""
## Add Technical Indicators

Now let's add technical indicators and visualize them.
"""

# Add technical indicators
df_tech = add_technical_indicators(df)

# Display first few rows with technical indicators
print(df_tech.head())

"""
## Visualize Moving Averages
"""

# Plot price with moving averages
plt.figure(figsize=(14, 7))
plt.plot(df_tech['date'], df_tech['close'], label='Close Price')
plt.plot(df_tech['date'], df_tech['MA5'], label='5-day MA')
plt.plot(df_tech['date'], df_tech['MA10'], label='10-day MA')
plt.plot(df_tech['date'], df_tech['MA20'], label='20-day MA')
plt.plot(df_tech['date'], df_tech['MA50'], label='50-day MA')
plt.title(f'{symbol} Price with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.savefig(f'../models/{symbol}_moving_averages.png')
plt.close()

"""
## Visualize Bollinger Bands
"""

# Plot Bollinger Bands
plt.figure(figsize=(14, 7))
plt.plot(df_tech['date'], df_tech['close'], label='Close Price')
plt.plot(df_tech['date'], df_tech['BB_upper'], label='Upper Band', color='green', alpha=0.7)
plt.plot(df_tech['date'], df_tech['BB_middle'], label='Middle Band', color='orange', alpha=0.7)
plt.plot(df_tech['date'], df_tech['BB_lower'], label='Lower Band', color='red', alpha=0.7)
plt.fill_between(df_tech['date'], df_tech['BB_upper'], df_tech['BB_lower'], alpha=0.1)
plt.title(f'{symbol} Bollinger Bands')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.savefig(f'../models/{symbol}_bollinger_bands.png')
plt.close()

"""
## Visualize RSI and MACD
"""

# Plot RSI
plt.figure(figsize=(14, 5))
plt.plot(df_tech['date'], df_tech['RSI'], color='purple')
plt.axhline(y=70, color='r', linestyle='--', alpha=0.5)
plt.axhline(y=30, color='g', linestyle='--', alpha=0.5)
plt.fill_between(df_tech['date'], df_tech['RSI'], 70, where=(df_tech['RSI']>=70), color='r', alpha=0.3)
plt.fill_between(df_tech['date'], df_tech['RSI'], 30, where=(df_tech['RSI']<=30), color='g', alpha=0.3)
plt.title(f'{symbol} Relative Strength Index (RSI)')
plt.xlabel('Date')
plt.ylabel('RSI')
plt.savefig(f'../models/{symbol}_rsi.png')
plt.close()

# Plot MACD
plt.figure(figsize=(14, 5))
plt.plot(df_tech['date'], df_tech['MACD'], label='MACD', color='blue')
plt.plot(df_tech['date'], df_tech['MACD_signal'], label='Signal Line', color='red')
plt.bar(df_tech['date'], df_tech['MACD'] - df_tech['MACD_signal'], label='Histogram', alpha=0.3)
plt.title(f'{symbol} Moving Average Convergence Divergence (MACD)')
plt.xlabel('Date')
plt.ylabel('MACD')
plt.legend()
plt.savefig(f'../models/{symbol}_macd.png')
plt.close()

"""
## Correlation between Technical Indicators
"""

# Select technical indicators for correlation analysis
tech_indicators = ['close', 'MA5', 'MA10', 'MA20', 'MA50', 'RSI', 'MACD', 'MACD_signal', 
                   'BB_upper', 'BB_middle', 'BB_lower', 'volatility', 'price_roc', 'momentum']

# Compute correlation matrix
corr_matrix = df_tech[tech_indicators].corr()

# Plot correlation heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
plt.title(f'{symbol} Technical Indicators Correlation')
plt.tight_layout()
plt.savefig(f'../models/{symbol}_correlation_matrix.png')
plt.close()

"""
## Identify Potential Swing Trading Signals

Let's look for potential swing trading signals based on common patterns.
"""

# Create a function to identify potential swing trading signals
def identify_swing_signals(df):
    signals = pd.DataFrame(index=df.index)
    signals['date'] = df['date']
    signals['price'] = df['close']
    
    # RSI oversold (<30) and overbought (>70) signals
    signals['rsi_oversold'] = (df['RSI'] < 30).astype(int)
    signals['rsi_overbought'] = (df['RSI'] > 70).astype(int)
    
    # MACD crossover signals
    signals['macd_cross_above'] = ((df['MACD'] > df['MACD_signal']) & 
                                  (df['MACD'].shift(1) <= df['MACD_signal'].shift(1))).astype(int)
    signals['macd_cross_below'] = ((df['MACD'] < df['MACD_signal']) & 
                                  (df['MACD'].shift(1) >= df['MACD_signal'].shift(1))).astype(int)
    
    # Bollinger Band breakout signals
    signals['bb_upper_breakout'] = (df['close'] > df['BB_upper']).astype(int)
    signals['bb_lower_breakout'] = (df['close'] < df['BB_lower']).astype(int)
    
    # Moving Average crossover signals
    signals['ma5_cross_above_ma20'] = ((df['MA5'] > df['MA20']) & 
                                      (df['MA5'].shift(1) <= df['MA20'].shift(1))).astype(int)
    signals['ma5_cross_below_ma20'] = ((df['MA5'] < df['MA20']) & 
                                      (df['MA5'].shift(1) >= df['MA20'].shift(1))).astype(int)
    
    # Combine buy signals
    signals['buy_signal'] = ((signals['rsi_oversold'] == 1) | 
                            (signals['macd_cross_above'] == 1) | 
                            (signals['bb_lower_breakout'] == 1) |
                            (signals['ma5_cross_above_ma20'] == 1)).astype(int)
    
    # Combine sell signals
    signals['sell_signal'] = ((signals['rsi_overbought'] == 1) | 
                             (signals['macd_cross_below'] == 1) | 
                             (signals['bb_upper_breakout'] == 1) |
                             (signals['ma5_cross_below_ma20'] == 1)).astype(int)
    
    return signals

# Identify signals
signals = identify_swing_signals(df_tech)

# Display recent signals
recent_signals = signals.tail(30)
print("Recent Buy Signals:")
buy_dates = recent_signals[recent_signals['buy_signal'] == 1][['date', 'price']]
if len(buy_dates) > 0:
    print(buy_dates)
else:
    print("No recent buy signals")
    
print("\nRecent Sell Signals:")
sell_dates = recent_signals[recent_signals['sell_signal'] == 1][['date', 'price']]
if len(sell_dates) > 0:
    print(sell_dates)
else:
    print("No recent sell signals")

"""
## Visualize Signals on Price Chart
"""

# Plot price with buy and sell signals
plt.figure(figsize=(14, 7))

# Plot price
plt.plot(df_tech['date'], df_tech['close'], label='Close Price', alpha=0.6)

# Plot buy signals
buy_signals = signals[signals['buy_signal'] == 1]
if len(buy_signals) > 0:
    plt.scatter(buy_signals['date'], buy_signals['price'], color='green', label='Buy Signal', 
                marker='^', s=100)

# Plot sell signals
sell_signals = signals[signals['sell_signal'] == 1]
if len(sell_signals) > 0:
    plt.scatter(sell_signals['date'], sell_signals['price'], color='red', label='Sell Signal', 
                marker='v', s=100)

plt.title(f'{symbol} Price with Swing Trading Signals')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.savefig(f'../models/{symbol}_trading_signals.png')
plt.close()

"""
## Backtesting a Simple Swing Trading Strategy

Let's backtest a simple strategy based on our signals.
"""

def backtest_strategy(df, signals, initial_capital=10000.0):
    # Create a portfolio dataframe
    portfolio = pd.DataFrame(index=signals.index)
    portfolio['date'] = signals['date']
    portfolio['price'] = signals['price']
    portfolio['holdings'] = 0.0
    portfolio['cash'] = initial_capital
    portfolio['total'] = initial_capital
    
    # Initialize signals
    position = 0  # 0 = no position, 1 = long position
    entry_price = 0
    shares = 0
    
    # Loop through the signals
    for i in range(len(portfolio)):
        # Update portfolio value for current day
        if i > 0:
            portfolio.loc[portfolio.index[i], 'holdings'] = shares * portfolio.loc[portfolio.index[i], 'price']
            portfolio.loc[portfolio.index[i], 'cash'] = portfolio.loc[portfolio.index[i-1], 'cash']
            portfolio.loc[portfolio.index[i], 'total'] = portfolio.loc[portfolio.index[i], 'holdings'] + portfolio.loc[portfolio.index[i], 'cash']
        
        # Check for buy signal
        if signals.loc[signals.index[i], 'buy_signal'] == 1 and position == 0:
            # Calculate number of shares to buy (use 95% of cash to account for fees)
            price = portfolio.loc[portfolio.index[i], 'price']
            cash = portfolio.loc[portfolio.index[i], 'cash'] * 0.95
            shares = int(cash // price)  # Integer division to get whole number of shares
            
            if shares > 0:
                # Update position
                position = 1
                entry_price = price
                
                # Update portfolio for current day
                portfolio.loc[portfolio.index[i], 'cash'] -= shares * price
                portfolio.loc[portfolio.index[i], 'holdings'] = shares * price
                portfolio.loc[portfolio.index[i], 'total'] = portfolio.loc[portfolio.index[i], 'holdings'] + portfolio.loc[portfolio.index[i], 'cash']
                
                print(f"BUY: {signals.loc[signals.index[i], 'date'].date()} - {shares} shares @ ${price:.2f} = ${shares * price:.2f}")
        
        # Check for sell signal
        elif signals.loc[signals.index[i], 'sell_signal'] == 1 and position == 1:
            # Sell all shares
            price = portfolio.loc[portfolio.index[i], 'price']
            sale_amount = shares * price
            profit = sale_amount - (shares * entry_price)
            profit_pct = (price / entry_price - 1) * 100
            
            # Update position
            position = 0
            
            # Update portfolio for current day
            portfolio.loc[portfolio.index[i], 'cash'] += sale_amount
            portfolio.loc[portfolio.index[i], 'holdings'] = 0
            portfolio.loc[portfolio.index[i], 'total'] = portfolio.loc[portfolio.index[i], 'holdings'] + portfolio.loc[portfolio.index[i], 'cash']
            
            print(f"SELL: {signals.loc[signals.index[i], 'date'].date()} - {shares} shares @ ${price:.2f} = ${sale_amount:.2f}, Profit: ${profit:.2f} ({profit_pct:.2f}%)")
            shares = 0
    
    # Calculate strategy performance
    portfolio['returns'] = portfolio['total'].pct_change()
    portfolio['cumulative_returns'] = (1 + portfolio['returns']).cumprod() - 1
    
    # Calculate benchmark performance (buy and hold)
    benchmark_shares = int(initial_capital // portfolio.loc[portfolio.index[0], 'price'])
    portfolio['benchmark'] = benchmark_shares * portfolio['price'] + (initial_capital - benchmark_shares * portfolio.loc[portfolio.index[0], 'price'])
    portfolio['benchmark_returns'] = portfolio['benchmark'].pct_change()
    portfolio['benchmark_cumulative_returns'] = (1 + portfolio['benchmark_returns']).cumprod() - 1
    
    # Print strategy performance
    total_return = portfolio['total'].iloc[-1] / initial_capital - 1
    benchmark_return = portfolio['benchmark'].iloc[-1] / initial_capital - 1
    print(f"\nStrategy Performance:")
    print(f"Initial Capital: ${initial_capital:.2f}")
    print(f"Final Portfolio Value: ${portfolio['total'].iloc[-1]:.2f}")
    print(f"Total Return: {total_return*100:.2f}%")
    print(f"Benchmark Return (Buy & Hold): {benchmark_return*100:.2f}%")
    
    return portfolio

# Run the backtest
portfolio = backtest_strategy(df_tech, signals)

# Plot performance comparison
plt.figure(figsize=(14, 7))
plt.plot(portfolio['date'], portfolio['cumulative_returns'] * 100, label='Strategy Returns', color='blue')
plt.plot(portfolio['date'], portfolio['benchmark_cumulative_returns'] * 100, label='Buy & Hold Returns', color='green', alpha=0.7)
plt.title(f'{symbol} Strategy vs Buy & Hold Performance')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns (%)')
plt.legend()
plt.tight_layout()
plt.savefig(f'../models/{symbol}_strategy_performance.png')
plt.close()

"""
## Conclusions and Next Steps

Based on our analysis, here are some conclusions and next steps for our stock prediction model:

1. **Pattern Identification**: We've identified common technical patterns that could be useful for predicting price movements.
2. **Feature Engineering**: The technical indicators we've calculated can serve as important features for our machine learning model.
3. **Model Training**: The next step is to use this historical data with technical indicators to train our LSTM model.
4. **Trading Strategy**: We've backtested a simple trading strategy that we can refine based on our model's predictions.
5. **Future Improvements**: 
   - Incorporate sentiment analysis from news and social media
   - Include more advanced technical indicators
   - Optimize model hyperparameters
   - Implement risk management rules
"""

print("\nAnalysis completed. Check the models directory for visualization results.") 