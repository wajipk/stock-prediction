import os
import requests
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time

def fetch_market_index(index_symbol='KSE100', days=30):
    """
    Fetch market index data (e.g., KSE100, KSE30) from stocks.wajipk.com or alternative APIs
    
    Args:
        index_symbol (str): Market index symbol (default: KSE100)
        days (int): Number of days of historical data to fetch
        
    Returns:
        pd.DataFrame: DataFrame containing historical index data or None if unavailable
    """
    print(f"Fetching market index data for {index_symbol} for the last {days} days...")
    
    # API endpoint - using the trades endpoint as specified
    url = f"https://stocks.wajipk.com/api/indices-trades?symbol={index_symbol}"
    
    try:
        # Make API request
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Parse JSON response
        data = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        if len(df) == 0:
            print(f"Warning: No data returned from API for {index_symbol}")
            print(f"Skipping market trend analysis due to missing {index_symbol} data")
            return None
        
        # Handle the columns as in the fetch_stock_data function
        column_mapping = {
            'displaydate': 'date',
            'value': 'close',
            'price': 'close',  # Alternative name for close price
            'open_price': 'open',
            'high_price': 'high',
            'low_price': 'low',
            'volume': 'volume'
        }
        
        # Apply column renaming for columns that exist
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        # Ensure required columns exist with fallback values if needed
        if 'close' not in df.columns and 'price' in df.columns:
            df['close'] = df['price']
        elif 'close' not in df.columns:
            print(f"Warning: No 'close' price column in API response for {index_symbol}")
            print(f"Skipping market trend analysis due to missing price data")
            return None
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Convert 'close' column to float
        try:
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            # Drop rows with NaN values after conversion
            na_count = df['close'].isna().sum()
            if na_count > 0:
                print(f"Warning: {na_count} rows had non-numeric values in 'close' column and were dropped")
                df = df.dropna(subset=['close']).reset_index(drop=True)
            
            if len(df) == 0:
                print(f"Warning: All data was dropped due to non-numeric 'close' values")
                return None
        except Exception as e:
            print(f"Warning: Error converting 'close' column to numeric: {e}")
            return None
        
        # Sort by date in ascending order (oldest first, newest last)
        df = df.sort_values('date', ascending=True).reset_index(drop=True)
        
        # Limit to the requested number of days
        if len(df) > days:
            df = df.tail(days).reset_index(drop=True)
        
        return df
    
    except Exception as e:
        print(f"Warning: Error fetching market index data for {index_symbol}: {e}")
        print(f"Skipping market trend analysis due to data unavailability")
        return None

def fetch_sector_data(sector_name, days=30):
    """
    Fetch sector index data (if available via API)
    
    Args:
        sector_name (str): Sector name or symbol
        days (int): Number of days of historical data to fetch
        
    Returns:
        pd.DataFrame: DataFrame containing sector index data
    """
    print(f"Attempting to fetch sector data for {sector_name}...")
    # This function would need to be implemented according to the available API
    # or data source that provides sector information.
    
    # For demonstration, we're returning None for now
    print("Sector-specific data fetching is not yet implemented")
    return None

def calculate_market_trend(market_df, window=5):
    """
    Calculate market trend indicators
    
    Args:
        market_df (pd.DataFrame): DataFrame with market index data
        window (int): Window size for calculating trends
        
    Returns:
        dict: Dictionary containing market trend indicators
    """
    if market_df is None or len(market_df) < window+1:
        print("Not enough market data to calculate trends")
        return {
            'market_direction': 'unknown',
            'market_strength': 0,
            'market_momentum': 0,
            'market_volatility': 0,
            'recent_performance': 0
        }
    
    # Ensure close column is numeric
    try:
        market_df['close'] = market_df['close'].astype(float)
    except Exception as e:
        print(f"Warning: Error converting 'close' column to float: {e}")
        print("Returning default market trend values")
        return {
            'market_direction': 'unknown',
            'market_strength': 0,
            'market_momentum': 0,
            'market_volatility': 0,
            'recent_performance': 0
        }
    
    # Calculate market direction (bullish/bearish)
    recent_change = (market_df['close'].iloc[-1] / market_df['close'].iloc[-window-1] - 1) * 100
    market_direction = 'bullish' if recent_change > 0 else 'bearish'
    
    # Calculate market strength (magnitude of trend)
    market_strength = abs(recent_change)
    
    # Calculate market momentum (rate of change)
    market_df['daily_return'] = market_df['close'].pct_change() * 100
    momentum = market_df['daily_return'].iloc[-window:].mean()
    
    # Calculate volatility
    volatility = market_df['daily_return'].iloc[-window:].std()
    
    # Calculate recent performance (last day)
    recent_performance = market_df['daily_return'].iloc[-1]
    
    return {
        'market_direction': market_direction,
        'market_strength': market_strength,
        'market_momentum': momentum,
        'market_volatility': volatility,
        'recent_performance': recent_performance
    }

def get_market_sentiment_score(market_trend):
    """
    Calculate a market sentiment score from trend indicators
    
    Args:
        market_trend (dict): Dictionary of market trend indicators
        
    Returns:
        float: Market sentiment score (-1 to 1, where 1 is very bullish)
    """
    # Default neutral sentiment if trend data is not available
    if not market_trend or market_trend['market_direction'] == 'unknown':
        return 0.0
    
    # Base score from direction
    base_score = 0.5 if market_trend['market_direction'] == 'bullish' else -0.5
    
    # Adjust based on strength (0-10% change scales to 0-0.3 adjustment)
    strength_adj = min(0.3, market_trend['market_strength'] / 30)
    
    # Adjust based on momentum (-3 to +3% daily change scales to -0.1 to +0.1)
    momentum_adj = np.clip(market_trend['market_momentum'] / 30, -0.1, 0.1)
    
    # Adjust based on volatility (negative impact)
    volatility_adj = -min(0.1, market_trend['market_volatility'] / 30)
    
    # Calculate final score and clip to (-1, 1) range
    sentiment_score = np.clip(base_score + strength_adj + momentum_adj + volatility_adj, -1.0, 1.0)
    
    return sentiment_score

def adjust_prediction_for_market_trend(predicted_prices, market_sentiment_score, adjustment_factor=0.03):
    """
    Adjust predicted prices based on overall market trend
    
    Args:
        predicted_prices (list or np.array): List or array of predicted prices
        market_sentiment_score (float): Market sentiment score (-1 to 1)
        adjustment_factor (float): Maximum percentage adjustment to apply
        
    Returns:
        list or np.array: Adjusted predicted prices
    """
    # Check if predicted_prices is None or empty
    if predicted_prices is None or (isinstance(predicted_prices, list) and len(predicted_prices) == 0) or \
       (isinstance(predicted_prices, np.ndarray) and predicted_prices.size == 0) or market_sentiment_score == 0:
        return predicted_prices
    
    # Calculate adjustment percentage (market_sentiment_score * adjustment_factor)
    # For example: 0.8 sentiment * 0.03 = 2.4% upward adjustment
    adjustment_percentage = market_sentiment_score * adjustment_factor
    
    # Apply adjustment to each prediction
    # If it's a numpy array, we can do it directly
    if isinstance(predicted_prices, np.ndarray):
        return predicted_prices * (1 + adjustment_percentage)
    else:
        # If it's a list, iterate through it
        adjusted_prices = []
        for price in predicted_prices:
            adjusted_price = price * (1 + adjustment_percentage)
            adjusted_prices.append(adjusted_price)
        return adjusted_prices

def get_market_trend_analysis(stock_symbol, days=30):
    """
    Analyze market trends and return a comprehensive analysis
    
    Args:
        stock_symbol (str): Stock symbol for sector determination
        days (int): Number of days for analysis
        
    Returns:
        tuple: (market_trends, sentiment_score) or (None, 0.0) if market data unavailable
    """
    # 1. Fetch market index data
    market_df = fetch_market_index(days=days)
    
    # If no market data is available, return None for trends and neutral for sentiment
    if market_df is None:
        print("Market trend analysis skipped: No market index data available")
        return None, 0.0
    
    # Check if we have enough data points
    if len(market_df) < 5:  # Need at least a few days for meaningful analysis
        print(f"Market trend analysis skipped: Insufficient data points ({len(market_df)} available, need at least 5)")
        return None, 0.0
    
    # 2. Calculate market trends
    market_trends = calculate_market_trend(market_df)
    
    # 3. Get sentiment score
    sentiment_score = get_market_sentiment_score(market_trends) if market_trends else 0.0
    
    # 4. Log the analysis
    if market_trends:
        print(f"\nMarket Trend Analysis:")
        print(f"- Direction: {market_trends['market_direction']}")
        print(f"- Strength: {market_trends['market_strength']:.2f}%")
        print(f"- Momentum: {market_trends['market_momentum']:.2f}%")
        print(f"- Volatility: {market_trends['market_volatility']:.2f}%")
        print(f"- Recent Performance: {market_trends['recent_performance']:.2f}%")
        print(f"- Overall Market Sentiment Score: {sentiment_score:.2f}")
    else:
        print("Could not perform market trend analysis due to missing data")
    
    return market_trends, sentiment_score 