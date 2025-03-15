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
            'market_direction': 'sideways',  # Default to sideways when no data
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
            'market_direction': 'sideways',  # Default to sideways on error
            'market_strength': 0,
            'market_momentum': 0,
            'market_volatility': 0,
            'recent_performance': 0
        }
    
    # Calculate market direction with more nuanced classification
    recent_change = (market_df['close'].iloc[-1] / market_df['close'].iloc[-window-1] - 1) * 100
    
    # Define thresholds for sideways market (between -1.5% and +1.5% over the window period)
    sideways_threshold = 1.5
    
    if abs(recent_change) <= sideways_threshold:
        market_direction = 'sideways'
    else:
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
    
    # Check for range-bound trading (sideways with higher volatility)
    max_price = market_df['close'].iloc[-window:].max()
    min_price = market_df['close'].iloc[-window:].min()
    price_range_pct = (max_price - min_price) / min_price * 100
    
    # If price stayed within a narrow range but had significant oscillations, it's sideways
    if price_range_pct < 3.0 and volatility > 0.5:
        market_direction = 'sideways'
        print(f"Detected range-bound trading: {price_range_pct:.2f}% range with {volatility:.2f}% volatility")
    
    # Check for mean reversion pattern (significant reversals)
    reversals = 0
    direction = None
    for day in range(-window + 1, 0):
        current_direction = 'up' if market_df['daily_return'].iloc[day] > 0 else 'down'
        if direction is not None and current_direction != direction:
            reversals += 1
        direction = current_direction
    
    # High number of reversals indicates sideways/choppy market
    if reversals >= window * 0.6:  # If more than 60% of days showed reversals
        market_direction = 'sideways'
        print(f"Detected choppy market with {reversals} direction reversals in {window} days")
    
    print(f"Market trend: {market_direction.upper()} (change: {recent_change:.2f}%, strength: {market_strength:.2f}%, volatility: {volatility:.2f}%)")
    
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
    if not market_trend:
        print("Market trend data unavailable, using neutral sentiment (0.0)")
        return 0.0
    
    # Handle sideways market differently - force near-neutral sentiment
    if market_trend['market_direction'] == 'sideways':
        # Calculate a small sentiment adjustment based on recent momentum
        # But keep it very close to neutral
        momentum_adj = np.clip(market_trend['market_momentum'] / 50, -0.05, 0.05)
        sentiment_score = momentum_adj
        
        print(f"Sideways market detected - using near-neutral sentiment: {sentiment_score:.3f}")
        return sentiment_score
    
    # Base score from direction - more conservative approach
    if market_trend['market_direction'] == 'bullish':
        base_score = 0.15  # Reduced from 0.3 to be much less aggressive
    elif market_trend['market_direction'] == 'bearish':
        base_score = -0.15  # Reduced from -0.3 to be much less aggressive
    else:
        base_score = 0.0  # Neutral
    
    # Adjust based on strength (0-10% change scales to 0-0.10 adjustment)
    # Reduced from 0.2 to 0.1 to be much less aggressive
    strength_adj = min(0.1, market_trend['market_strength'] / 100)
    
    # Adjust based on momentum (-3 to +3% daily change scales to -0.05 to +0.05)
    # More conservative adjustment
    momentum_adj = np.clip(market_trend['market_momentum'] / 60, -0.05, 0.05)
    
    # Adjust based on volatility (negative impact) - increased impact
    volatility_adj = -min(0.15, market_trend['market_volatility'] / 25)
    
    # Calculate final score and clip to a narrower range
    sentiment_score = np.clip(base_score + strength_adj + momentum_adj + volatility_adj, -0.4, 0.4)
    
    # Apply mean reversion factor - markets tend to revert to mean
    # If market has been strongly trending one way, reduce sentiment in that direction
    if (market_trend['market_direction'] == 'bullish' and market_trend['market_strength'] > 5) or \
       (market_trend['market_direction'] == 'bearish' and market_trend['market_strength'] > 5):
        # Apply stronger mean reversion when market has moved significantly
        reversion_factor = min(0.5, market_trend['market_strength'] / 30)
        sentiment_score *= (1 - reversion_factor)
        print(f"Applied mean reversion factor of {reversion_factor:.2f} due to strong {market_trend['market_direction']} trend")
    
    print(f"Market sentiment calculated: {sentiment_score:.3f} (base: {base_score:.2f}, " + 
          f"strength: {strength_adj:.3f}, momentum: {momentum_adj:.3f}, volatility: {volatility_adj:.3f})")
    
    return sentiment_score

def adjust_prediction_for_market_trend(predicted_prices, market_sentiment_score, adjustment_factor=0.03, previous_close=None, market_trends=None):
    """
    Adjust stock price predictions based on market sentiment and trends
    
    Args:
        predicted_prices: List or numpy array of predicted prices
        market_sentiment_score: The market sentiment score (-1 to 1)
        adjustment_factor: How much to adjust predictions based on market sentiment
        previous_close: Previous closing price (if available)
        market_trends: Market trend information (if available)
        
    Returns:
        adjusted_prices: Adjusted predicted prices
    """
    if predicted_prices is None or (isinstance(predicted_prices, list) and len(predicted_prices) == 0) or \
       (isinstance(predicted_prices, np.ndarray) and predicted_prices.size == 0):
        return predicted_prices
    
    # Extract market direction if available
    market_direction = 'unknown'
    if market_trends is not None and isinstance(market_trends, dict) and 'market_direction' in market_trends:
        market_direction = market_trends['market_direction']
    
    # For sideways markets, enforce strong mean reversion
    if market_direction == 'sideways':
        print(f"Sideways market detected - applying strong mean reversion")
        market_sentiment_score *= 0.3  # Reduce sentiment impact in sideways markets
        
        # Check if predicted_prices is a single value or an array
        if previous_close is not None:
            # Handle case where predicted_prices is a single value (float or numpy.float64)
            if isinstance(predicted_prices, (float, np.float64, np.float32, int, np.int64, np.int32)):
                # For a single value, just apply a simple adjustment
                pct_change = (predicted_prices / previous_close) - 1
                reversion_strength = 0.3  # Use a moderate reversion strength for single values
                # Apply mean reversion: pull the prediction back toward the previous close
                predicted_prices = previous_close * (1 + pct_change * (1 - reversion_strength))
            elif hasattr(predicted_prices, '__len__') and len(predicted_prices) > 0:
                # Original code for array of predictions
                # Convert to numpy array for easier manipulation
                prices_array = np.array(predicted_prices) if not isinstance(predicted_prices, np.ndarray) else predicted_prices.copy()
                
                # Calculate percentage change from previous close
                pct_changes = (prices_array / previous_close) - 1
                
                # Apply stronger mean reversion for later days in sideways markets
                for i in range(len(prices_array)):
                    # Increasing mean reversion strength for later days (0.1 to 0.5)
                    reversion_strength = min(0.1 + (i * 0.08), 0.5)
                    
                    # If price is going up, pull it down; if going down, pull it up
                    if pct_changes[i] > 0:
                        # Pull positive changes back toward zero
                        pct_changes[i] *= (1 - reversion_strength)
                    else:
                        # Pull negative changes back toward zero
                        pct_changes[i] *= (1 - reversion_strength)
                
                # Convert back to prices
                mean_reverted_prices = previous_close * (1 + pct_changes)
                
                # Return immediately with mean-reverted prices
                print(f"Applied mean reversion in sideways market, reducing trend magnitude")
                
                # Return the appropriate type
                if isinstance(predicted_prices, list):
                    return mean_reverted_prices.tolist()
                return mean_reverted_prices
    
    # If market sentiment is neutral or nearly neutral, make minimal adjustments
    if abs(market_sentiment_score) < 0.1:
        print(f"Market sentiment is near neutral ({market_sentiment_score:.2f}), applying minimal adjustment")
        return predicted_prices
    
    # Apply sentiment score to adjust prediction
    # Positive sentiment -> higher prediction, negative sentiment -> lower prediction
    adjustment = predicted_prices * market_sentiment_score * adjustment_factor
    
    # Apply the adjustment
    adjusted_prediction = predicted_prices + adjustment
    
    # Ensure we don't predict negative prices
    if isinstance(adjusted_prediction, (list, np.ndarray)):
        adjusted_prediction = np.maximum(adjusted_prediction, 0.01)
    else:
        adjusted_prediction = max(adjusted_prediction, 0.01)
    
    # Log the adjustment
    if isinstance(predicted_prices, (list, np.ndarray)) and len(predicted_prices) > 0:
        avg_original = np.mean(predicted_prices)
        avg_adjusted = np.mean(adjusted_prediction)
        print(f"Market sentiment adjustment: {market_sentiment_score:.4f} -> Price adjustment: {(avg_adjusted - avg_original):.2f} ({(avg_adjusted/avg_original - 1)*100:.2f}%)")
    else:
        # For single value
        print(f"Market sentiment adjustment: {market_sentiment_score:.4f} -> Price adjustment: {(adjusted_prediction - predicted_prices):.2f} ({(adjusted_prediction/predicted_prices - 1)*100:.2f}%)")
    
    return adjusted_prediction

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