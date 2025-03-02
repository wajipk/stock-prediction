import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
from src.rules import default_rules, adjust_stock_data, fetch_and_load_payouts


def load_stock_data(symbol, apply_rules=True):
    """
    Load stock data from CSV file
    
    Args:
        symbol (str): Stock symbol
        apply_rules (bool): Whether to apply financial rules to the data
        
    Returns:
        pd.DataFrame: DataFrame containing stock data
    """
    # Check for data in company-specific directory first
    company_dir = os.path.join('data', symbol)
    company_file_path = os.path.join(company_dir, "historical_data.csv")
    
    if os.path.exists(company_file_path):
        file_path = company_file_path
    else:
        # Fallback to old path format for backward compatibility
        file_path = f"data/{symbol}_historical_data.csv"
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        return None
    df = pd.read_csv(file_path)
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Ensure data is sorted by date
    df = df.sort_values('date', ascending=True).reset_index(drop=True)
    
    # Print the last date after loading the data
    last_date = df['date'].iloc[-1]
    print(f"Last date in the dataset: {last_date.strftime('%Y-%m-%d')}")
    
    # Filter out future dates
    from datetime import datetime
    current_date = datetime.now().date()
    future_data_count = len(df[df['date'].dt.date > current_date])
    if future_data_count > 0:
        print(f"Removing {future_data_count} records with future dates from dataset")
        df = df[df['date'].dt.date <= current_date].reset_index(drop=True)
    
    print(f"Loaded {len(df)} records for {symbol}")
    # Apply financial rules if requested
    if apply_rules:
        # Try to fetch payout data from API
        fetch_and_load_payouts(symbol)
        
        # Apply the rules
        print("Applying financial rules to adjust stock prices...")
        df = adjust_stock_data(df, symbol)
    
    return df


def get_indicators_file_path(symbol):
    """
    Get the path to the technical indicators file
    
    Args:
        symbol (str): Stock symbol
        
    Returns:
        str: Path to the technical indicators file
    """
    company_dir = os.path.join('data', symbol)
    os.makedirs(company_dir, exist_ok=True)
    return os.path.join(company_dir, "technical_indicators.csv")


def load_or_calculate_technical_indicators(df, symbol):
    """
    Load technical indicators from CSV if available, otherwise calculate
    and save them for future use
    
    Args:
        df (pd.DataFrame): DataFrame containing stock data
        symbol (str): Stock symbol
        
    Returns:
        pd.DataFrame: DataFrame with added technical indicators
    """
    try:
        # Check if the dataset is too small
        if df is None or len(df) == 0:
            print("No data to process")
            return pd.DataFrame()
            
        # For extremely small datasets (1-2 rows), just add basic indicators and return
        if len(df) <= 2:
            print(f"Dataset has only {len(df)} rows, using basic indicators only")
            indicators_df = add_basic_indicators(df)
            
            # Save the indicators
            indicators_file = get_indicators_file_path(symbol)
            indicators_df.to_csv(indicators_file, index=False)
            print(f"Basic indicators for small dataset saved to {indicators_file}")
            
            return indicators_df
            
        indicators_file = get_indicators_file_path(symbol)
        
        # Check if the latest data in the dataframe
        latest_date = df['date'].max()
        
        # Check if indicators file exists
        if os.path.exists(indicators_file):
            print(f"Loading technical indicators from {indicators_file}")
            indicators_df = pd.read_csv(indicators_file)
            
            # Convert date to datetime
            indicators_df['date'] = pd.to_datetime(indicators_df['date'])
            
            # Check if indicators are up to date with our stock data
            if indicators_df['date'].max() >= latest_date:
                print("Technical indicators are up to date")
                return indicators_df
            else:
                print("Technical indicators need updating with newer data")
                # Only process the new data
                last_indicator_date = indicators_df['date'].max()
                new_data = df[df['date'] > last_indicator_date].copy()
                
                # If no new data to process, return existing indicators
                if len(new_data) == 0:
                    print("No new data to process, returning existing indicators")
                    return indicators_df
                
                # Calculate indicators for new data
                try:
                    new_indicators = add_advanced_technical_indicators(new_data)
                    
                    # If new_indicators is None or empty, use basic indicators
                    if new_indicators is None or len(new_indicators) == 0:
                        print("Advanced indicators calculation failed, using basic indicators")
                        new_indicators = add_basic_indicators(new_data)
                    
                    # Combine old and new indicators
                    updated_indicators = pd.concat([indicators_df, new_indicators], ignore_index=True)
                    updated_indicators = updated_indicators.drop_duplicates(subset=['date']).reset_index(drop=True)
                    
                    # Save the updated indicators
                    updated_indicators.to_csv(indicators_file, index=False)
                    print(f"Updated technical indicators saved to {indicators_file}")
                    
                    return updated_indicators
                except Exception as e:
                    print(f"Error updating indicators: {e}")
                    print("Falling back to using existing indicators")
                    return indicators_df
        else:
            # Calculate indicators for all data and save
            print(f"Calculating technical indicators for {symbol}...")
            try:
                indicators_df = add_advanced_technical_indicators(df)
                
                # If indicators_df is None or empty, use basic indicators
                if indicators_df is None or len(indicators_df) == 0:
                    print("Advanced indicators calculation failed, using basic indicators")
                    indicators_df = add_basic_indicators(df)
                
                # Save the indicators
                indicators_df.to_csv(indicators_file, index=False)
                print(f"Technical indicators saved to {indicators_file}")
                
                return indicators_df
            except Exception as e:
                print(f"Error calculating technical indicators: {e}")
                print("Attempting fallback to basic indicators...")
                
                # Try a more basic version with just essential indicators
                indicators_df = add_basic_indicators(df)
                
                # Save the basic indicators
                indicators_df.to_csv(indicators_file, index=False)
                print(f"Basic indicators saved to {indicators_file}")
                
                return indicators_df
    except Exception as e:
        print(f"Error in load_or_calculate_technical_indicators: {e}")
        # Final fallback - just add the basic indicators without saving
        return add_basic_indicators(df)


def add_basic_indicators(df):
    """
    Add only the most basic technical indicators that don't require TA-Lib
    This is a fallback function when the advanced indicators calculation fails
    
    Args:
        df (pd.DataFrame): DataFrame containing stock data
        
    Returns:
        pd.DataFrame: DataFrame with added basic technical indicators
    """
    print("Adding basic technical indicators only...")
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Store original date column for preserving all data
    dates = df['date'].copy()
    
    # Basic Moving Averages - use smaller windows for limited data
    df['MA5'] = df['close'].rolling(window=min(5, len(df) // 2 if len(df) > 4 else 2)).mean()
    df['MA10'] = df['close'].rolling(window=min(10, len(df) // 2 if len(df) > 6 else 3)).mean()
    df['MA20'] = df['close'].rolling(window=min(20, len(df) // 2 if len(df) > 10 else 5)).mean()
    df['MA50'] = df['close'].rolling(window=min(50, len(df) // 3 if len(df) > 15 else 7)).mean()
    
    # Exponential Moving Averages
    df['EMA10'] = df['close'].ewm(span=min(10, len(df) // 2 if len(df) > 6 else 3), adjust=False).mean()
    df['EMA20'] = df['close'].ewm(span=min(20, len(df) // 2 if len(df) > 10 else 5), adjust=False).mean()
    
    # Simple RSI - adjust window size for small datasets
    rsi_window = min(14, len(df) // 3 if len(df) > 20 else 5)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=rsi_window).mean()
    avg_loss = loss.rolling(window=rsi_window).mean()
    rs = avg_gain / avg_loss.replace(0, 0.001)  # Avoid division by zero
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Price change
    df['price_change'] = df['close'].pct_change()
    
    # Volume change
    df['volume_change'] = df['volume'].pct_change()
    
    # Add placeholders for TA-Lib indicators to ensure compatibility
    talib_indicators = ['TEMA', 'DEMA', 'MOM', 'PPO', 'HT_DCPERIOD', 
                      'CDL_DOJI', 'CDL_HAMMER', 'CDL_ENGULFING']
    
    for indicator in talib_indicators:
        if indicator in ['CDL_DOJI', 'CDL_HAMMER', 'CDL_ENGULFING']:
            df[indicator] = 0
        elif indicator == 'HT_DCPERIOD':
            df[indicator] = 20
        else:
            df[indicator] = df['close'].rolling(window=min(20, len(df) // 2 if len(df) > 10 else 5)).mean()
    
    # Add calendar features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    
    # Handle NaN values - forward fill first, then backward fill, then fill with zeros
    # This ensures we don't lose any rows
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # CRITICAL: If we somehow still have NaNs (unlikely), replace them with zeros instead of dropping rows
    if df.isna().any().any():
        print(f"Warning: Still found NaN values in data after filling methods. Replacing with zeros.")
        df = df.fillna(0)
    
    print(f"Added basic technical indicators. Data has {len(df)} rows.")
    
    return df


def add_advanced_technical_indicators(df):
    """
    Add advanced technical indicators to the dataframe
    
    Args:
        df (pd.DataFrame): DataFrame containing stock data
        
    Returns:
        pd.DataFrame: DataFrame with added technical indicators
    """
    # Try to import talib, only when needed
    has_talib = False
    try:
        import talib
        has_talib = True
    except ImportError:
        has_talib = False
        print("Warning: TA-Lib not installed. Using basic indicators only.")
    
    if df is None or len(df) == 0:
        print("No data to process")
        return None
    
    # Check if the dataset is too small for complex indicators
    if len(df) < 50:
        print(f"Warning: Dataset has only {len(df)} rows, which may be insufficient for reliable advanced indicators")
        print("Using basic indicators with adjusted window sizes for small dataset")
        return add_basic_indicators(df)
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Store original data for preserving recent records
    original_dates = df['date'].copy()
    
    # Ensure data is sorted by date
    df = df.sort_values('date', ascending=True).reset_index(drop=True)
    
    # Basic Moving Averages
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA10'] = df['close'].rolling(window=10).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()
    df['MA200'] = df['close'].rolling(window=min(200, len(df) // 2)).mean()
    
    # Exponential Moving Averages
    df['EMA10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # Moving Average Crossovers - important for trend signals
    df['MA5_10_cross'] = df['MA5'] - df['MA10']
    df['MA10_20_cross'] = df['MA10'] - df['MA20']
    df['MA20_50_cross'] = df['MA20'] - df['MA50']
    df['EMA_cross'] = df['EMA10'] - df['EMA20']
    
    # Relative Strength Index (RSI)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss.replace(0, 0.001)  # Avoid division by zero
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # RSI divergence
    df['RSI_slope'] = df['RSI'].diff(5)
    df['price_slope'] = df['close'].diff(5)
    df['RSI_divergence'] = np.where(
        (df['RSI_slope'] > 0) & (df['price_slope'] < 0) |
        (df['RSI_slope'] < 0) & (df['price_slope'] > 0),
        1, 0
    )
    
    # Moving Average Convergence Divergence (MACD)
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    
    # Bollinger Bands
    df['BB_middle'] = df['close'].rolling(window=20).mean()
    df['BB_std'] = df['close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
    df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)
    
    # Bollinger Band Width - measure of volatility
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle'].replace(0, 0.001)
    
    # Bollinger %B - position within Bollinger Bands
    bb_range = df['BB_upper'] - df['BB_lower']
    bb_range = bb_range.replace(0, 0.001)  # Avoid division by zero
    df['BB_B'] = (df['close'] - df['BB_lower']) / bb_range
    
    # Volatility
    df['volatility_10'] = df['close'].rolling(window=10).std()
    df['volatility_20'] = df['close'].rolling(window=20).std()
    df['volatility_ratio'] = df['volatility_10'] / df['volatility_20'].replace(0, 0.001)
    
    # Price Rate of Change - momentum indicators
    df['price_roc_5'] = df['close'].pct_change(periods=5) * 100
    df['price_roc_10'] = df['close'].pct_change(periods=10) * 100
    df['price_roc_20'] = df['close'].pct_change(periods=20) * 100
    
    # Average Directional Index (ADX) - simplified version
    df['tr1'] = abs(df['high'] - df['low'])
    df['tr2'] = abs(df['high'] - df['close'].shift())
    df['tr3'] = abs(df['low'] - df['close'].shift())
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=14).mean()
    
    # Directional movement
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']
    
    df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    
    # Avoid division by zero
    atr_nonzero = df['atr'].replace(0, 0.001)
    df['plus_di'] = 100 * df['plus_dm'].rolling(window=14).mean() / atr_nonzero
    df['minus_di'] = 100 * df['minus_dm'].rolling(window=14).mean() / atr_nonzero
    
    # ADX calculation - avoid division by zero
    di_sum = df['plus_di'] + df['minus_di']
    di_sum = di_sum.replace(0, 0.001)
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / di_sum
    df['ADX'] = df['dx'].rolling(window=14).mean()
    
    # Volume indicators
    df['volume_ma10'] = df['volume'].rolling(window=10).mean()
    df['volume_ma20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma10'].replace(0, 0.001)
    
    # On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    
    # Price momentum
    df['momentum_5'] = df['close'] - df['close'].shift(5)
    df['momentum_10'] = df['close'] - df['close'].shift(10)
    
    # Chaikin Money Flow
    high_low_diff = df['high'] - df['low']
    high_low_diff = high_low_diff.replace(0, 0.001)  # Avoid division by zero
    df['MFM'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / high_low_diff
    df['MFV'] = df['MFM'] * df['volume']
    vol_sum = df['volume'].rolling(window=20).sum()
    vol_sum = vol_sum.replace(0, 0.001)  # Avoid division by zero
    df['CMF'] = df['MFV'].rolling(window=20).sum() / vol_sum
    
    # Stochastic Oscillator
    df['lowest_low'] = df['low'].rolling(window=14).min()
    df['highest_high'] = df['high'].rolling(window=14).max()
    # Avoid division by zero
    high_low_range = df['highest_high'] - df['lowest_low']
    high_low_range = high_low_range.replace(0, 0.001)
    df['%K'] = 100 * ((df['close'] - df['lowest_low']) / high_low_range)
    df['%D'] = df['%K'].rolling(window=3).mean()
    
    # Commodity Channel Index (CCI)
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['tp_ma'] = df['tp'].rolling(window=20).mean()
    df['tp_dev'] = (df['tp'] - df['tp_ma']).abs()
    df['tp_dev_ma'] = df['tp_dev'].rolling(window=20).mean()
    # Avoid division by zero
    tp_dev_nonzero = df['tp_dev_ma'].replace(0, 0.001)
    df['CCI'] = (df['tp'] - df['tp_ma']) / (0.015 * tp_dev_nonzero)
    
    # Price vs Volume relationship
    df['price_vol_corr'] = df['close'].rolling(window=10).corr(df['volume'])
    
    # Ichimoku Cloud components - advanced trend indicator, but these use future data shifts
    # We'll calculate these but not let them affect our most recent data
    df['tenkan_sen'] = (df['high'].rolling(window=9).max() + df['low'].rolling(window=9).min()) / 2
    df['kijun_sen'] = (df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2
    
    # Store the data before adding forward-looking indicators
    df_recent = df.copy()
    
    # Add forward-looking indicators (these will create NaN values for recent dates)
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
    df['senkou_span_b'] = ((df['high'].rolling(window=52).max() + df['low'].rolling(window=52).min()) / 2).shift(26)
    df['chikou_span'] = df['close'].shift(-26)  # This creates NaN values for the 26 most recent days
    
    # Advanced TA-Lib indicators if available
    if has_talib:
        try:
            # Trend Indicators
            df['TEMA'] = talib.TEMA(df['close'], timeperiod=20)
            df['DEMA'] = talib.DEMA(df['close'], timeperiod=20)
            
            # Momentum Indicators
            df['MOM'] = talib.MOM(df['close'], timeperiod=10)
            df['PPO'] = talib.PPO(df['close'], fastperiod=12, slowperiod=26, matype=0)
            
            # Cycle Indicators
            df['HT_DCPERIOD'] = talib.HT_DCPERIOD(df['close'])
            
            # Pattern Recognition
            df['CDL_DOJI'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
            df['CDL_HAMMER'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
            df['CDL_ENGULFING'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
            
        except Exception as e:
            print(f"Warning: Some TA-Lib indicators could not be added: {e}")
            # Ensure all required TA-Lib columns exist with fallback values
            talib_indicators = ['TEMA', 'DEMA', 'MOM', 'PPO', 'HT_DCPERIOD', 
                               'CDL_DOJI', 'CDL_HAMMER', 'CDL_ENGULFING']
            for indicator in talib_indicators:
                if indicator not in df.columns:
                    print(f"Adding fallback column for missing TA-Lib indicator: {indicator}")
                    # Use appropriate fallback based on indicator type
                    if indicator in ['CDL_DOJI', 'CDL_HAMMER', 'CDL_ENGULFING']:
                        df[indicator] = 0  # Pattern indicators are typically 0 or ±100
                    elif indicator == 'HT_DCPERIOD':
                        df[indicator] = 20  # Typical cycle period
                    else:
                        # For trend/momentum indicators, use a transformed version of existing indicators
                        df[indicator] = df['close'].rolling(window=20).mean()
    else:
        # If TA-Lib is not available, add placeholder columns
        print("TA-Lib not available, adding placeholder columns")
        talib_indicators = ['TEMA', 'DEMA', 'MOM', 'PPO', 'HT_DCPERIOD', 
                           'CDL_DOJI', 'CDL_HAMMER', 'CDL_ENGULFING']
        for indicator in talib_indicators:
            if indicator in ['CDL_DOJI', 'CDL_HAMMER', 'CDL_ENGULFING']:
                df[indicator] = 0
            elif indicator == 'HT_DCPERIOD':
                df[indicator] = 20
            else:
                df[indicator] = df['close'].rolling(window=20).mean()
    
    # Add day of week, month, quarter (calendar features)
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    
    # Drop auxiliary columns used for calculations
    auxiliary_cols = ['tr1', 'tr2', 'tr3', 'up_move', 'down_move', 'tp', 'tp_ma', 'tp_dev', 'tp_dev_ma',
                     'lowest_low', 'highest_high', 'MFM', 'MFV']
    df = df.drop(columns=[col for col in auxiliary_cols if col in df.columns])
    
    # Instead of dropping all NaN rows, which would lose recent data,
    # we'll handle NaNs differently based on where they appear
    
    # First, try forward and backward filling to preserve as much data as possible
    df_filled = df.fillna(method='ffill').fillna(method='bfill')
    
    # Count how many NaNs remain after filling
    na_count = df_filled.isna().sum().sum()
    
    if na_count > 0:
        print(f"Warning: {na_count} NaN values remain after forward/backward filling")
        # If there are still NaNs, we'll take a more aggressive approach
        
        # For backward indicators (everything except the Ichimoku forward-looking ones)
        backward_indicators = df.columns.difference(['senkou_span_a', 'senkou_span_b', 'chikou_span'])
        
        # Try to preserve as many rows as possible by only requiring backward indicators
        df_historical = df_filled.dropna(subset=backward_indicators)
        
        if len(df_historical) < 10:  # If we have too few rows, fall back to basic indicators
            print(f"Warning: Only {len(df_historical)} valid rows after handling NaNs. Falling back to basic indicators.")
            return add_basic_indicators(df)
    else:
        # No NaNs after filling, we can use all the data
        df_historical = df_filled
    
    # For the most recent data (which would have NaNs in forward-looking indicators),
    # use the preserved version without forward-looking indicators
    df_recent = df_recent.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # Identify recent dates that might have been filtered out
    if not df_historical.empty:
        recent_date_threshold = df_historical['date'].max()
        df_recent_filtered = df_recent[df_recent['date'] > recent_date_threshold]
    else:
        df_recent_filtered = df_recent
    
    # Merge historical and recent data
    if not df_recent_filtered.empty:
        # Add the forward-looking columns to df_recent so it has the same structure as df_historical
        for col in ['senkou_span_a', 'senkou_span_b', 'chikou_span']:
            if col not in df_recent_filtered.columns:
                df_recent_filtered[col] = np.nan
        
        # Ensure columns match exactly
        common_columns = list(set(df_historical.columns) & set(df_recent_filtered.columns))
        
        if common_columns and 'date' in common_columns:
            df_recent_filtered = df_recent_filtered[common_columns]
            df_historical = df_historical[common_columns]
            
            # Combine the datasets
            df_final = pd.concat([df_historical, df_recent_filtered], ignore_index=True)
            df_final = df_final.drop_duplicates(subset='date').sort_values('date').reset_index(drop=True)
        else:
            df_final = df_historical
    else:
        df_final = df_historical
    
    # Final check for empty dataframe
    if df_final.empty or len(df_final) == 0:
        print("Warning: Advanced indicators resulted in an empty dataframe. Falling back to basic indicators.")
        return add_basic_indicators(df)
    
    # Final NaN check - replace any remaining with 0
    if df_final.isna().any().any():
        print("Warning: Still found NaN values in final dataframe. Filling with zeros.")
        df_final = df_final.fillna(0)
    
    print(f"Added advanced technical indicators. Data now has {len(df_final)} rows.")
    # Print the first and last dates to help troubleshoot
    if not df_final.empty:
        print(f"First date: {df_final['date'].min()}, Last date: {df_final['date'].max()}")
    
    return df_final


def add_technical_indicators(df):
    """
    Add technical indicators to the dataframe (legacy function)
    
    Args:
        df (pd.DataFrame): DataFrame containing stock data
        
    Returns:
        pd.DataFrame: DataFrame with added technical indicators
    """
    # Call the advanced version for better results
    return add_advanced_technical_indicators(df)


def prepare_train_test_data(df, target_column='close', window_size=10, test_size=0.2, prediction_days=5):
    """
    Prepare data for model training and testing
    
    Args:
        df (pd.DataFrame): DataFrame with technical indicators
        target_column (str): Column to predict
        window_size (int): Number of previous days to use for prediction
        test_size (float): Proportion of data to use for testing
        prediction_days (int): Number of days ahead to predict
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler, features)
    """
    try:
        if df is None or len(df) == 0:
            print("No data to prepare")
            return None, None, None, None, None, None
        
        # Check if dataset is too small for training
        if len(df) < window_size + prediction_days:
            print(f"WARNING: Dataset too small ({len(df)} rows) for window_size={window_size} and prediction_days={prediction_days}")
            print("Need at least window_size + prediction_days rows for training")
            return None, None, None, None, None, None
        
        # Ensure data is sorted by date
        df = df.sort_values('date', ascending=True).reset_index(drop=True)
        
        # Print diagnostic info
        print(f"Preparing data with {len(df)} rows, window_size={window_size}, prediction_days={prediction_days}")
        
        # Shift the target column to get future values
        df['target'] = df[target_column].shift(-prediction_days)
        
        # Drop rows with NaN in target or any feature column
        print(f"Data shape before dropna: {df.shape}")
        df = df.dropna()
        print(f"Data shape after dropna: {df.shape}")
        
        # If after dropping NaN we have too few rows, return None
        if len(df) < window_size + 1:
            print(f"ERROR: After dropping NaN values, only {len(df)} rows remain, which is insufficient")
            return None, None, None, None, None, None
        
        # Select features (all except date and target)
        features = df.columns.difference(['date', 'target'])
        print(f"Selected {len(features)} features: {', '.join(features[:5])}...")
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df[features])
        scaled_target = df['target'].values
        
        # Create sequences with aligned indexes
        X, y = [], []
        
        # Ensure we have enough data for the sequences
        if len(scaled_data) <= window_size:
            print(f"ERROR: Not enough data points ({len(scaled_data)}) for the window size ({window_size})")
            return None, None, None, None, None, None
        
        print(f"Creating sequences from {window_size} to {len(scaled_data)}")
        for i in range(window_size, len(scaled_data)):
            X.append(scaled_data[i-window_size:i])
            y.append(scaled_target[i])
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # CRITICAL DEBUG INFO: Verify the length of both arrays
        print(f"Created X array with shape {X.shape} and y array with shape {y.shape}")
        
        # ERROR CHECK: Ensure X and y have the same number of samples
        if len(X) != len(y):
            print(f"WARNING: Data cardinality mismatch detected! X has {len(X)} samples while y has {len(y)} samples.")
            print("Fixing by ensuring both arrays have the same number of samples...")
            min_samples = min(len(X), len(y))
            X = X[:min_samples]
            y = y[:min_samples]
            print(f"After fix: X shape: {X.shape}, y shape: {y.shape}")
        
        # Ensure we have enough data after all preprocessing
        if len(X) < 10:  # Arbitrary small number
            print(f"ERROR: Not enough processed samples ({len(X)}) after sequence creation")
            return None, None, None, None, None, None
        
        # Split into train and test sets
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Prepared data with shape: X_train {X_train.shape}, y_train {y_train.shape}, X_test {X_test.shape}, y_test {y_test.shape}")
        
        return X_train, X_test, y_train, y_test, scaler, features
    
    except Exception as e:
        print(f"Error preparing training data: {e}")
        return None, None, None, None, None, None


def prepare_prediction_data(df, target_column='close', window_size=10):
    """
    Prepare the most recent data for prediction
    
    Args:
        df (pd.DataFrame): DataFrame with technical indicators
        target_column (str): Column to predict
        window_size (int): Number of previous days to use for prediction
        
    Returns:
        tuple: (X_pred, scaler)
    """
    if df is None or len(df) == 0:
        print("No data to prepare")
        return None, None
    
    # Ensure data is sorted by date
    df = df.sort_values('date', ascending=True).reset_index(drop=True)
    
    # Select features (all except date)
    features = df.columns.difference(['date'])
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[features])
    
    # Create sequence for the most recent data
    X_pred = scaled_data[-window_size:].reshape(1, window_size, len(features))
    
    return X_pred, scaler, features 