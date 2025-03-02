import os
import argparse
import numpy as np
import pandas as pd
from pandas import Timestamp
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import matplotlib.dates as mdates
import sys

from src.preprocessing import load_stock_data, add_technical_indicators
from src.market_analysis import get_market_trend_analysis, adjust_prediction_for_market_trend
from src.prediction_reward_system import PredictionRewardSystem


def format_date_safely(date_obj):
    """
    Safely format a date object to string, handling different types
    
    Args:
        date_obj: Date object (could be string, datetime, pd.Timestamp, etc.)
        
    Returns:
        str: Formatted date string in YYYY-MM-DD format
    """
    if isinstance(date_obj, str):
        return date_obj
    elif isinstance(date_obj, datetime) or isinstance(date_obj, pd.Timestamp):
        return date_obj.strftime('%Y-%m-%d')
    else:
        # Try to convert to string as fallback
        return str(date_obj)


def load_model_and_metadata(symbol, model_dir='models'):
    """
    Load the trained model and associated metadata
    
    Args:
        symbol (str): Stock symbol
        model_dir (str): Directory containing the model
        
    Returns:
        tuple: (model, scaler, features)
    """
    # Create company-specific model directory path
    company_model_dir = os.path.join(model_dir, symbol)
    
    # Enhanced debugging
    print(f"Looking for models for {symbol} in the following locations:")
    
    # Check for nested model directory structure (from train_and_predict_agp.py)
    nested_advanced_model_path = os.path.join(company_model_dir, symbol, "advanced_model.keras")
    nested_legacy_model_path = os.path.join(company_model_dir, symbol, "lstm_model.keras")
    nested_checkpoint_path = os.path.join(company_model_dir, symbol, "checkpoint.keras")
    nested_scaler_path = os.path.join(company_model_dir, symbol, "scaler.pkl")
    nested_features_path = os.path.join(company_model_dir, symbol, "features.txt")
    
    # Define paths for both previous folder structures
    # Standard new folder structure
    new_advanced_model_path = os.path.join(company_model_dir, "advanced_model.keras")
    new_legacy_model_path = os.path.join(company_model_dir, "lstm_model.keras")
    new_scaler_path = os.path.join(company_model_dir, "scaler.pkl")
    new_features_path = os.path.join(company_model_dir, "features.txt")
    
    # Old folder structure
    old_advanced_model_path = os.path.join(model_dir, f"{symbol}_advanced_model.keras")
    old_legacy_model_path = os.path.join(model_dir, f"{symbol}_lstm_model.keras")
    old_scaler_path = os.path.join(model_dir, f"{symbol}_scaler.pkl")
    old_features_path = os.path.join(model_dir, f"{symbol}_features.txt")
    
    # Print debug information about paths being checked
    print(f"  - Nested structure: {nested_advanced_model_path}")
    print(f"  - Nested checkpoint: {nested_checkpoint_path}")
    print(f"  - New structure: {new_advanced_model_path}")
    print(f"  - Old structure: {old_advanced_model_path}")
    
    # Try to find a valid model in all possible structures, prioritizing newer models
    # Check nested structure first (from recent training)
    if os.path.exists(nested_advanced_model_path):
        model_path = nested_advanced_model_path
        scaler_path = nested_scaler_path
        features_path = nested_features_path
        print(f"Found advanced model in nested structure: {model_path}")
    elif os.path.exists(nested_checkpoint_path):
        model_path = nested_checkpoint_path
        scaler_path = nested_scaler_path
        features_path = nested_features_path
        print(f"Found checkpoint model in nested structure: {model_path}")
    elif os.path.exists(nested_legacy_model_path):
        model_path = nested_legacy_model_path
        scaler_path = nested_scaler_path
        features_path = nested_features_path
        print(f"Found legacy model in nested structure: {model_path}")
    # Then check new structure
    elif os.path.exists(new_advanced_model_path):
        model_path = new_advanced_model_path
        scaler_path = new_scaler_path
        features_path = new_features_path
        print(f"Found advanced model in new structure: {model_path}")
    elif os.path.exists(new_legacy_model_path):
        model_path = new_legacy_model_path
        scaler_path = new_scaler_path
        features_path = new_features_path
        print(f"Found legacy model in new structure: {model_path}")
    # Then check old structure
    elif os.path.exists(old_advanced_model_path):
        model_path = old_advanced_model_path
        scaler_path = old_scaler_path
        features_path = old_features_path
        print(f"Found advanced model in old structure: {model_path}")
    elif os.path.exists(old_legacy_model_path):
        model_path = old_legacy_model_path
        scaler_path = old_scaler_path
        features_path = old_features_path
        print(f"Found legacy model in old structure: {model_path}")
    else:
        print(f"Error: No model found for {symbol} in any structure.")
        print("Please train a model first by running the pipeline without the --skip_training flag.")
        return None, None, None
    
    # Check if metadata files exist
    if not os.path.exists(scaler_path):
        print(f"Error: Scaler file {scaler_path} does not exist")
        return None, None, None
    
    if not os.path.exists(features_path):
        print(f"Error: Features file {features_path} does not exist")
        return None, None, None
    
    # Load model
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None
    
    # Load scaler
    try:
        scaler = joblib.load(scaler_path)
        print(f"Successfully loaded scaler from {scaler_path}")
    except Exception as e:
        print(f"Error loading scaler: {e}")
        return None, None, None
    
    # Load features
    try:
        with open(features_path, 'r') as f:
            features = [line.strip() for line in f.readlines()]
        print(f"Successfully loaded {len(features)} features from {features_path}")
    except Exception as e:
        print(f"Error loading features: {e}")
        return None, None, None
    
    print(f"Successfully loaded model and metadata for {symbol}")
    return model, scaler, features


def prepare_prediction_data(df, features, scaler, window_size=10):
    """
    Prepare data for prediction
    
    Args:
        df (pd.DataFrame): DataFrame with technical indicators
        features (list): Feature names to use
        scaler (sklearn.preprocessing.MinMaxScaler): Scaler for normalizing data
        window_size (int): Number of previous days to use for prediction
        
    Returns:
        np.array: Prepared data for prediction
    """
    if df is None or len(df) == 0:
        print(f"Error: No data for prediction")
        return None, None
    
    # Check if we have enough data for the window size
    if len(df) < window_size:
        print(f"Error: Not enough data for prediction (need at least {window_size} records but only found {len(df)})")
        return None, None
    
    # Select features from dataframe
    try:
        # Create a copy of the dataframe to avoid modifying the original
        df_copy = df.copy()
        
        # Check if all required features exist in the dataframe
        missing_features = [f for f in features if f not in df_copy.columns]
        if missing_features:
            print(f"Warning: Missing features in data: {missing_features}")
            print("Adding placeholder values for missing features")
            
            # Add missing features with appropriate placeholder values
            for feature in missing_features:
                # Use simple heuristics to generate reasonable placeholder values
                if feature in ['day_of_week', 'month', 'quarter']:
                    # For date-related features, try to derive from date if possible
                    if 'date' in df_copy.columns:
                        try:
                            if feature == 'day_of_week':
                                df_copy[feature] = df_copy['date'].dt.dayofweek
                            elif feature == 'month':
                                df_copy[feature] = df_copy['date'].dt.month
                            elif feature == 'quarter':
                                df_copy[feature] = df_copy['date'].dt.quarter
                        except:
                            # If date conversion fails, use default values
                            df_copy[feature] = 0
                    else:
                        df_copy[feature] = 0
                elif feature in ['CDL_DOJI', 'CDL_HAMMER', 'CDL_ENGULFING']:
                    # Candlestick patterns are typically 0 (no pattern) or 100/-100 (pattern present)
                    df_copy[feature] = 0
                elif feature in ['TEMA', 'DEMA', 'MA5', 'MA10', 'MA20', 'MA50', 'MA200', 'EMA10', 'EMA20', 'EMA50']:
                    # Moving averages can use close price as a simple substitute
                    if 'close' in df_copy.columns:
                        df_copy[feature] = df_copy['close']
                    else:
                        # If no close price, use the mean of the data
                        df_copy[feature] = df_copy.mean(numeric_only=True).mean()
                else:
                    # For other features, use zeros or mean of existing features
                    print(f"Using default value for unknown feature type: {feature}")
                    df_copy[feature] = 0
        
        # Now select only the required features
        df_features = df_copy[features]
        
        # Check for NaN values
        if df_features.isna().any().any():
            print("Warning: Data contains NaN values. Filling with forward fill method...")
            df_features = df_features.ffill()
            # If still have NaNs, use backward fill
            if df_features.isna().any().any():
                df_features = df_features.bfill()
            # If still have NaNs after both methods, replace with zeros
            if df_features.isna().any().any():
                df_features = df_features.fillna(0)
                print("Warning: Some NaN values could not be filled with forward/backward fill. Using zeros.")
        
        # Scale the data
        scaled_data = scaler.transform(df_features)
        
        # Create sequences
        X = []
        for i in range(window_size, len(scaled_data) + 1):
            X.append(scaled_data[i-window_size:i])
        
        X = np.array(X)
        
        # Final check that we have at least one sequence
        if len(X) == 0:
            print(f"Error: Could not create any valid sequences with window size {window_size}")
            return None, None
            
        return X, scaled_data
        
    except Exception as e:
        print(f"Error preparing prediction data: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def predict_future_prices_advanced(model, X, df, scaler, features, days_ahead=5, smoothing_factor=0.7, apply_market_trends=True, market_adjustment_factor=0.03, market_trend_info=None, reward_system=None, window_size=10):
    """
    Predict future stock prices
    
    Args:
        model: Trained model
        X: Prepared sequence data
        df: DataFrame with historical data
        scaler: Fitted scaler
        features: Feature names
        days_ahead: Number of days to predict ahead
        smoothing_factor: Smoothing factor for predictions (0.0-1.0)
        apply_market_trends: Whether to apply market trend adjustments
        market_adjustment_factor: Factor controlling how much market trends affect predictions
        market_trend_info: Tuple containing market trend information (trends, sentiment_score)
        reward_system: PredictionRewardSystem instance for tracking predictions
        window_size: Size of the window used for predictions
        
    Returns:
        tuple: (predictions, confidence_intervals, significant_days, future_dates)
    """
    # Validate inputs
    if X is None or len(X) == 0:
        print("Error: No input data for prediction")
        return None, None, None, None
    
    # Handle case where X is a tuple (likely from prepare_prediction_data)
    if isinstance(X, tuple):
        print("Detected tuple return value from prepare_prediction_data")
        if len(X) >= 1 and X[0] is not None:
            X_data = X[0]  # Extract the actual array from the tuple
        else:
            print("Error: X tuple does not contain valid data")
            return None, None, None, None
    else:
        X_data = X  # X is already the data we need
    
    # Get the most recent window
    try:
        last_window = X_data[-1:].copy()
    except (IndexError, AttributeError) as e:
        print(f"Error accessing last window from input data: {e}")
        print(f"X_data type: {type(X_data)}, shape (if array): {getattr(X_data, 'shape', 'No shape')}")
        return None, None, None, None
    
    # Store the original dataframe
    df_future = df.copy()
    
    try:
        last_date = df_future['date'].iloc[-1]
    except (IndexError, KeyError) as e:
        print(f"Error accessing date column: {e}")
        return None, None, None, None
    
    # Make predictions
    prices = []
    confidence_intervals = []
    
    # Use the last window as our starting point
    current_window = last_window.copy()
    
    # Extract the feature indices for close price
    try:
        close_idx = features.index('close')
    except ValueError:
        print("Warning: 'close' column not found in features. Using first feature instead.")
        close_idx = 0
    
    # Get the min and max values for the close price to ensure predictions are in a reasonable range
    if hasattr(scaler, 'data_min_') and hasattr(scaler, 'data_max_'):
        min_price = scaler.data_min_[close_idx]
        max_price = scaler.data_max_[close_idx]
    else:
        min_price = 0
        max_price = 1
    
    # Get the last closing price for applying PSX price limits
    last_close_price = df['close'].iloc[-1]
    
    # Generate future predictions
    for i in range(days_ahead):
        # Predict the next day's price
        pred = model.predict(current_window, verbose=0)
        
        # Add some randomness to simulate confidence intervals
        # This is a simple way to generate intervals and could be improved
        noise_level = 0.01  # 1% random noise
        lower_bound = pred * (1 - noise_level)
        upper_bound = pred * (1 + noise_level)
        
        # Apply PSX price limit rules
        if i == 0:
            # For first day prediction, apply limits based on the last actual closing price
            pred_value = apply_psx_price_limits(pred[0][0], last_close_price)
        else:
            # For subsequent days, apply limits based on the previous day's prediction
            pred_value = apply_psx_price_limits(pred[0][0], prices[-1])
            
        # Recompute confidence intervals after applying price limits
        lower_bound_value = pred_value * (1 - noise_level)
        upper_bound_value = pred_value * (1 + noise_level)
            
        # Store the prediction and confidence intervals
        prices.append(pred_value)
        confidence_intervals.append((lower_bound_value, upper_bound_value))
        
        # Update the window for the next prediction (rolling window approach)
        try:
            # Check the structure of current_window to extract features correctly
            if current_window.ndim > 2 and current_window.shape[1] > 0 and current_window.shape[2] > 0:
                # For 3D arrays (sequences, time steps, features)
                next_features = current_window[0, -1].copy()  # Get the last time step from first sequence
            else:
                print(f"Warning: Unexpected current_window shape: {current_window.shape}")
                # Create a dummy array of the right size for the next prediction
                next_features = np.zeros(len(features))
                
            # Update the close price feature with our prediction
            # Apply smoothing if this is not the first prediction
            if i > 0 and smoothing_factor > 0:
                smoothed_price = smoothing_factor * prices[-1] + (1 - smoothing_factor) * prices[-2]
                next_features[close_idx] = smoothed_price
            else:
                next_features[close_idx] = prices[-1]
                
            # Ensure prediction is within reasonable bounds (based on min/max of training data)
            next_features[close_idx] = max(min_price, min(max_price, next_features[close_idx]))
            
            # Roll the window forward (depends on the shape of current_window)
            if current_window.ndim > 2:
                # Create a new array with the updated features
                new_window = current_window.copy()
                # Shift the time steps and replace the last one
                new_window[0, :-1] = current_window[0, 1:]
                new_window[0, -1] = next_features
                current_window = new_window
            else:
                print("Warning: Cannot update window, unexpected shape. Using last window for next prediction.")
        except Exception as e:
            print(f"Error updating window for next prediction: {e}")
            print(f"Current window shape: {current_window.shape}, type: {type(current_window)}")
            # For debugging, print more information
            print(f"Next features shape: {getattr(next_features, 'shape', 'N/A')}, type: {type(next_features)}")
            # Use the same window for the next prediction
    
    # Convert to numpy arrays
    predictions = np.array(prices)
    confidence_intervals = np.array(confidence_intervals)
    
    # Reverse the scaling for the predictions and intervals
    try:
        # Create a dummy array with the same shape as what the scaler expects
        dummy = np.zeros((len(predictions), len(features)))
        # Set the close price column
        dummy[:, close_idx] = predictions
        
        # Inverse transform to get the actual prices
        dummy_inverse = scaler.inverse_transform(dummy)
        predictions = dummy_inverse[:, close_idx]
        
        # IMPORTANT: Apply price limits again AFTER inverse transform
        # This ensures the final predictions respect the PSX price limit rules
        # First day is based on last actual close price
        predictions[0] = apply_psx_price_limits(predictions[0], last_close_price)
        # Subsequent days are each based on the previous prediction
        for i in range(1, len(predictions)):
            predictions[i] = apply_psx_price_limits(predictions[i], predictions[i-1])
        
        # Do the same for confidence intervals
        dummy_lower = np.zeros((len(confidence_intervals), len(features)))
        dummy_upper = np.zeros((len(confidence_intervals), len(features)))
        
        dummy_lower[:, close_idx] = [x[0] for x in confidence_intervals]
        dummy_upper[:, close_idx] = [x[1] for x in confidence_intervals]
        
        dummy_lower_inverse = scaler.inverse_transform(dummy_lower)
        dummy_upper_inverse = scaler.inverse_transform(dummy_upper)
        
        # Apply price limits to confidence intervals as well
        for i in range(len(confidence_intervals)):
            if i == 0:
                lower_limit = apply_psx_price_limits(dummy_lower_inverse[i, close_idx], last_close_price)
                upper_limit = apply_psx_price_limits(dummy_upper_inverse[i, close_idx], last_close_price)
            else:
                lower_limit = apply_psx_price_limits(dummy_lower_inverse[i, close_idx], predictions[i-1])
                upper_limit = apply_psx_price_limits(dummy_upper_inverse[i, close_idx], predictions[i-1])
            confidence_intervals[i] = (lower_limit, upper_limit)
        
    except Exception as e:
        print(f"Warning: Error in inverse transform: {e}")
        # If there's an error, we'll just return the scaled predictions
    
    # Apply market trend adjustments if requested
    if apply_market_trends and market_adjustment_factor != 0:
        print(f"\nAdjusting predictions based on market sentiment (factor: {market_adjustment_factor})...")
        if market_trend_info is None:
            print("Warning: market_trend_info is None, using neutral sentiment (0.0)")
            sentiment_score = 0.0
        else:
            sentiment_score = market_trend_info[1] if isinstance(market_trend_info, tuple) and len(market_trend_info) > 1 else 0.0
            print(f"Using market sentiment score: {sentiment_score}")
            
        predictions = adjust_prediction_for_market_trend(
            predictions, 
            sentiment_score, 
            adjustment_factor=market_adjustment_factor
        )
        print(f"Market-adjusted predictions applied with factor: {market_adjustment_factor}")
    
    # Generate future dates variable
    future_dates = []
    
    # Save predictions to reward system
    if reward_system is not None:
        print("Saving predictions to the reward system")
        try:
            # Generate future dates starting from the day after the last date in the dataframe
            last_date_raw = df_future['date'].iloc[-1]
            
            # Convert to datetime object
            if isinstance(last_date_raw, str):
                try:
                    last_date = datetime.strptime(last_date_raw, '%Y-%m-%d')
                except ValueError:
                    try:
                        last_date = datetime.strptime(last_date_raw, '%Y/%m/%d')
                    except ValueError:
                        print(f"Warning: Could not parse date format: {last_date_raw}, using current date")
                        last_date = datetime.now()
            elif isinstance(last_date_raw, pd.Timestamp):
                last_date = last_date_raw.to_pydatetime()
            else:
                print(f"Warning: Unknown date format: {type(last_date_raw)}, using current date")
                last_date = datetime.now()
            
            # Generate future dates starting from the next day
            future_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(len(predictions))]
            
            # Save predictions with future dates
            for i, date in enumerate(future_dates):
                if i < len(predictions):  # Make sure the prediction exists
                    # Create a model version string with relevant parameters
                    model_version = f"window={window_size},smooth={smoothing_factor}"
                    if apply_market_trends:
                        model_version += f",mkt_adj={market_adjustment_factor}"
                    
                    # Save the prediction
                    reward_system.save_prediction(date, predictions[i], model_version=model_version)
                else:
                    print(f"Warning: No prediction available for date {date} (index {i})")
        except Exception as e:
            print(f"Error during prediction: {e}")
            print(f"This might be because no model exists for {symbol} yet.")
    else:
        # If no reward system, still generate future dates
        try:
            last_date_raw = df_future['date'].iloc[-1]
            
            # Convert to datetime object
            if isinstance(last_date_raw, str):
                try:
                    last_date = datetime.strptime(last_date_raw, '%Y-%m-%d')
                except ValueError:
                    try:
                        last_date = datetime.strptime(last_date_raw, '%Y/%m/%d')
                    except ValueError:
                        print(f"Warning: Could not parse date format: {last_date_raw}, using current date")
                        last_date = datetime.now()
            elif isinstance(last_date_raw, pd.Timestamp):
                last_date = last_date_raw.to_pydatetime()
            else:
                print(f"Warning: Unknown date format: {type(last_date_raw)}, using current date")
                last_date = datetime.now()
            
            # Generate future dates starting from the next day
            future_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(len(predictions))]
        except Exception as e:
            print(f"Warning: Could not generate future dates: {e}")
            # Generate generic dates if we couldn't parse the last date
            future_dates = [f"Day_{i+1}" for i in range(len(predictions))]
    
    # Identify significant days
    significant_days = identify_significant_movements(predictions, threshold_pct=2.0)
    
    # Return the predictions, confidence intervals, significant days, and future dates
    return predictions, confidence_intervals, significant_days, future_dates


def apply_psx_price_limits(predicted_price, previous_price):
    """
    Apply Pakistan Stock Exchange (PSX) price limit rules
    
    PSX Rules:
    - Upper cap: +10% or 1 PKR (whichever is higher)
    - Lower cap: -10% or 1 PKR (whichever is higher in absolute terms)
    
    Args:
        predicted_price (float): Predicted price
        previous_price (float): Previous day's price
        
    Returns:
        float: Price adjusted according to PSX rules
    """
    # Calculate percentage-based limits
    upper_limit_pct = previous_price * 1.10  # +10%
    lower_limit_pct = previous_price * 0.90  # -10%
    
    # Calculate absolute value limits
    upper_limit_abs = previous_price + 1  # +1 PKR
    lower_limit_abs = previous_price - 1  # -1 PKR
    
    # Use the higher of the percentage or absolute limits
    upper_limit = max(upper_limit_pct, upper_limit_abs)
    # For lower limit, we want the higher value (less negative change)
    lower_limit = min(lower_limit_pct, lower_limit_abs)
    
    # Apply the limits
    if predicted_price > upper_limit:
        return upper_limit
    elif predicted_price < lower_limit:
        return lower_limit
    else:
        return predicted_price


def identify_significant_movements(predictions, threshold_pct=2.0):
    """
    Identify significant price movements in the predictions
    
    Args:
        predictions (np.array): Predicted prices
        threshold_pct (float): Threshold percentage change to consider a movement significant
        
    Returns:
        dict: Dictionary with 'up' and 'down' keys containing indices of days with significant up/down movements
    """
    significant_days = {'up': [], 'down': []}
    
    # Calculate daily percentage changes
    pct_changes = np.diff(predictions) / predictions[:-1] * 100
    
    # Find days with changes exceeding the threshold
    for i, pct_change in enumerate(pct_changes):
        if abs(pct_change) >= threshold_pct:
            # +1 because diff reduces array length by 1, but make sure it doesn't exceed array bounds
            if i + 1 < len(predictions):
                if pct_change > 0:
                    significant_days['up'].append(i + 1)
                else:
                    significant_days['down'].append(i + 1)
    
    return significant_days


def visualize_predictions_advanced(df, predictions, confidence_intervals, significant_days, symbol, future_dates, days_ahead):
    """
    Visualize historical data with predictions and confidence intervals
    
    Args:
        df: DataFrame with historical data
        predictions: List of predicted prices
        confidence_intervals: List of tuples (lower, upper) for confidence intervals
        significant_days: List of tuples (day_index, movement) for significant price movements
        symbol: Stock symbol
        future_dates: List of future dates for predictions
        days_ahead: Number of days predicted ahead
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import os
    from datetime import datetime, timedelta
    import numpy as np
    
    # Create output directory if it doesn't exist
    output_dir = f"models/{symbol}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot setup
    plt.figure(figsize=(12, 6))
    
    # Get the last date from historical data
    last_date = df['date'].iloc[-1]
    
    # Convert to datetime if it's not already
    if not isinstance(last_date, datetime) and not isinstance(last_date, np.datetime64):
        # Try parsing with various formats
        try:
            last_date = datetime.strptime(last_date, '%Y-%m-%d')
        except (ValueError, TypeError):
            try:
                last_date = datetime.strptime(last_date, '%Y/%m/%d')
            except (ValueError, TypeError):
                print(f"Warning: Could not parse date {last_date}, using current date instead")
                last_date = datetime.now()
    
    # Plot historical data (only include weekdays)
    historical_dates = df['date'].tolist()
    historical_prices = df['close'].tolist()
    
    # Filter out weekends for better visualization
    valid_historical = [(date, price) for date, price in zip(historical_dates, historical_prices) 
                      if isinstance(date, datetime) and date.weekday() < 5]
    
    if valid_historical:
        valid_dates, valid_prices = zip(*valid_historical)
        plt.plot(valid_dates, valid_prices, label='Historical Data', color='blue')
    
    # Plot predictions and confidence intervals
    # Filter out weekends from future dates for better visualization
    valid_indices = [i for i, date in enumerate(future_dates) 
                    if isinstance(date, datetime) and date.weekday() < 5]
    
    if valid_indices:
        # Make sure predictions follow PSX rules sequentially for visualization
        # Create a copy to avoid modifying the original predictions
        last_close_price = df['close'].iloc[-1]
        visualized_predictions = predictions.copy()
        visualized_confidence = confidence_intervals.copy()
        
        # Apply price limits to predictions used for visualization
        visualized_predictions[0] = apply_psx_price_limits(visualized_predictions[0], last_close_price)
        for i in range(1, len(visualized_predictions)):
            visualized_predictions[i] = apply_psx_price_limits(visualized_predictions[i], visualized_predictions[i-1])
        
        # Apply price limits to confidence intervals as well
        for i in range(len(visualized_confidence)):
            if i == 0:
                lower_limit = apply_psx_price_limits(visualized_confidence[i][0], last_close_price)
                upper_limit = apply_psx_price_limits(visualized_confidence[i][1], last_close_price)
            else:
                lower_limit = apply_psx_price_limits(visualized_confidence[i][0], visualized_predictions[i-1])
                upper_limit = apply_psx_price_limits(visualized_confidence[i][1], visualized_predictions[i-1])
            visualized_confidence[i] = (lower_limit, upper_limit)
        
        valid_dates = [future_dates[i] for i in valid_indices]
        valid_pred = [visualized_predictions[i] for i in valid_indices]
        valid_conf = [visualized_confidence[i] for i in valid_indices]
        
        plt.plot(valid_dates, valid_pred, label='Predictions', color='red', marker='o')
        
        # Plot confidence intervals
        lower_bounds = [conf[0] for conf in valid_conf]
        upper_bounds = [conf[1] for conf in valid_conf]
        plt.fill_between(valid_dates, lower_bounds, upper_bounds, color='red', alpha=0.2, label='Confidence Interval')
        
        # Mark significant movements if present
        for movement_type, days in significant_days.items():
            for day in days:
                if 0 <= day < len(valid_indices):
                    idx = valid_indices.index(day) if day in valid_indices else None
                    if idx is not None and idx < len(valid_pred):
                        marker_color = 'green' if movement_type == 'up' else 'red'
                        marker_style = '^' if movement_type == 'up' else 'v'
                        plt.plot(valid_dates[idx], valid_pred[idx], marker=marker_style, color=marker_color, 
                                markersize=10, label=f'Significant {movement_type}' if idx == 0 else "")
    
    # Format the plot
    plt.title(f'{symbol} Stock Price Prediction for {days_ahead} days')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    plt.legend()
    
    # Format the date axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.gcf().autofmt_xdate()
    
    # Save the plot
    plt.savefig(f'{output_dir}/future_prediction.png')
    
    # Print summary
    print("\nPrediction Summary:")
    print(f"Current price: {historical_prices[-1]:.2f}")
    if len(predictions) > 0:
        print(f"Predicted price after {days_ahead} days: {predictions[-1]:.2f}")
        change = predictions[-1] - historical_prices[-1]
        percent_change = (change / historical_prices[-1]) * 100
        print(f"Change: {change:.2f} ({percent_change:.2f}%)")
    
    print("\nSignificant price movements predicted on:")
    for movement_type, days in significant_days.items():
        for day in days:
            if 0 <= day < len(future_dates):
                date = future_dates[day]
                # Format the date safely using our helper function
                date_str = format_date_safely(date)
                print(f"  {date_str}: {movement_type}")
    
    plt.close()


def predict_future(model, df, scaler, features, days=7, window_size=10):
    """
    Make predictions for future days
    
    Args:
        model (tf.keras.Model): Trained model
        df (pd.DataFrame): DataFrame with historical data
        scaler (sklearn.preprocessing.MinMaxScaler): Scaler used for normalization
        features (list): List of features used for prediction
        days (int): Number of days to predict ahead
        window_size (int): Window size for sequences
        
    Returns:
        tuple: (predictions, dates) - Array of predictions and corresponding dates
    """
    try:
        # Prepare data for prediction
        X, scaled_data = prepare_prediction_data(df, features, scaler, window_size)
        
        if X is None:
            return None, None
        
        # Create future dates for visualization
        last_date_raw = df['date'].iloc[-1]
        
        if isinstance(last_date_raw, str):
            try:
                last_date = datetime.strptime(last_date_raw, '%Y-%m-%d')
            except ValueError:
                try:
                    last_date = datetime.strptime(last_date_raw, '%Y/%m/%d')
                except ValueError:
                    print(f"Warning: Could not parse date format: {last_date_raw}, using current date")
                    last_date = datetime.now()
        elif isinstance(last_date_raw, pd.Timestamp):
            last_date = last_date_raw.to_pydatetime()
        else:
            print(f"Warning: Unknown date format: {type(last_date_raw)}, using current date")
            last_date = datetime.now()
        
        # Print the last date from the dataset to clearly show when predictions start from
        print(f"\nIMPORTANT: Last date in historical data: {last_date.strftime('%Y-%m-%d')}")
        print(f"Predictions will start from: {(last_date + timedelta(days=1)).strftime('%Y-%m-%d')}")
        
        future_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(days)]
        
        # Predict future prices
        predictions, confidence_intervals, significant_days, future_dates = predict_future_prices_advanced(
            model, X, df, scaler, features, days_ahead=days, window_size=window_size, market_trend_info=None
        )
        
        return predictions, future_dates
    
    except Exception as e:
        print(f"Error during future predictions: {e}")
        return None, None


def analyze_predictions(predictions, dates, threshold=2.0):
    """
    Analyze predictions for significant movements
    
    Args:
        predictions (np.array): Predicted prices
        dates (list): List of dates corresponding to predictions
        threshold (float): Threshold percentage for significant movements
    """
    # Calculate daily percentage changes
    pct_changes = np.diff(predictions) / predictions[:-1] * 100
    
    # Print predictions
    print(f"\nPredictions for the next {len(predictions)} days:")
    for i, (date, price) in enumerate(zip(dates, predictions)):
        movement = ""
        if i > 0:
            pct_change = pct_changes[i-1]
            if abs(pct_change) >= threshold:
                movement = f" ⚠️ SIGNIFICANT MOVEMENT: {pct_change:.2f}%"
        
        # Format the date safely using our helper function
        date_str = format_date_safely(date)
        print(f"  {date_str}: {price:.2f}{movement}")


def plot_predictions(df, predictions, dates, symbol, model_dir):
    """
    Create a visualization of predicted prices
    
    Args:
        df (pd.DataFrame): DataFrame with historical data
        predictions (list): List of predicted prices
        dates (list): List of dates for predictions
        symbol (str): Stock symbol
        model_dir (str): Directory to save the plot
        
    Returns:
        str: Path to the saved visualization
    """
    # Convert to numpy arrays
    hist_dates = pd.to_datetime(df['date']).values
    hist_prices = df['close'].values
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot historical data
    plt.plot(hist_dates, hist_prices, label='Historical Data', color='blue')
    
    # Plot predictions
    plt.plot(dates, predictions, label='Predictions', color='red', linestyle='--')
    
    # Add markers for predictions
    plt.scatter(dates, predictions, color='red', marker='o')
    
    # Add price annotations
    for i, (date, price) in enumerate(zip(dates, predictions)):
        plt.annotate(f'PKR {price:.2f}', 
                   (date, price), 
                   textcoords="offset points",
                   xytext=(0,10), 
                   ha='center')
    
    # Add labels and title
    plt.title(f'{symbol} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Create base model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Create company-specific model directory
    company_model_dir = os.path.join(model_dir, symbol)
    os.makedirs(company_model_dir, exist_ok=True)
    
    # Save the plot
    plot_path = os.path.join(company_model_dir, "future_prediction.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Saved prediction plot to {plot_path}")
    return plot_path


def main(symbol=None, window_size=10, days_ahead=5, threshold=2.0, 
         smoothing_factor=0.7, apply_market_trends=True, market_adjustment_factor=0.03,
         market_trend_info=None, reward_system=None, df_with_indicators=None):
    """
    Main function for predicting stock prices
    
    Args:
        symbol (str): Stock symbol
        window_size (int): Window size for sequences
        days_ahead (int): Number of days ahead to predict
        threshold (float): Threshold percentage for significant price movements
        smoothing_factor (float): Smoothing factor for predictions (0.0-1.0)
        apply_market_trends (bool): Whether to apply market trend adjustments
        market_adjustment_factor (float): Factor for market trend adjustments
        market_trend_info (dict): Market trend information if pre-calculated
        reward_system (PredictionRewardSystem): Reward system for tracking predictions
        df_with_indicators (pd.DataFrame): DataFrame with pre-calculated technical indicators
    
    Returns:
        tuple: (predictions, confidence_intervals, significant_days, future_dates) or None if error
    """
    try:
        if symbol is None:
            # Parse command line arguments
            parser = argparse.ArgumentParser(description='Stock Price Prediction')
            parser.add_argument('--symbol', type=str, required=True, help='Stock symbol')
            parser.add_argument('--days', type=int, default=5, help='Number of days to predict ahead')
            parser.add_argument('--window', type=int, default=10, help='Window size for sequences')
            parser.add_argument('--threshold', type=float, default=2.0, help='Threshold percentage for significant movements')
            parser.add_argument('--smoothing', type=float, default=0.7, help='Smoothing factor for predictions (0.0-1.0)')
            parser.add_argument('--no_rules', action='store_true', help='Skip applying financial rules')
            parser.add_argument('--no_market_trends', action='store_true', help='Skip market trend adjustments')
            parser.add_argument('--market_adjustment', type=float, default=0.03, help='Market adjustment factor (0.0-0.1)')
            parser.add_argument('--reward_threshold', type=float, default=0.05, help='Threshold for reward system')
            parser.add_argument('--no_reward_system', action='store_true', help='Skip using reward system')
            parser.add_argument('--force_cpu', action='store_true', help='Force using CPU')
            
            args = parser.parse_args()
            
            # Set parameters from command line arguments
            symbol = args.symbol
            days_ahead = args.days
            window_size = args.window
            threshold = args.threshold
            smoothing_factor = args.smoothing
            apply_market_trends = not args.no_market_trends
            market_adjustment_factor = args.market_adjustment
            no_rules = args.no_rules
            
            # Force CPU if requested
            if args.force_cpu:
                print("Forcing CPU usage as requested")
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            
            # Initialize reward system if enabled
            if not args.no_reward_system:
                reward_system = PredictionRewardSystem(threshold=args.reward_threshold)
                print(f"Prediction reward system enabled with threshold {args.reward_threshold}")
        
        # Ensure we have a symbol
        if symbol is None:
            print("Error: No stock symbol provided")
            return None
        
        print(f"Predicting future prices for {symbol}")
        
        # Load model and metadata
        model, scaler, features = load_model_and_metadata(symbol)
        
        if model is None or scaler is None or features is None:
            print(f"Error: Failed to load model for {symbol}")
            record_failed_prediction(symbol, "Failed to load model or metadata")
            return None
        
        # Load or use provided data with indicators
        if df_with_indicators is None:
            # Load stock data
            df = load_stock_data(symbol, apply_rules=True)
            
            if df is None or len(df) == 0:
                print(f"Error: No data found for {symbol}")
                record_failed_prediction(symbol, "No data found")
                return None
            
            # Add technical indicators
            print("Calculating technical indicators...")
            from src.preprocessing import load_or_calculate_technical_indicators
            try:
                df_with_indicators = load_or_calculate_technical_indicators(df, symbol)
            except Exception as e:
                print(f"Error during technical indicator calculation: {e}")
                record_failed_prediction(symbol, f"Technical indicator calculation failed: {e}")
                import traceback
                traceback.print_exc()
                return None
        else:
            print("Using pre-calculated technical indicators")
        
        if df_with_indicators is None or len(df_with_indicators) == 0:
            print(f"Error: Failed to calculate technical indicators for {symbol}")
            record_failed_prediction(symbol, "Technical indicators calculation returned empty result")
            return None
        
        # Prepare data for prediction
        X, scaled_data = prepare_prediction_data(df_with_indicators, features, scaler, window_size)
        
        if X is None or len(X) == 0:
            print("Error: Failed to prepare prediction data")
            record_failed_prediction(symbol, "Failed to prepare prediction data")
            return None
        
        # Get market trend information if not provided
        if market_trend_info is None and apply_market_trends:
            try:
                market_trend_info = get_market_trend_analysis(symbol)
            except Exception as e:
                print(f"Warning: Error getting market trend information: {e}")
                print("Continuing without market trend adjustments")
                apply_market_trends = False
        
        # Predict future prices
        predictions, confidence_intervals, significant_days, future_dates = predict_future_prices_advanced(
            model, X, df_with_indicators, scaler, features, 
            days_ahead=days_ahead, 
            smoothing_factor=smoothing_factor,
            apply_market_trends=apply_market_trends,
            market_adjustment_factor=market_adjustment_factor,
            market_trend_info=market_trend_info,
            reward_system=reward_system,
            window_size=window_size
        )
        
        if predictions is None:
            print("Error: Failed to make predictions")
            record_failed_prediction(symbol, "Failed to make predictions")
            return None
        
        # Create company-specific model directory
        company_model_dir = os.path.join('models', symbol)
        os.makedirs(company_model_dir, exist_ok=True)
        
        # Visualize predictions
        visualize_predictions_advanced(
            df_with_indicators, predictions, confidence_intervals, 
            significant_days, symbol, future_dates, days_ahead
        )
        
        print(f"Successfully completed prediction for {symbol}")
        return predictions, confidence_intervals, significant_days, future_dates
    
    except Exception as e:
        print(f"Unexpected error during prediction for {symbol}: {e}")
        record_failed_prediction(symbol, f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None


def record_failed_prediction(symbol, reason):
    """
    Record a failed prediction attempt in failed_companies.txt
    
    Args:
        symbol (str): Stock symbol
        reason (str): Reason for failure
    """
    try:
        with open('failed_companies.txt', 'a') as f:
            f.write(f"{symbol}\n")
        print(f"Recorded {symbol} in failed_companies.txt - Reason: {reason}")
    except Exception as e:
        print(f"Error recording failed prediction: {e}")


if __name__ == "__main__":
    main() 