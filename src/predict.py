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
import json
import time
import requests

from src.preprocessing import load_stock_data, add_technical_indicators, prepare_prediction_data
from src.market_analysis import get_market_trend_analysis, adjust_prediction_for_market_trend
from src.prediction_reward_system import PredictionRewardSystem
from src.reality_check import validate_predictions_against_reality, get_column_case_insensitive


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
    
    # First, check if we have a best model selected by the model selection system
    best_model_info_path = os.path.join(company_model_dir, "best_model_info.json")
    if os.path.exists(best_model_info_path):
        print(f"Found best model info at: {best_model_info_path}")
        try:
            import json
            with open(best_model_info_path, 'r') as f:
                best_model_info = json.load(f)
                
            # Get best model path and type
            model_path = best_model_info['model_path']
            is_dl_model = best_model_info['is_dl_model']
            model_type = best_model_info['best_model_type']
            model_name = best_model_info['model_name']
            
            # Check if model file exists
            if os.path.exists(model_path):
                print(f"Using best model ({model_name}) for {symbol}")
                
                # Load model based on type
                if is_dl_model:
                    model = tf.keras.models.load_model(model_path)
                else:
                    model = joblib.load(model_path)
                
                # Load scaler and features
                scaler_path = os.path.join(company_model_dir, "scaler.pkl")
                features_path = os.path.join(company_model_dir, "features.txt")
                
                if os.path.exists(scaler_path):
                    scaler = joblib.load(scaler_path)
                else:
                    print(f"Warning: Scaler not found at {scaler_path}")
                    scaler = None
                
                if os.path.exists(features_path):
                    with open(features_path, 'r') as f:
                        features = f.read().splitlines()
                else:
                    print(f"Warning: Features not found at {features_path}")
                    features = None
                
                print(f"Successfully loaded best model ({model_name}) for {symbol}")
                return model, scaler, features
            else:
                print(f"Warning: Best model file not found at {model_path}")
                # Continue to try other model paths
        except Exception as e:
            print(f"Error loading best model: {e}")
            import traceback
            traceback.print_exc()
            # Continue to try other model paths
    
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
        
        # Check for and replace infinity values
        print("Checking for and handling infinity or extremely large values...")
        df_features = df_features.replace([np.inf, -np.inf], np.nan)
        
        # Check for extremely large values and cap them
        for col in df_features.select_dtypes(include=np.number).columns:
            # Check for extreme values (simple approach: cap at reasonable limits)
            df_features[col] = df_features[col].clip(-1e6, 1e6)
        
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
        
        # Final safety check for any remaining issues
        if df_features.isna().any().any() or np.isinf(df_features.values).any():
            print("WARNING: Data still contains NaN or infinity values after cleaning. Replacing with zeros.")
            df_features = df_features.replace([np.inf, -np.inf], 0).fillna(0)
        
        # Try to scale the data with error handling
        try:
            scaled_data = scaler.transform(df_features)
        except Exception as scale_error:
            print(f"Error during scaling: {scale_error}")
            print("Attempting fallback scaling method...")
            
            # Fallback: do manual min-max scaling using known feature ranges
            scaled_data_list = []
            for i, feature_name in enumerate(features):
                feature_values = df_features[feature_name].values
                # Replace any problematic values
                feature_values = np.nan_to_num(feature_values, nan=0.0, posinf=1.0, neginf=0.0)
                # Try to get min/max from scaler for this feature
                try:
                    feature_min = scaler.data_min_[i]
                    feature_max = scaler.data_max_[i]
                    # If min and max are the same, just use 0
                    if feature_min == feature_max:
                        scaled_feature = np.zeros_like(feature_values)
                    else:
                        # Manual min-max scaling
                        scaled_feature = (feature_values - feature_min) / (feature_max - feature_min)
                        # Clip to [0, 1] range to be safe
                        scaled_feature = np.clip(scaled_feature, 0, 1)
                except:
                    # If getting min/max fails, just normalize to [0,1] based on current values
                    curr_min = np.min(feature_values)
                    curr_max = np.max(feature_values)
                    if curr_min == curr_max:
                        scaled_feature = np.zeros_like(feature_values)
                    else:
                        scaled_feature = (feature_values - curr_min) / (curr_max - curr_min)
                        scaled_feature = np.clip(scaled_feature, 0, 1)
                
                scaled_data_list.append(scaled_feature)
            
            # Convert to numpy array and reshape to the right format
            scaled_data = np.column_stack(scaled_data_list)
        
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


def generate_weekday_dates(start_date, num_days):
    """
    Generate a list of weekday dates starting from the given date
    
    Args:
        start_date (datetime): Starting date
        num_days (int): Number of weekdays to generate
        
    Returns:
        list: List of datetime objects representing weekdays
    """
    dates = []
    current_date = start_date
    
    while len(dates) < num_days:
        current_date = current_date + timedelta(days=1)
        # Skip weekends (0 = Monday, 6 = Sunday)
        if current_date.weekday() < 5:  # 0-4 are weekdays
            dates.append(current_date)
    
    return dates


def predict_future_prices_advanced(model, X, df, scaler, features, days_ahead=5, smoothing_factor=0.7, apply_market_trends=True, market_adjustment_factor=0.03, market_trend_info=None, reward_system=None, window_size=10, is_dl_model=True):
    """
    Predict future prices using the advanced model
    
    Args:
        model: Trained model
        X (np.array): Last window of data for prediction
        df (pd.DataFrame): DataFrame with historical data
        scaler: Scaler used for normalization
        features (list): Feature names used for prediction
        days_ahead (int): Number of days ahead to predict
        smoothing_factor (float): Factor for smoothing predictions (0.0-1.0)
        apply_market_trends (bool): Whether to apply market trend adjustments
        market_adjustment_factor (float): Factor for market trend adjustments
        market_trend_info (dict): Market trend information
        reward_system (PredictionRewardSystem): System for tracking and rewarding predictions
        window_size (int): Window size for sequences
        is_dl_model (bool): Whether the model is a deep learning model (True) or traditional ML model (False)
        
    Returns:
        tuple: (predictions, confidence_intervals, significant_days, future_dates)
    """
    # Predictions will be stored here
    predictions = []
    confidence_intervals = []
    future_dates = []  # Initialize as an empty list
    
    # Get last date and price from the historical data
    try:
        df_copy = df.copy()
        last_date = pd.to_datetime(df_copy['date'].iloc[-1])
        last_price = df_copy['close'].iloc[-1]
        print(f"Last known date: {last_date.strftime('%Y-%m-%d')}, Price: {last_price:.2f}")
    except Exception as e:
        print(f"Error getting last date/price: {e}")
        # Use today's date as fallback
        last_date = datetime.now()
        # Use a default price if needed - this is just a placeholder
        last_price = 100.0
        print(f"Using fallback last date: {last_date.strftime('%Y-%m-%d')}, Price: {last_price:.2f}")
    
    # Generate future dates for visualization
    future_dates = generate_weekday_dates(last_date, days_ahead)
    
    # Last X for prediction
    current_x = X.copy()
    
    # Get the index of close price in features
    close_idx = -1
    for i, feature in enumerate(features):
        if feature.lower() == 'close':
            close_idx = i
            break
    
    # Make sure we have the close price index
    if close_idx == -1:
        print("Warning: Could not find 'close' in features, using last value as fallback")
    
    # For tracking the cumulative error
    cumulative_error_factor = 1.0
    
    # Initialize current_date with last_date
    current_date = last_date
    
    # Predict for each day ahead
    for i in range(days_ahead):
        # Generate next weekday date
        while True:
            current_date = current_date + timedelta(days=1)
            # Skip weekends (0 = Monday, 6 = Sunday)
            if current_date.weekday() < 5:  # 0-4 are weekdays
                break
        
        future_dates.append(current_date)
        
        # Make prediction with current window
        # Initialize predicted_price_normalized to None to avoid UnboundLocalError
        predicted_price_normalized = None
        
        try:
            if is_dl_model:
                # Deep learning models need 3D input (samples, time_steps, features)
                # Check if current_x has extra dimensions and reshape if needed
                if len(current_x.shape) > 3:
                    # Improved reshape logic to handle (1, 358, 10, 73) shape
                    # First, determine the exact shape needed
                    expected_shape = (1, window_size, len(features))
                    
                    # Log the current shape for debugging
                    print(f"DEBUG: current_x shape before reshape: {current_x.shape}, expected: {expected_shape}")
                    
                    # Try different approaches to get the right shape
                    if current_x.shape[1] > 1 and current_x.shape[2] == window_size:
                        # If shape is like (1, 358, 10, 73), we need to take just one sample
                        # Take the most recent window (last index of dimension 1)
                        current_x_reshaped = current_x[0, -1:, :, :].reshape(1, window_size, len(features))
                    else:
                        # Alternative approach - reshape by flattening and rebuilding
                        try:
                            # Extract the most recent window_size * features elements and reshape
                            flattened = current_x.flatten()
                            elements_needed = window_size * len(features)
                            current_x_reshaped = flattened[-elements_needed:].reshape(1, window_size, len(features))
                        except Exception as reshape_error:
                            print(f"Reshape error: {reshape_error}, trying direct reshape of last elements")
                            # Last resort - take the last elements directly
                            current_x_reshaped = current_x[0, 0, :, :].reshape(1, window_size, len(features))
                    
                    print(f"DEBUG: Reshaped to: {current_x_reshaped.shape}")
                    predicted_price_normalized = model.predict(current_x_reshaped)
                else:
                    # If shape is already correct or has 3 dimensions, ensure it's (1, window_size, features)
                    if len(current_x.shape) == 3 and current_x.shape[0] != 1:
                        # If batch dimension is not 1, reshape to have batch size of 1
                        current_x_reshaped = current_x[-1:, :, :].reshape(1, window_size, len(features))
                        predicted_price_normalized = model.predict(current_x_reshaped)
                    else:
                        # Use as is
                        predicted_price_normalized = model.predict(current_x)
                    
                    # Extract prediction value (deep learning models return arrays)
                    prediction_value = predicted_price_normalized[0][0]
            else:
                # Traditional ML models need 2D input (samples, features)
                # Reshape the 3D input to 2D by flattening the time dimension
                print(f"Using traditional ML model. Current X shape: {current_x.shape}")
                if len(current_x.shape) == 3:
                    # Convert from (1, window_size, features) to (1, window_size*features)
                    current_x_flat = current_x.reshape(current_x.shape[0], -1)
                    print(f"Reshaped to 2D: {current_x_flat.shape}")
                    predicted_price_normalized = model.predict(current_x_flat)
                elif len(current_x.shape) > 3:
                    # Handle more complex shapes by taking the most recent window and flattening
                    if current_x.shape[1] > 1:
                        # Take the most recent sample
                        current_x_flat = current_x[0, -1].reshape(1, -1)
                    else:
                        current_x_flat = current_x.reshape(1, -1)
                    print(f"Reshaped complex input to 2D: {current_x_flat.shape}")
                    predicted_price_normalized = model.predict(current_x_flat)
                else:
                    # Already 2D, use as is
                    predicted_price_normalized = model.predict(current_x)
                
                # Extract prediction value (ML models often return 1D arrays)
                if hasattr(predicted_price_normalized, 'shape'):
                    if len(predicted_price_normalized.shape) > 0:
                        prediction_value = predicted_price_normalized[0]
                    else:
                        prediction_value = predicted_price_normalized
                else:
                    prediction_value = predicted_price_normalized
            
            # Convert prediction back to original scale
            if scaler is not None:
                # Create a dummy array with the same shape as the training data
                dummy = np.zeros((1, len(features)))
                # Put the predicted value in the close price position
                dummy[0, close_idx] = prediction_value
                # Inverse transform
                predicted_price = scaler.inverse_transform(dummy)[0, close_idx]
            else:
                predicted_price = prediction_value
            
            # Apply smoothing to make predictions more realistic
            # This uses an exponential moving average approach
            if i > 0:
                # The further into the future, the more we smooth
                adaptive_smoothing = min(0.9, smoothing_factor * (1 + (i * 0.1)))
                predicted_price = (adaptive_smoothing * predictions[-1]) + ((1 - adaptive_smoothing) * predicted_price)
        except Exception as e:
            print(f"Error during prediction: {e}")
            # Fallback to a simple forecast based on last price
            predicted_price = last_price * (1 + (0.001 * (i+1)))  # Small increase
        
        # Apply market trend adjustments if available
        if apply_market_trends and market_trend_info is not None:
            original_price = predicted_price
            
            # market_trend_info is a tuple (market_trends, sentiment_score)
            # Extract the sentiment score (second element)
            market_trends = None
            sentiment_score = 0.0
            
            if isinstance(market_trend_info, tuple) and len(market_trend_info) >= 2:
                market_trends, sentiment_score = market_trend_info
            elif isinstance(market_trend_info, (int, float)):
                # If market_trend_info is already a score, use it directly
                sentiment_score = market_trend_info
                
            predicted_price = adjust_prediction_for_market_trend(
                predicted_price, 
                sentiment_score,  # Pass sentiment_score instead of market_trend_info
                adjustment_factor=market_adjustment_factor,
                previous_close=last_price,  # Add previous_close for better adjustments
                market_trends=market_trends  # Pass market_trends as well
            )
            print(f"Applied market trend adjustment: {original_price:.2f} -> {predicted_price:.2f}")
        
        # Apply cumulative error adjustment (models tend to be overconfident further into future)
        if i > 0:
            # Increase uncertainty for predictions further in the future
            cumulative_error_factor *= 1.05  # 5% increase in uncertainty per day
            
            # Add some random noise that increases with each future day
            # This is a simple way to simulate increasing uncertainty
            noise_factor = 0.005 * i  # 0.5% per day
            # Ensure the scale parameter is always positive by using absolute value
            noise = np.random.normal(0, abs(noise_factor * predicted_price))
            predicted_price += noise
        
        # Ensure non-negative price
        predicted_price = max(0.01, predicted_price)
        
        # IMMEDIATE REALITY CHECK: Prevent absurdly large predictions right away
        # No stock can realistically increase or decrease more than 30% in a single day
        max_daily_change = 0.30  # 30% maximum daily change
        if i > 0:
            previous_prediction = predictions[-1]
            daily_return = (predicted_price / previous_prediction) - 1
            
            if abs(daily_return) > max_daily_change:
                print(f"WARNING: Detected unrealistic daily change of {daily_return:.2%}")
                # Limit to maximum allowed daily change
                predicted_price = previous_prediction * (1 + (max_daily_change * np.sign(daily_return)))
                print(f"Limiting prediction to {predicted_price:.2f} (max {max_daily_change:.0%} change)")
        
        # Also check against starting price for first few days
        cumulative_return = (predicted_price / last_price) - 1
        max_total_change = 0.50  # 50% maximum total change for a 5-day period
        if abs(cumulative_return) > max_total_change:
            print(f"WARNING: Detected unrealistic cumulative change of {cumulative_return:.2%}")
            # Limit to maximum allowed change from starting price
            predicted_price = last_price * (1 + (max_total_change * np.sign(cumulative_return)))
            print(f"Limiting prediction to {predicted_price:.2f} (max {max_total_change:.0%} total change)")
        
        # Calculate confidence interval (wider as we go further into future)
        base_uncertainty = 0.02  # 2% base uncertainty
        day_factor = 1 + (i * 0.5)  # Increases with each day
        uncertainty = base_uncertainty * day_factor * predicted_price
        confidence_interval = (predicted_price - uncertainty, predicted_price + uncertainty)
        
        # Save prediction
        predictions.append(predicted_price)
        confidence_intervals.append(confidence_interval)
        
        # Save prediction to reward system if provided
        if reward_system is not None:
            try:
                date_str = current_date.strftime('%Y-%m-%d')
                # Save prediction with model information
                reward_system.save_prediction(date_str, predicted_price)
                print(f"Saved prediction for {date_str} to reward system: {predicted_price:.2f} (Model: {reward_system.model_name})")
            except Exception as e:
                print(f"Error saving prediction to reward system: {e}")
        
        # Update the current_x for next prediction
        if scaler is not None:
            # Create a dummy array with the same shape as the training data
            dummy = np.zeros((1, len(features)))
            
            # Fill with the last values from current_x
            for j in range(len(features)):
                dummy[0, j] = current_x[0, -1, j]
            
            # Update the close price with our prediction
            # Check if predicted_price_normalized is available
            if predicted_price_normalized is not None:
                # Handle different types of predicted_price_normalized based on model type
                if is_dl_model:
                    # Deep learning models return arrays
                    dummy[0, close_idx] = predicted_price_normalized[0][0]
                else:
                    # Traditional ML models might return scalars or 1D arrays
                    if hasattr(predicted_price_normalized, 'shape') and len(predicted_price_normalized.shape) > 0:
                        # It's an array
                        dummy[0, close_idx] = predicted_price_normalized[0]
                    else:
                        # It's a scalar
                        dummy[0, close_idx] = predicted_price_normalized
            else:
                # If prediction failed, use the last known price with a small random change
                # This allows the loop to continue even if one prediction fails
                random_change = np.random.uniform(-0.005, 0.005)  # Small random change ±0.5%
                dummy[0, close_idx] = current_x[0, -1, close_idx] * (1 + random_change)
                print(f"Using fallback prediction due to error")
            
            # Get the final prediction value and add it to the sequence
            # This involves shifting the window left and adding new value at the end
            current_x[0, :-1, :] = current_x[0, 1:, :]
            current_x[0, -1, :] = dummy[0, :]
        else:
            # Simple update without scaling
            current_x[0, :-1, :] = current_x[0, 1:, :]
            # Fallback: just copy the last row and update the close price
            current_x[0, -1, :] = current_x[0, -2, :]
            current_x[0, -1, close_idx] = predicted_price
    
    # Check if future_dates is a scalar instead of a list
    # This can happen if the function had an early return
    if not isinstance(future_dates, list) or len(future_dates) == 0:
        print(f"Warning: future_dates is not a list or is empty. Creating new date list. Current value: {future_dates}")
        # Generate dates starting from last_date
        future_dates = []
        current_date = last_date
        for i in range(days_ahead):
            # Generate next weekday date
            while True:
                current_date = current_date + timedelta(days=1)
                # Skip weekends (0 = Monday, 6 = Sunday)
                if current_date.weekday() < 5:  # 0-4 are weekdays
                    break
            future_dates.append(current_date)
    
    # Make sure predictions and future_dates have the same length
    if len(predictions) != len(future_dates):
        print(f"Warning: predictions ({len(predictions)}) and future_dates ({len(future_dates)}) have different lengths. Adjusting...")
        # Use the shorter length
        min_length = min(len(predictions), len(future_dates))
        predictions = predictions[:min_length]
        future_dates = future_dates[:min_length]
    
    # If we have only one prediction but it's a float, convert it to a list
    if isinstance(predictions, (float, np.float64, np.float32)):
        predictions = [float(predictions)]
    
    # Initialize significant_days if it doesn't exist
    if 'significant_days' not in locals():
        significant_days = []
    
    # Return values according to the updated function documentation
    return predictions, confidence_intervals, significant_days, future_dates


def apply_psx_price_limits(predicted_price, reference_price, market_direction=None):
    """
    Apply Pakistan Stock Exchange (PSX) circuit breaker limits to predicted prices
    
    PSX has circuit breakers that limit daily price movements to +/- 7.5% of the previous day's close
    This function ensures predictions stay within these realistic limits
    
    Args:
        predicted_price (float): The predicted price
        reference_price (float): The reference price (usually previous day's close)
        market_direction (str): Market direction ('up', 'down', or None)
        
    Returns:
        float: The price after applying circuit breaker limits
    """
    # Default circuit breaker limits for PSX (7.5%)
    DEFAULT_LIMIT = 0.075
    
    # If reference price is not provided or invalid, return the predicted price as is
    if reference_price is None or not isinstance(reference_price, (int, float)) or reference_price <= 0:
        return predicted_price
    
    # Calculate upper and lower limits
    upper_limit = reference_price * (1 + DEFAULT_LIMIT)
    lower_limit = reference_price * (1 - DEFAULT_LIMIT)
    
    # Apply market direction bias if provided
    if market_direction == 'up':
        # In an upward market, bias toward the upper limit
        upper_limit = reference_price * (1 + DEFAULT_LIMIT)
        lower_limit = reference_price * (1 - (DEFAULT_LIMIT * 0.8))  # Less downside
    elif market_direction == 'down':
        # In a downward market, bias toward the lower limit
        upper_limit = reference_price * (1 + (DEFAULT_LIMIT * 0.8))  # Less upside
        lower_limit = reference_price * (1 - DEFAULT_LIMIT)
    
    # Apply limits
    if predicted_price > upper_limit:
        return upper_limit
    elif predicted_price < lower_limit:
        return lower_limit
    else:
        return predicted_price


def visualize_predictions_advanced(df, predictions, confidence_intervals, significant_days, symbol, future_dates, days_ahead, market_trend_info=None):
    """
    Visualize the predictions with confidence intervals and significant days
    
    Args:
        df: DataFrame with historical data
        predictions: Predicted prices
        confidence_intervals: Confidence intervals for predictions
        significant_days: Days where significant price changes are expected
        symbol: Stock symbol
        future_dates: Dates for the predictions
        days_ahead: Number of days ahead predicted
        market_trend_info: Tuple containing market trend information (trends, sentiment_score)
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
    valid_indices = []
    
    # Check if future_dates is iterable
    if isinstance(future_dates, (list, tuple, np.ndarray)) and not isinstance(future_dates, (float, np.float64, np.float32)):
        valid_indices = [i for i, date in enumerate(future_dates) 
                        if isinstance(date, datetime) and date.weekday() < 5]
    else:
        print(f"Warning: future_dates is not iterable (type: {type(future_dates)}). Skipping visualization.")
        # Return early if we can't visualize
        return None
    
    if valid_indices:
        # Make sure predictions follow PSX rules sequentially for visualization
        # Create a copy to avoid modifying the original predictions
        last_close_price = get_column_case_insensitive(df, 'Close').iloc[-1]
        visualized_predictions = predictions.copy()
        visualized_confidence = confidence_intervals.copy()
        
        # Get market direction if available
        if market_trend_info is not None and isinstance(market_trend_info, tuple) and len(market_trend_info) > 0:
            market_direction = market_trend_info[0].get('market_direction', 'unknown') if isinstance(market_trend_info[0], dict) else 'unknown'
        else:
            market_direction = 'unknown'
        
        # Apply price limits to predictions used for visualization
        visualized_predictions[0] = apply_psx_price_limits(visualized_predictions[0], last_close_price, market_direction=market_direction)
        for i in range(1, len(visualized_predictions)):
            visualized_predictions[i] = apply_psx_price_limits(visualized_predictions[i], visualized_predictions[i-1], market_direction=market_direction)
        
        # Apply limits to confidence intervals
        for i in range(len(visualized_confidence)):
            if i == 0:
                lower_limit = apply_psx_price_limits(visualized_confidence[i][0], last_close_price, market_direction=market_direction)
                upper_limit = apply_psx_price_limits(visualized_confidence[i][1], last_close_price, market_direction=market_direction)
            else:
                lower_limit = apply_psx_price_limits(visualized_confidence[i][0], visualized_predictions[i-1], market_direction=market_direction)
                upper_limit = apply_psx_price_limits(visualized_confidence[i][1], visualized_predictions[i-1], market_direction=market_direction)
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
        if significant_days:
            if isinstance(significant_days, dict):
                # Handle dictionary format
                for movement_type, days in significant_days.items():
                    for day in days:
                        if 0 <= day < len(valid_indices):
                            idx = valid_indices.index(day) if day in valid_indices else None
                            if idx is not None and idx < len(valid_pred):
                                marker_color = 'green' if movement_type == 'up' else 'red'
                                plt.plot(valid_dates[idx], valid_pred[idx], marker='o', markersize=8, color=marker_color)
            elif isinstance(significant_days, list):
                # Handle list format - assume all are significant without specific type
                for day in significant_days:
                    if 0 <= day < len(valid_indices):
                        idx = valid_indices.index(day) if day in valid_indices else None
                        if idx is not None and idx < len(valid_pred):
                            plt.plot(valid_dates[idx], valid_pred[idx], marker='o', markersize=8, color='purple')
    
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
    if significant_days:
        if isinstance(significant_days, dict):
            # Handle dictionary format
            for movement_type, days in significant_days.items():
                for day in days:
                    if 0 <= day < len(future_dates):
                        date = future_dates[day]
                        # Format the date safely using our helper function
                        date_str = format_date_safely(date)
                        print(f"  {date_str}: {movement_type}")
        elif isinstance(significant_days, list):
            # Handle list format - assume all are significant without specific type
            for day in significant_days:
                if 0 <= day < len(future_dates):
                    date = future_dates[day]
                    # Format the date safely using our helper function
                    date_str = format_date_safely(date)
                    print(f"  {date_str}: significant movement")
    
    plt.close()


def predict_future(model, df, scaler, features, days=7, window_size=10, is_dl_model=True, 
               smoothing_factor=0.7, apply_market_trends=True, market_adjustment_factor=0.03, 
               market_trend_info=None, reward_system=None):
    """
    Make predictions for future days
    
    Args:
        model (tf.keras.Model): Trained model
        df (pd.DataFrame): DataFrame with historical data
        scaler (sklearn.preprocessing.MinMaxScaler): Scaler used for normalization
        features (list): List of features used for prediction
        days (int): Number of days to predict ahead
        window_size (int): Window size for sequences
        is_dl_model (bool): Whether the model is a deep learning model (True) or traditional ML model (False)
        smoothing_factor (float): Factor for smoothing predictions (0.0-1.0)
        apply_market_trends (bool): Whether to apply market trend adjustments
        market_adjustment_factor (float): Factor for market trend adjustments
        market_trend_info (dict): Market trend information
        reward_system (PredictionRewardSystem): System for tracking and rewarding predictions
        
    Returns:
        tuple: (predictions, dates) - Array of predictions and corresponding dates
    """
    try:
        # Prepare data for prediction
        X, scaled_data = prepare_prediction_data(df, features, scaler, window_size)
        
        if X is None or len(X) == 0:
            print("Error: Failed to prepare prediction data")
            record_failed_prediction(symbol, "Failed to prepare prediction data")
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
        
        # Calculate the next weekday
        next_date = last_date + timedelta(days=1)
        while next_date.weekday() >= 5:  # Skip weekend days
            next_date += timedelta(days=1)
            
        print(f"Predictions will start from: {next_date.strftime('%Y-%m-%d')} (next weekday)")
        
        future_dates = generate_weekday_dates(last_date, days)
        
        # Get market trend information if not provided
        if market_trend_info is None and apply_market_trends:
            try:
                market_trend_info = get_market_trend_analysis(symbol)
            except Exception as e:
                print(f"Warning: Error getting market trend information: {e}")
                print("Continuing without market trend adjustments")
                apply_market_trends = False
        
        # If we have a reward system, update its model information
        if reward_system is not None:
            print(f"Using model '{model_name}' for predictions")
            # Update reward system with model info for future reference
            if hasattr(reward_system, 'set_model_info') and callable(getattr(reward_system, 'set_model_info', None)):
                try:
                    reward_system.set_model_info(model_type, model_name)
                except Exception as e:
                    print(f"Warning: Could not update reward system with model info: {e}")
        
        # Predict future prices
        predictions, confidence_intervals, significant_days, future_dates = predict_future_prices_advanced(
            model, X, df, scaler, features, 
            days_ahead=days, 
            smoothing_factor=smoothing_factor,
            apply_market_trends=apply_market_trends,
            market_adjustment_factor=market_adjustment_factor,
            market_trend_info=market_trend_info,
            reward_system=reward_system,
            window_size=window_size,
            is_dl_model=is_dl_model  # Pass the is_dl_model parameter
        )
        
        if predictions is None:
            print("Error: Failed to make predictions")
            record_failed_prediction(symbol, "Failed to make predictions")
            return None
        
        # Check for different volatility regimes
        if df is not None and len(df) >= 30:
            # Calculate historical volatility over different windows
            try:
                print("\nHistorical volatility analysis:")
                close_prices = get_column_case_insensitive(df, 'Close').values[-30:]
                returns = np.diff(close_prices) / close_prices[:-1]
                volatility = np.std(returns) * 100  # Convert to percentage
                print(f"30-day historical volatility: {volatility:.2f}%")
            except Exception as e:
                print(f"Warning: Error during volatility analysis: {e}")
        
        # Apply final validation and reality check if enabled
        if reality_check:
            print("\nPerforming final reality check on predictions...")
            try:
                # Get the closing price from whichever data source is available
                if df is not None:
                    last_close_price = get_column_case_insensitive(df, 'Close').iloc[-1]
                else:
                    # If df is not available, try to use close from df_with_indicators
                    last_close_price = get_column_case_insensitive(df_with_indicators, 'Close').iloc[-1]
                
                is_realistic, warnings, adjusted_predictions = validate_predictions_against_reality(
                    predictions, 
                    last_close_price, 
                    df=df if df is not None else df_with_indicators
                )
                
                if not is_realistic and len(warnings) > 0:
                    print("Adjusting predictions after reality check...")
                    predictions = adjusted_predictions
                    confidence_intervals = [(adj_pred * 0.95, adj_pred * 1.05) for adj_pred in adjusted_predictions]
                    
                    # Recalculate significant days with the adjusted predictions
                    significant_days = identify_significant_movements(predictions, threshold_pct=threshold)
            except Exception as e:
                print(f"Warning: Error during reality check: {e}")
                print("Continuing with original predictions")
        
        # Try a second model with different parameters if predictions look unrealistic
        if predictions is not None and len(predictions) >= days_ahead and apply_market_trends:
            # Check if all predictions are going in the same direction
            all_increasing = all(predictions[i] >= predictions[i-1] for i in range(1, len(predictions)))
            all_decreasing = all(predictions[i] <= predictions[i-1] for i in range(1, len(predictions)))
            
            if (all_increasing or all_decreasing) and days_ahead >= 4:
                print("\nDetected potentially unrealistic prediction pattern (all increasing or all decreasing)")
                print("Trying alternative prediction with stricter mean reversion...")
                
                # Try again with a lower smoothing factor and market adjustment
                new_smoothing = max(0.3, smoothing_factor - 0.2)
                new_market_adj = market_adjustment_factor * 0.5
                
                # Get a second opinion
                new_predictions, new_intervals, new_significant, new_dates = predict_future_prices_advanced(
                    model, X, df_with_indicators, scaler, features, days_ahead=days_ahead,
                    smoothing_factor=new_smoothing,
                    apply_market_trends=apply_market_trends,
                    market_adjustment_factor=new_market_adj,
                    market_trend_info=market_trend_info,
                    reward_system=reward_system,
                    window_size=window_size,
                    is_dl_model=is_dl_model  # Pass the is_dl_model parameter
                )
                
                # Check if the new prediction passes a basic reality check
                try:
                    # Get the closing price from whichever data source is available
                    if df is not None:
                        last_close_price = get_column_case_insensitive(df, 'Close').iloc[-1]
                    else:
                        # If df is not available, try to use close from df_with_indicators
                        last_close_price = get_column_case_insensitive(df_with_indicators, 'Close').iloc[-1]
                    
                    is_realistic, _, _ = validate_predictions_against_reality(
                        new_predictions, 
                        last_close_price, 
                        df=df if df is not None else df_with_indicators
                    )
                    
                    if is_realistic:
                        print("Using alternative prediction with stricter parameters")
                        predictions = new_predictions
                        confidence_intervals = new_intervals
                        significant_days = new_significant
                    else:
                        print("Alternative prediction also didn't pass reality check. Using original with adjustments.")
                except Exception as e:
                    print(f"Warning: Error checking alternative predictions: {e}")
                    print("Continuing with original predictions")
        
        # Create company-specific model directory
        company_model_dir = os.path.join('models', symbol)
        os.makedirs(company_model_dir, exist_ok=True)
        
        # Visualize predictions
        visualize_predictions_advanced(
            df_with_indicators, predictions, confidence_intervals, 
            significant_days, symbol, future_dates, days_ahead, market_trend_info
        )
        
        # Send predictions to API
        try:
            # Get the last close price
            if df_with_indicators is not None:
                last_close_price = get_column_case_insensitive(df_with_indicators, 'Close').iloc[-1]
            elif df is not None:
                last_close_price = get_column_case_insensitive(df, 'Close').iloc[-1]
            else:
                last_close_price = None
                
            # Send predictions to API
            send_predictions_to_api(symbol, predictions, future_dates, last_close_price, df_with_indicators)
        except Exception as e:
            print(f"Warning: Error sending predictions to API: {e}")
        
        # Print results
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


def main(symbol=None, window_size=10, days_ahead=5, threshold=2.0, 
         smoothing_factor=0.7, apply_market_trends=True, market_adjustment_factor=0.03,
         market_trend_info=None, reward_system=None, df_with_indicators=None,
         reality_check=True):
    """
    Main function to predict future stock prices
    
    Args:
        symbol (str): Stock symbol
        window_size (int): Number of previous days to use for prediction
        days_ahead (int): Number of days ahead to predict
        threshold (float): Threshold percentage for significant movements
        smoothing_factor (float): Factor for smoothing predictions (0.0-1.0)
        apply_market_trends (bool): Whether to apply market trend adjustments
        market_adjustment_factor (float): Factor controlling how much market trends affect predictions
        market_trend_info (tuple): Market trend information (optional)
        reward_system (PredictionRewardSystem): Optional reward system instance
        df_with_indicators (pd.DataFrame): Pre-calculated technical indicators (optional)
        reality_check (bool): Whether to validate predictions against historical patterns
        
    Returns:
        None
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
            
        # Determine if the model is a deep learning model
        # We need to check if we're using a model selected by the model_selection system
        is_dl_model = True  # Default to True for backward compatibility
        best_model_info_path = os.path.join('models', symbol, "best_model_info.json")
        if os.path.exists(best_model_info_path):
            try:
                with open(best_model_info_path, 'r') as f:
                    best_model_info = json.load(f)
                is_dl_model = best_model_info.get('is_dl_model', True)
                model_name = best_model_info.get('model_name', "Unknown")
                model_type = best_model_info.get('best_model_type', "Unknown")
                print(f"Model determined from best_model_info: {model_name} ({model_type})")
                print(f"Model type: {'Deep Learning' if is_dl_model else 'Traditional ML'}")
            except Exception as e:
                print(f"Warning: Could not determine model type from best_model_info: {e}")
                print("Defaulting to Deep Learning model type")
                model_name = "Unknown"
                model_type = "Unknown"
        else:
            # If no best_model_info, try to guess based on file extension
            if hasattr(model, '__module__') and 'keras' in model.__module__:
                is_dl_model = True
                model_name = "Deep Learning (Keras)"
                model_type = "Unknown Deep Learning"
                print("Model determined to be Deep Learning based on model class")
            elif hasattr(model, 'predict_proba') or hasattr(model, 'feature_importances_'):
                is_dl_model = False
                model_name = "Traditional ML"
                model_type = "Unknown Traditional ML"
                print("Model determined to be Traditional ML based on model attributes")
            else:
                print("Could not determine model type, defaulting to Deep Learning")
                model_name = "Unknown Model Type"
                model_type = "Unknown"
                
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
            # Make sure df is defined even when using pre-calculated indicators
            try:
                # Try to use df_with_indicators as df, as it should have the core data too
                df = df_with_indicators.copy()
            except Exception as e:
                print(f"Warning: Could not create df from df_with_indicators: {e}")
                # In this case, we should load the stock data separately
                try:
                    df = load_stock_data(symbol, apply_rules=True)
                    if df is None or len(df) == 0:
                        print(f"Warning: Could not load stock data for {symbol}, volatility analysis may be skipped")
                        # Create a minimal df to avoid errors
                        df = df_with_indicators[['date', 'close']].copy() if 'date' in df_with_indicators.columns and 'close' in df_with_indicators.columns else None
                except Exception as load_error:
                    print(f"Warning: Error loading stock data: {load_error}")
                    # Create a minimal df to avoid errors
                    df = df_with_indicators[['date', 'close']].copy() if 'date' in df_with_indicators.columns and 'close' in df_with_indicators.columns else None
        
        if df_with_indicators is None or len(df_with_indicators) == 0:
            print(f"Error: Failed to calculate technical indicators for {symbol}")
            record_failed_prediction(symbol, "Technical indicators calculation returned empty result")
            return None
        
        # Get market trend information if not provided
        if market_trend_info is None and apply_market_trends:
            try:
                market_trend_info = get_market_trend_analysis(symbol)
            except Exception as e:
                print(f"Warning: Error getting market trend information: {e}")
                print("Continuing without market trend adjustments")
                apply_market_trends = False
        
        # Prepare data for prediction
        X, scaled_data = prepare_prediction_data(df_with_indicators, features, scaler, window_size)
        
        if X is None or len(X) == 0:
            print("Error: Failed to prepare prediction data")
            record_failed_prediction(symbol, "Failed to prepare prediction data")
            return None
        
        # If we have a reward system, update its model information
        if reward_system is not None:
            print(f"Using model '{model_name}' for predictions")
            # Update reward system with model info for future reference
            if hasattr(reward_system, 'set_model_info') and callable(getattr(reward_system, 'set_model_info', None)):
                try:
                    reward_system.set_model_info(model_type, model_name)
                except Exception as e:
                    print(f"Warning: Could not update reward system with model info: {e}")
        
        # Predict future prices
        predictions, confidence_intervals, significant_days, future_dates = predict_future_prices_advanced(
            model, X, df_with_indicators, scaler, features, 
            days_ahead=days_ahead, 
            smoothing_factor=smoothing_factor,
            apply_market_trends=apply_market_trends,
            market_adjustment_factor=market_adjustment_factor,
            market_trend_info=market_trend_info,
            reward_system=reward_system,
            window_size=window_size,
            is_dl_model=is_dl_model  # Pass the is_dl_model parameter
        )
        
        if predictions is None:
            print("Error: Failed to make predictions")
            record_failed_prediction(symbol, "Failed to make predictions")
            return None
        
        # Check for different volatility regimes
        if df is not None and len(df) >= 30:
            # Calculate historical volatility over different windows
            try:
                print("\nHistorical volatility analysis:")
                close_prices = get_column_case_insensitive(df, 'Close').values[-30:]
                returns = np.diff(close_prices) / close_prices[:-1]
                volatility = np.std(returns) * 100  # Convert to percentage
                print(f"30-day historical volatility: {volatility:.2f}%")
            except Exception as e:
                print(f"Warning: Error during volatility analysis: {e}")
        
        # Apply final validation and reality check if enabled
        if reality_check:
            print("\nPerforming final reality check on predictions...")
            try:
                # Get the closing price from whichever data source is available
                if df is not None:
                    last_close_price = get_column_case_insensitive(df, 'Close').iloc[-1]
                else:
                    # If df is not available, try to use close from df_with_indicators
                    last_close_price = get_column_case_insensitive(df_with_indicators, 'Close').iloc[-1]
                
                is_realistic, warnings, adjusted_predictions = validate_predictions_against_reality(
                    predictions, 
                    last_close_price, 
                    df=df if df is not None else df_with_indicators
                )
                
                if not is_realistic and len(warnings) > 0:
                    print("Adjusting predictions after reality check...")
                    predictions = adjusted_predictions
                    confidence_intervals = [(adj_pred * 0.95, adj_pred * 1.05) for adj_pred in adjusted_predictions]
                    
                    # Recalculate significant days with the adjusted predictions
                    significant_days = identify_significant_movements(predictions, threshold_pct=threshold)
            except Exception as e:
                print(f"Warning: Error during reality check: {e}")
                print("Continuing with original predictions")
        
        # Try a second model with different parameters if predictions look unrealistic
        if predictions is not None and len(predictions) >= days_ahead and apply_market_trends:
            # Check if all predictions are going in the same direction
            all_increasing = all(predictions[i] >= predictions[i-1] for i in range(1, len(predictions)))
            all_decreasing = all(predictions[i] <= predictions[i-1] for i in range(1, len(predictions)))
            
            if (all_increasing or all_decreasing) and days_ahead >= 4:
                print("\nDetected potentially unrealistic prediction pattern (all increasing or all decreasing)")
                print("Trying alternative prediction with stricter mean reversion...")
                
                # Try again with a lower smoothing factor and market adjustment
                new_smoothing = max(0.3, smoothing_factor - 0.2)
                new_market_adj = market_adjustment_factor * 0.5
                
                # Get a second opinion
                new_predictions, new_intervals, new_significant, new_dates = predict_future_prices_advanced(
                    model, X, df_with_indicators, scaler, features, days_ahead=days_ahead,
                    smoothing_factor=new_smoothing,
                    apply_market_trends=apply_market_trends,
                    market_adjustment_factor=new_market_adj,
                    market_trend_info=market_trend_info,
                    reward_system=reward_system,
                    window_size=window_size,
                    is_dl_model=is_dl_model  # Pass the is_dl_model parameter
                )
                
                # Check if the new prediction passes a basic reality check
                try:
                    # Get the closing price from whichever data source is available
                    if df is not None:
                        last_close_price = get_column_case_insensitive(df, 'Close').iloc[-1]
                    else:
                        # If df is not available, try to use close from df_with_indicators
                        last_close_price = get_column_case_insensitive(df_with_indicators, 'Close').iloc[-1]
                    
                    is_realistic, _, _ = validate_predictions_against_reality(
                        new_predictions, 
                        last_close_price, 
                        df=df if df is not None else df_with_indicators
                    )
                    
                    if is_realistic:
                        print("Using alternative prediction with stricter parameters")
                        predictions = new_predictions
                        confidence_intervals = new_intervals
                        significant_days = new_significant
                    else:
                        print("Alternative prediction also didn't pass reality check. Using original with adjustments.")
                except Exception as e:
                    print(f"Warning: Error checking alternative predictions: {e}")
                    print("Continuing with original predictions")
        
        # Create company-specific model directory
        company_model_dir = os.path.join('models', symbol)
        os.makedirs(company_model_dir, exist_ok=True)
        
        # Visualize predictions
        visualize_predictions_advanced(
            df_with_indicators, predictions, confidence_intervals, 
            significant_days, symbol, future_dates, days_ahead, market_trend_info
        )
        
        # Send predictions to API
        try:
            # Get the last close price
            if df_with_indicators is not None:
                last_close_price = get_column_case_insensitive(df_with_indicators, 'Close').iloc[-1]
            elif df is not None:
                last_close_price = get_column_case_insensitive(df, 'Close').iloc[-1]
            else:
                last_close_price = None
                
            # Send predictions to API
            send_predictions_to_api(symbol, predictions, future_dates, last_close_price, df_with_indicators)
        except Exception as e:
            print(f"Warning: Error sending predictions to API: {e}")
        
        # Print results
        print(f"Successfully completed prediction for {symbol}")
        return predictions, confidence_intervals, significant_days, future_dates
    
    except Exception as e:
        print(f"Unexpected error during prediction for {symbol}: {e}")
        record_failed_prediction(symbol, f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None


def identify_significant_movements(predictions, threshold_pct=2.0):
    """
    Identify days with significant price movements in the predictions.
    
    Args:
        predictions (list): List of predicted prices
        threshold_pct (float): Threshold percentage for significant movement
        
    Returns:
        list: Indices of days with significant movements
    """
    if len(predictions) < 2:
        return []
    
    significant_days = []
    
    # Calculate daily percentage changes
    for i in range(1, len(predictions)):
        prev_price = predictions[i-1]
        curr_price = predictions[i]
        
        # Avoid division by zero
        if prev_price == 0:
            continue
            
        pct_change = abs((curr_price - prev_price) / prev_price * 100)
        
        # If percentage change exceeds threshold, mark as significant
        if pct_change >= threshold_pct:
            significant_days.append(i)
    
    return significant_days


def send_predictions_to_api(symbol, predictions, future_dates, last_close_price=None, df=None):
    """
    Send predictions to an API endpoint
    
    Args:
        symbol (str): Stock symbol
        predictions (list): List of predicted prices
        future_dates (list): List of dates for the predictions
        last_close_price (float): Last known closing price
        df (pd.DataFrame): DataFrame with historical data
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # API endpoint URL - replace with actual API endpoint
        api_url = "https://stocks.wajipk.com/api/predictions"
        
        # Format dates to strings
        date_strings = [format_date_safely(date) for date in future_dates]
        
        # Create payload
        payload = {
            "symbol": symbol,
            "predictions": [
                {"date": date, "price": float(price)} 
                for date, price in zip(date_strings, predictions)
            ],
            "last_close_price": float(last_close_price) if last_close_price is not None else None,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Add metadata if available
        if df is not None:
            try:
                # Get the last row of data for metadata
                last_row = df.iloc[-1]
                metadata = {
                    "last_date": format_date_safely(last_row.get('date', None)),
                    "last_volume": float(last_row.get('volume', 0)),
                    "last_high": float(last_row.get('high', last_close_price)),
                    "last_low": float(last_row.get('low', last_close_price))
                }
                payload["metadata"] = metadata
            except Exception as e:
                print(f"Warning: Error adding metadata to API payload: {e}")
        
        # Print payload for debugging
        print(f"Sending predictions to API for {symbol}...")
        
        # Uncomment to actually send to API
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        print(f"Successfully sent predictions to API. Response: {response.status_code}")
        return True
        
        # For now, just print that we would send this data
        print("API integration is disabled. Would have sent the following data:")
        print(f"  Symbol: {symbol}")
        print(f"  Dates: {date_strings}")
        print(f"  Predictions: {[round(float(p), 2) for p in predictions]}")
        return True
        
    except Exception as e:
        print(f"Error sending predictions to API: {e}")
        return False


if __name__ == "__main__":
    main() 