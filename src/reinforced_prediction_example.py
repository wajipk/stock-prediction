import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reinforced_model import ReinforcedStockModel
from src.preprocessing import load_stock_data, add_technical_indicators, prepare_train_test_data
from src.data_collection import fetch_stock_data, save_stock_data


def train_reinforced_model(symbol, window_size=10, epochs=100, batch_size=32, learning_rate=0.001, 
                          dropout_rate=0.3, base_model_type='lstm', error_threshold=0.05):
    """
    Train a reinforced stock model for the given symbol
    
    Args:
        symbol (str): Stock symbol
        window_size (int): Window size for sequential data
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for model training
        dropout_rate (float): Dropout rate for regularization
        base_model_type (str): Base model type ('lstm', 'bilstm', 'gru', 'cnn_lstm', 'advanced')
        error_threshold (float): Error threshold for model adaptation
        
    Returns:
        ReinforcedStockModel: Trained reinforced model
    """
    print(f"Training reinforced model for {symbol} using {base_model_type} as base model...")
    
    # Step 1: Get data
    print("Loading and preparing data...")
    
    # Check if data exists in data directory, otherwise fetch it
    data_file = os.path.join('data', f'{symbol}.csv')
    if not os.path.exists(data_file):
        print(f"Fetching data for {symbol}...")
        df = fetch_stock_data(symbol, days=1000)
        save_stock_data(df, symbol)
    else:
        df = load_stock_data(symbol)
    
    # Add technical indicators
    df_with_indicators = add_technical_indicators(df)
    
    # Ensure we have a date column and it's in the right format
    # This is needed for prepare_train_test_data
    if 'date' not in df_with_indicators.columns and df_with_indicators.index.name == 'date':
        # If date is the index, reset it to be a column
        df_with_indicators = df_with_indicators.reset_index()
    
    # Make sure we're using only numeric features for model training
    # But keep the date column for proper sorting
    print("Ensuring all features are numeric while preserving the date column...")
    date_column = None
    if 'date' in df_with_indicators.columns:
        # Save the date column
        date_column = df_with_indicators['date'].copy()
        
        # Select numeric columns
        numeric_cols = df_with_indicators.select_dtypes(include=[np.number]).columns
        print(f"Selected {len(numeric_cols)} numeric features.")
        
        # Create a new dataframe with date and numeric columns
        df_processed = pd.DataFrame()
        df_processed['date'] = date_column
        for col in numeric_cols:
            df_processed[col] = df_with_indicators[col]
        
        df_with_indicators = df_processed
    
    # Prepare training and test data
    try:
        X_train, X_test, y_train, y_test, scaler, features = prepare_train_test_data(
            df_with_indicators, window_size=window_size, test_size=0.2
        )
        
        # Check if we got valid data back
        if X_train is None or X_test is None or y_train is None or y_test is None:
            print("Failed to prepare training data. Check the errors above.")
            return None
            
        # Create 20% validation set from training data
        val_size = int(len(X_train) * 0.2)
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        X_train = X_train[:-val_size]
        y_train = y_train[:-val_size]
        
        print(f"Data shapes:")
        print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
        print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    except Exception as e:
        print(f"Error preparing training data: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Step 2: Create and train the reinforced model
    model = ReinforcedStockModel(
        symbol=symbol,
        window_size=window_size,
        base_model_type=base_model_type,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        error_threshold=error_threshold
    )
    
    # Train the base model
    print("\nTraining base model...")
    model.train_base_model(
        X_train, y_train, 
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Step 3: Evaluate the model on test data
    print("\nEvaluating model on test data...")
    predictions = []
    actual_prices = []
    
    # Make predictions on test data
    for i in range(len(X_test)):
        # For demonstration, only use error correction after some predictions
        # In real usage, you would use error correction from the start
        apply_correction = i > window_size  # Use correction after collecting some error history
        
        # Make a prediction (without error correction initially)
        pred = model.predict(X_test[i], apply_correction=apply_correction)
        predictions.append(pred)
        actual = y_test[i]
        actual_prices.append(actual)
        
        # Update the model with the actual price (simulating getting market data the next day)
        # In real usage, you would do this as actual prices become available
        # For testing, we're using the test set actual prices
        date = (datetime.now() - timedelta(days=len(X_test) - i)).strftime('%Y-%m-%d')
        model.update_with_actual_price(date, actual)
        
        # Print progress
        if i % 10 == 0:
            print(f"Processed {i} of {len(X_test)} test samples")
    
    # Convert to numpy arrays for easier calculations
    predictions = np.array(predictions)
    actual_prices = np.array(actual_prices)
    
    # Calculate performance metrics
    mse = np.mean((predictions - actual_prices) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actual_prices))
    mape = np.mean(np.abs((predictions - actual_prices) / actual_prices)) * 100
    
    print("\nPerformance Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")
    
    # Validate model quality and retrain if necessary
    negative_predictions = (predictions < 0).sum()
    if negative_predictions > 0:
        print(f"Warning: Model generated {negative_predictions} negative price predictions out of {len(predictions)}")
        
    if mape > 50:  # If MAPE is very high, model might be unreliable
        print(f"Warning: Model has high error rate (MAPE: {mape:.2f}%). Consider retraining with different parameters.")
        
    # Check if predictions are reasonable
    price_mean = np.mean(actual_prices)
    price_std = np.std(actual_prices)
    unreasonable_predictions = ((predictions < price_mean - 3 * price_std) | (predictions > price_mean + 3 * price_std)).sum()
    
    if unreasonable_predictions > len(predictions) * 0.1:  # If more than 10% are unreasonable
        print(f"Warning: {unreasonable_predictions} predictions are outside reasonable range.")
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices, label='Actual Prices', color='blue')
    plt.plot(predictions, label='Predicted Prices', color='red')
    plt.title(f'Reinforced Model Predictions for {symbol}')
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig(os.path.join('models', symbol, 'reinforced_predictions.png'))
    plt.close()
    
    # Get prediction metrics from reward system
    metrics = model.get_recent_prediction_metrics()
    print("\nPrediction Accuracy Metrics from Reward System:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Save all model components
    model.save_models()
    
    # Return both the model and the features list for consistent prediction
    return model, features


def main():
    """
    Main function to run the reinforced model training and testing
    """
    parser = argparse.ArgumentParser(description='Train and test a reinforced stock prediction model')
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol')
    parser.add_argument('--window', type=int, default=10, help='Window size for sequences')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for model training')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate for model regularization')
    parser.add_argument('--base_model', type=str, default='lstm', 
                       choices=['lstm', 'bilstm', 'gru', 'cnn_lstm', 'advanced'],
                       help='Base model type')
    parser.add_argument('--error_threshold', type=float, default=0.05, 
                       help='Error threshold for model adaptation')
    
    args = parser.parse_args()
    
    # Ensure required directories exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Train the reinforced model
    train_reinforced_model(
        symbol=args.symbol,
        window_size=args.window,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout,
        base_model_type=args.base_model,
        error_threshold=args.error_threshold
    )


if __name__ == "__main__":
    main() 