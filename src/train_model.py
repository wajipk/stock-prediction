import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, GRU, Conv1D, MaxPooling1D, Flatten, Concatenate, BatchNormalization, Attention, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
import time
from datetime import datetime

from src.preprocessing import load_stock_data, add_technical_indicators, prepare_train_test_data
from src.prediction_reward_system import PredictionRewardSystem
from src.model_selection import ModelSelector, ModelFactory, ModelEvaluator, MODEL_REGISTRY


def build_advanced_model(input_shape, dropout_rate=0.3, learning_rate=0.001):
    """
    Build an advanced deep learning model for stock price prediction
    
    Args:
        input_shape (tuple): Shape of input data (window_size, n_features)
        dropout_rate (float): Dropout rate for regularization
        learning_rate (float): Learning rate for model training
        
    Returns:
        tf.keras.Model: Compiled advanced model
    """
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Convolutional branch - capturing local patterns
    conv_layer = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
    conv_layer = BatchNormalization()(conv_layer)
    conv_layer = MaxPooling1D(pool_size=2)(conv_layer)
    conv_layer = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(conv_layer)
    conv_layer = BatchNormalization()(conv_layer)
    conv_layer = MaxPooling1D(pool_size=2)(conv_layer)
    conv_layer = Flatten()(conv_layer)
    
    # LSTM branch - capturing temporal dependencies
    lstm_layer = Bidirectional(LSTM(units=64, return_sequences=True))(inputs)
    lstm_layer = Dropout(dropout_rate)(lstm_layer)
    lstm_layer = Bidirectional(LSTM(units=64, return_sequences=False))(lstm_layer)
    lstm_layer = Dropout(dropout_rate)(lstm_layer)
    
    # GRU branch - another recurrent approach
    gru_layer = GRU(units=64, return_sequences=True)(inputs)
    gru_layer = Dropout(dropout_rate)(gru_layer)
    gru_layer = GRU(units=64, return_sequences=False)(gru_layer)
    gru_layer = Dropout(dropout_rate)(gru_layer)
    
    # Merge the branches
    merged = Concatenate()([conv_layer, lstm_layer, gru_layer])
    
    # Dense layers for final prediction
    dense = Dense(units=128, activation='relu')(merged)
    dense = BatchNormalization()(dense)
    dense = Dropout(dropout_rate)(dense)
    dense = Dense(units=64, activation='relu')(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(dropout_rate/2)(dense)
    
    # Output layer
    outputs = Dense(units=1)(dense)
    
    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    return model


def build_lstm_model(input_shape, dropout_rate=0.2):
    """
    Build an LSTM model for stock price prediction (legacy function, use build_advanced_model instead)
    
    Args:
        input_shape (tuple): Shape of input data (window_size, n_features)
        dropout_rate (float): Dropout rate for regularization
        
    Returns:
        tf.keras.Model: Compiled LSTM model
    """
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(units=50, return_sequences=False),
        Dropout(dropout_rate),
        Dense(units=25),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model


def train_model(X_train, y_train, X_test, y_test, model_dir='models', symbol='STOCK', epochs=100, batch_size=32, learning_rate=0.001, dropout_rate=0.3, use_legacy_model=False, reward_system=None):
    """
    Train the advanced model
    
    Args:
        X_train (np.array): Training features
        y_train (np.array): Training target
        X_test (np.array): Testing features
        y_test (np.array): Testing target
        model_dir (str): Directory to save the model
        symbol (str): Stock symbol
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for model training
        dropout_rate (float): Dropout rate for model regularization
        use_legacy_model (bool): Use the legacy LSTM model instead of the advanced model
        reward_system (PredictionRewardSystem): Prediction reward system for tracking and improving predictions
        
    Returns:
        tuple: (model, history)
    """
    # Create base model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Create company-specific model directory
    company_model_dir = os.path.join(model_dir, symbol)
    os.makedirs(company_model_dir, exist_ok=True)
    
    # Detailed diagnostic info
    print("-" * 50)
    print("DIAGNOSTIC INFO:")
    print(f"X_train type: {type(X_train)}, shape: {X_train.shape}")
    print(f"y_train type: {type(y_train)}, shape: {y_train.shape if hasattr(y_train, 'shape') else 'No shape attribute'}")
    print(f"X_test type: {type(X_test)}, shape: {X_test.shape}")
    print(f"y_test type: {type(y_test)}, shape: {y_test.shape if hasattr(y_test, 'shape') else 'No shape attribute'}")
    
    # Ensure y_train and y_test are properly shaped numpy arrays
    if not isinstance(y_train, np.ndarray):
        print("Converting y_train to numpy array")
        y_train = np.array(y_train)
    
    if not isinstance(y_test, np.ndarray):
        print("Converting y_test to numpy array")
        y_test = np.array(y_test)
    
    # CRITICAL FIX: Check for data cardinality issues and fix them
    if len(X_train) != len(y_train):
        print(f"CRITICAL WARNING: Data cardinality mismatch in training data! X_train has {len(X_train)} samples while y_train has {len(y_train)} samples.")
        print("Fixing by ensuring both arrays have the same number of samples...")
        min_samples = min(len(X_train), len(y_train))
        X_train = X_train[:min_samples]
        y_train = y_train[:min_samples]
        print(f"After fix: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    
    if len(X_test) != len(y_test):
        print(f"CRITICAL WARNING: Data cardinality mismatch in test data! X_test has {len(X_test)} samples while y_test has {len(y_test)} samples.")
        print("Fixing by ensuring both arrays have the same number of samples...")
        min_samples = min(len(X_test), len(y_test))
        X_test = X_test[:min_samples]
        y_test = y_test[:min_samples]
        print(f"After fix: X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    # Ensure 1D shape for y arrays if needed
    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        print(f"Reshaping y_train from {y_train.shape} to ({y_train.shape[0]},)")
        y_train = y_train.reshape(-1)
        
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        print(f"Reshaping y_test from {y_test.shape} to ({y_test.shape[0]},)")
        y_test = y_test.reshape(-1)
    
    print("AFTER RESHAPING:")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    print("-" * 50)
    
    # Apply any adjustments suggested by the reward system
    if reward_system is not None:
        suggestions = reward_system.suggest_model_adjustments()
        if suggestions['needs_adjustment']:
            print(f"Applying model adjustments based on reward system: {suggestions['reason']}")
            
            # Adjust learning rate if suggested
            if suggestions['learning_rate_adjustment'] != 0:
                adjusted_learning_rate = learning_rate + suggestions['learning_rate_adjustment']
                print(f"Adjusting learning rate from {learning_rate} to {adjusted_learning_rate}")
                learning_rate = adjusted_learning_rate
            
            # Adjust dropout rate based on performance
            metrics = reward_system.get_overall_accuracy()
            if metrics['mean_accuracy'] is not None and metrics['mean_accuracy'] > 0.1:
                print(f"Adjusting dropout rate from {dropout_rate} to {dropout_rate * 1.1}")
                dropout_rate = min(0.5, dropout_rate * 1.1)  # Increase dropout but cap at 0.5

    # Build and compile model
    if use_legacy_model:
        print("Building standard LSTM model (legacy)...")
        model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]), dropout_rate=dropout_rate)
    else:
        print("Building advanced multi-branch deep learning model...")
        model = build_advanced_model(
            input_shape=(X_train.shape[1], X_train.shape[2]), 
            dropout_rate=dropout_rate,
            learning_rate=learning_rate
        )
        
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    checkpoint_path = os.path.join(company_model_dir, 'checkpoint.keras')
    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=0.00001,
        verbose=1
    )
    
    callbacks = [early_stopping, model_checkpoint, reduce_lr]
    
    # Set class weights if using reward system to focus on dates where predictions were wrong
    class_weights = None
    if reward_system is not None:
        # Create a mapping of indices to weights based on dates that we've had poor predictions on
        metrics = reward_system.get_overall_accuracy()
        if metrics['count'] > 0:
            print("Applying class weights based on historical prediction accuracy")
            
            # Initialize all weights to 1.0
            sample_weights = np.ones(len(X_train))
            
            # Get prediction history with accuracy data
            predictions_df = reward_system.get_prediction_history()
            valid_predictions = predictions_df.dropna(subset=['actual_price'])
            
            # For each prediction with poor accuracy, apply higher weight to corresponding training samples
            if len(valid_predictions) > 0:
                try:
                    # Get dates from training data if available (approximate method)
                    # This is a simplified approach - in production, you'd need to map exact dates
                    for i in range(len(sample_weights)):
                        # Apply slightly higher weight to all samples to improve overall learning
                        if np.random.random() < 0.2:  # Randomly apply higher weights to 20% of samples
                            sample_weights[i] = 1.2
                    
                    # Get the worst predictions and give them higher weight
                    worst_predictions = valid_predictions[valid_predictions['threshold_met'] == False]
                    if len(worst_predictions) > 0:
                        # Apply higher weights to more recent samples (last 20% of training data)
                        start_idx = int(len(sample_weights) * 0.8)
                        for i in range(start_idx, len(sample_weights)):
                            sample_weights[i] = 1.5  # Higher weight for more recent samples
                            
                    print(f"Applied weighted training: {len(worst_predictions)} bad predictions found")
                except Exception as e:
                    print(f"Warning: Error applying prediction weights: {e}")
                    print("Continuing without weighted training")
            
            # Create class weights dictionary from sample weights
            class_weights = {i: sample_weights[i] for i in range(len(sample_weights))}
    
    # Train model
    try:
        print(f"Training model with {epochs} epochs and batch size {batch_size}...")
        print(f"Learning rate: {learning_rate}, Dropout rate: {dropout_rate}")
        
        if class_weights is not None:
            print("Using weighted samples based on historical prediction accuracy")
            
            # Convert class weights to sample weights array
            # Make sure the sample weights array has exactly the same length as X_train
            sample_weights = np.ones(len(X_train))
            for i in range(len(sample_weights)):
                sample_weights[i] = class_weights.get(i, 1.0)
                
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1,
                sample_weight=sample_weights
            )
        else:
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
        print("Model training completed successfully!")
    except Exception as e:
        print(f"Error during model training: {e}")
        return None, None
    
    # Save the model
    model_filename = 'lstm_model.keras' if use_legacy_model else 'advanced_model.keras'
    model_path = os.path.join(company_model_dir, model_filename)
    
    try:
        model.save(model_path)
        print(f"Model saved to {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")
        
    return model, history


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        X_test (np.array): Test features
        y_test (np.array): Test target
        
    Returns:
        tuple: (mse, rmse, mae, mape, r2) - Performance metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Ensure y_test and y_pred have the same shape
    if len(y_test.shape) != len(y_pred.shape):
        if len(y_test.shape) > len(y_pred.shape):
            y_pred = y_pred.reshape(y_test.shape)
        else:
            y_test = y_test.reshape(y_pred.shape)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    # Avoid division by zero
    mask = y_test != 0
    y_test_safe = y_test[mask]
    y_pred_safe = y_pred[mask]
    
    if len(y_test_safe) > 0:
        mape = np.mean(np.abs((y_test_safe - y_pred_safe) / y_test_safe)) * 100
    else:
        mape = 0.0
    
    # Calculate R²
    r2 = r2_score(y_test, y_pred)
    
    # Print metrics
    print(f"Model evaluation metrics:")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²: {r2:.4f}")
    
    return mse, rmse, mae, mape, r2


def save_model_metadata(symbol, features, scaler, model_dir='models'):
    """
    Save model metadata (scaler and feature names)
    
    Args:
        symbol (str): Stock symbol
        features (list): Feature names
        scaler: Fitted scaler
        model_dir (str): Directory to save metadata
    """
    # Create base model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Create company-specific model directory
    company_model_dir = os.path.join(model_dir, symbol)
    os.makedirs(company_model_dir, exist_ok=True)
    
    # Save scaler
    scaler_path = os.path.join(company_model_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    
    # Save feature names
    features_path = os.path.join(company_model_dir, "features.txt")
    with open(features_path, 'w') as f:
        for feature in features:
            f.write(f"{feature}\n")
    
    print(f"Saved model metadata for {symbol}")


def plot_training_history(history, symbol, model_dir='models'):
    """
    Plot training history
    
    Args:
        history: Model training history
        symbol (str): Stock symbol
        model_dir (str): Directory to save plot
    """
    # Create base model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Create company-specific model directory
    company_model_dir = os.path.join(model_dir, symbol)
    os.makedirs(company_model_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Model Loss - {symbol}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.grid(True)
    
    # Save plot
    plot_path = os.path.join(company_model_dir, "training_history.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Saved training history plot to {plot_path}")


def update_previous_predictions_with_actual_prices(symbol, df, reward_system, no_rules=False):
    """
    Update previous predictions with actual prices if available in the current data
    
    Args:
        symbol (str): Stock symbol
        df (pd.DataFrame): DataFrame with stock data including actual prices
        reward_system (PredictionRewardSystem): The reward system instance to update
        no_rules (bool): Whether to skip applying financial rules
        
    Returns:
        int: Number of predictions updated
    """
    if reward_system is None or df is None or len(df) == 0:
        return 0
        
    print("\nUpdating previous predictions with actual prices (if available)...")
    try:
        # Get prediction history
        predictions_df = reward_system.get_prediction_history()
        
        # If we have no predictions, nothing to update
        if predictions_df.empty:
            print("No previous predictions found to update")
            return 0
            
        # Get predictions that don't have actual prices
        pending_predictions = predictions_df[predictions_df['actual_price'].isna()]
        
        if pending_predictions.empty:
            print("No pending predictions found that need actual price updates")
            return 0
            
        print(f"Found {len(pending_predictions)} predictions that need actual price updates")
        
        # Make sure stock data has datetime format for 'date' column
        stock_data = df.copy()
        stock_data['date'] = pd.to_datetime(stock_data['date'])
        
        # For each prediction without an actual price
        update_count = 0
        for _, row in pending_predictions.iterrows():
            pred_date = pd.to_datetime(row['date'])
            
            # Find this date in our stock data
            matching_data = stock_data[stock_data['date'] == pred_date]
            
            if not matching_data.empty:
                # We have actual price data for this prediction
                actual_price = matching_data['close'].values[0]
                
                # Update the prediction with actual price
                if reward_system.update_actual_price(pred_date, actual_price):
                    update_count += 1
                    print(f"  Updated prediction for {pred_date.strftime('%Y-%m-%d')}: Predicted={row['predicted_price']:.2f}, Actual={actual_price:.2f}")
        
        if update_count > 0:
            print(f"Successfully updated {update_count} predictions with actual prices")
            
            # Now that we've updated predictions, let's print accuracy metrics
            metrics = reward_system.get_overall_accuracy()
            if metrics['count'] > 0:
                print(f"\nPrediction accuracy metrics:")
                print(f"  Total predictions with actual prices: {metrics['count']}")
                print(f"  Mean accuracy: {metrics['mean_accuracy']*100:.2f}%")
                print(f"  Predictions meeting threshold: {metrics['threshold_met_count']} ({metrics['threshold_met_pct']*100:.2f}%)")
        else:
            print("No predictions could be updated with actual prices")
            
        return update_count
    except Exception as e:
        print(f"Warning: Error updating predictions with actual prices: {e}")
        import traceback
        traceback.print_exc()
        return 0


def main(symbol=None, window_size=10, epochs=100, batch_size=32, prediction_days=5, 
         learning_rate=0.001, dropout_rate=0.3, use_legacy_model=False, 
         no_rules=False, df_with_indicators=None, reward_system=None,
         use_model_selection=True, models_to_try=None, priority_metric='mape',
         accuracy_threshold=0.75, sequential_selection=True, model_dir='models'):
    """
    Main function for model training
    
    Args:
        symbol (str): Stock symbol
        window_size (int): Window size for sequences
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        prediction_days (int): Number of days to predict ahead
        learning_rate (float): Learning rate for model training
        dropout_rate (float): Dropout rate for regularization
        use_legacy_model (bool): Whether to use the legacy LSTM model
        no_rules (bool): Whether to apply financial rules
        df_with_indicators (pd.DataFrame): Pre-calculated technical indicators
        reward_system (PredictionRewardSystem): Optional reward system instance
        use_model_selection (bool): Whether to use model selection
        models_to_try (list): List of models to try
        priority_metric (str): Metric to prioritize for model selection
        accuracy_threshold (float): Minimum acceptable accuracy (0.0-1.0)
        sequential_selection (bool): Whether to try models sequentially until finding acceptable accuracy
        model_dir (str): Directory to save models in, defaults to 'models'
    
    Returns:
        tuple: (model, feature_list, scaler)
    """
    # If no symbol is provided via argument, try to get it from command line
    if symbol is None:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Train stock prediction model')
        parser.add_argument('--symbol', type=str, required=True, help='Stock symbol')
        parser.add_argument('--window', type=int, default=10, help='Window size for sequences')
        parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
        parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
        parser.add_argument('--prediction_days', type=int, default=5, help='Number of days ahead to predict')
        parser.add_argument('--no_rules', action='store_true', help='Skip applying financial rules')
        parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for model')
        parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate for model')
        parser.add_argument('--use_legacy_model', action='store_true', help='Use legacy LSTM model')
        parser.add_argument('--force_cpu', action='store_true', help='Force CPU usage')
        parser.add_argument('--no_reward_system', action='store_true', help='Skip using reward system')
        parser.add_argument('--reward_threshold', type=float, default=0.05, help='Reward system threshold')
        parser.add_argument('--no_model_selection', action='store_true', help='Disable automatic model selection (use single model approach instead)')
        parser.add_argument('--models_to_try', type=str, nargs='+', help='List of models to try')
        parser.add_argument('--priority_metric', type=str, default='mape', 
                         choices=['mse', 'rmse', 'mae', 'mape', 'r2'], 
                         help='Metric to prioritize for model selection')
        
        args = parser.parse_args()
        
        # Set parameters from command line arguments
        symbol = args.symbol
        window_size = args.window
        epochs = args.epochs
        batch_size = args.batch_size
        prediction_days = args.prediction_days
        no_rules = args.no_rules
        learning_rate = args.learning_rate
        dropout_rate = args.dropout
        use_legacy_model = args.use_legacy_model
        use_model_selection = not args.no_model_selection  # Model selection is on by default unless disabled
        models_to_try = args.models_to_try
        priority_metric = args.priority_metric
        
        # Force CPU if requested
        if args.force_cpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
        # Initialize reward system if enabled
        if not args.no_reward_system:
            reward_system = PredictionRewardSystem(symbol=args.symbol, threshold=args.reward_threshold)
            print(f"Prediction reward system enabled with threshold {args.reward_threshold}")
    
    # Ensure we have a symbol
    if symbol is None:
        print("Error: No stock symbol provided")
        return
    
    print(f"Training prediction model for {symbol}")
    
    # Load data - use pre-calculated indicators if provided
    if df_with_indicators is None:
        print("Loading stock data...")
        df = load_stock_data(symbol, apply_rules=not no_rules)
        if df is None or len(df) == 0:
            print(f"Error: No data found for {symbol}")
            return
            
        print("Adding technical indicators...")
        # Use the load_or_calculate_technical_indicators function to get indicators
        from src.preprocessing import load_or_calculate_technical_indicators
        df_with_indicators = load_or_calculate_technical_indicators(df, symbol)
    else:
        print("Using pre-calculated technical indicators")
    
    if df_with_indicators is None or len(df_with_indicators) == 0:
        print(f"Error: Failed to add technical indicators for {symbol}")
        return
    
    # Before training, check if we need to update any previous predictions with actual prices
    if reward_system is not None:
        update_previous_predictions_with_actual_prices(symbol, df_with_indicators, reward_system, no_rules=no_rules)
    
    # Prepare data for training
    print("Preparing data for training...")
    X_train, X_test, y_train, y_test, scaler, feature_list = prepare_train_test_data(
        df_with_indicators, window_size=window_size, prediction_days=prediction_days
    )
    
    print(f"Data prepared successfully:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples: {len(X_test)}")
    print(f"  Features: {X_train.shape[2]} ({', '.join(feature_list[:5])}...)")
    
    # Add callbacks for early stopping and learning rate reduction
    callbacks = []
    callbacks.append(
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        )
    )
    callbacks.append(
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-5,
            verbose=1
        )
    )
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, symbol, "checkpoint.keras"),
            save_best_only=True,
            verbose=1
        )
    )
    
    # If using model selection approach
    if use_model_selection:
        # Initialize model selector
        model_selector = ModelSelector(symbol=symbol, model_dir=model_dir)
        
        # Check if we already have a model for this symbol
        best_model_info_file = os.path.join(model_dir, symbol, "best_model_info.json")
        has_existing_model = os.path.exists(best_model_info_file)
        
        # Check historical prediction accuracy if we have a reward system
        current_model_acceptable = False
        if has_existing_model and reward_system:
            print("\nChecking historical prediction accuracy for existing model...")
            metrics = reward_system.get_overall_accuracy()
            
            if metrics['count'] > 0 and metrics['mean_accuracy'] is not None:
                historical_accuracy = 1.0 - metrics['mean_accuracy']  # Convert error to accuracy
                print(f"Historical prediction accuracy: {historical_accuracy*100:.2f}%")
                
                if historical_accuracy >= accuracy_threshold:
                    print(f"Existing model meets accuracy threshold ({accuracy_threshold*100:.2f}%)")
                    current_model_acceptable = True
                else:
                    print(f"Existing model does not meet accuracy threshold: {historical_accuracy*100:.2f}% < {accuracy_threshold*100:.2f}%")
                    print("Will train new models to find a better one")
            else:
                print("No historical prediction data available to evaluate existing model")
        
        # If existing model is acceptable, we can skip training new models
        if has_existing_model and current_model_acceptable:
            print("\nKeeping existing model as it meets accuracy requirements")
            
            # Load existing model info
            with open(best_model_info_file, 'r') as f:
                import json
                best_model_info = json.load(f)
            
            best_model_type = best_model_info['best_model_type']
            is_dl_model = best_model_info['is_dl_model']
            model_path = best_model_info['model_path']
            
            print(f"Using model: {best_model_info['model_name']} ({best_model_type})")
            
            # Load the model
            if is_dl_model:
                model = tf.keras.models.load_model(model_path)
            else:
                model = joblib.load(model_path)
            
            # Return early with existing model and metadata
            return model, feature_list, scaler
            
        # Proceed with model selection
        print("\nUsing model selection approach...")
        print(f"Available models: {', '.join(MODEL_REGISTRY.keys())}")
        
        # If specific models are requested, validate them
        if models_to_try:
            valid_models = []
            for model_name in models_to_try:
                if model_name in MODEL_REGISTRY:
                    valid_models.append(model_name)
                else:
                    print(f"Warning: Unknown model '{model_name}'. Skipping.")
            
            models_to_try = valid_models
            print(f"Training the following models: {', '.join(models_to_try)}")
        else:
            # Default to using all models
            models_to_try = list(MODEL_REGISTRY.keys())
            print(f"Training models: {', '.join(models_to_try)}")
        
        # Determine whether to use sequential or parallel approach
        if sequential_selection:
            print(f"\nUsing sequential model selection with accuracy threshold: {accuracy_threshold*100:.2f}%")
            
            # Train and evaluate models one by one until finding one with acceptable accuracy
            best_model_type, best_model_data, results = model_selector.try_models_sequentially(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models_to_try=models_to_try,
                accuracy_threshold=accuracy_threshold,
                window_size=window_size,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                dropout_rate=dropout_rate,
                callbacks=callbacks,
                priority_metric=priority_metric,
                reward_system=reward_system
            )
        else:
            print("\nTraining all models in parallel (legacy approach)")
            
            # Train and evaluate all requested models at once
            results = model_selector.train_and_evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models_to_try=models_to_try,
                window_size=window_size,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                dropout_rate=dropout_rate,
                callbacks=callbacks
            )
            
            # Select best model based on priority metric
            best_model_type, best_model_data = model_selector.select_best_model(
                results=results,
                priority_metric=priority_metric
            )
        
        print(f"\nBest model for {symbol} based on {priority_metric}: {MODEL_REGISTRY[best_model_type]}")
        print(f"Best model metrics:")
        for metric, value in best_model_data['metrics'].items():
            print(f"  {metric.upper()}: {value:.4f}")
        
        # Save metadata using the best model's information
        save_model_metadata(symbol, feature_list, scaler, model_dir=model_dir)
        
        # Create visualization of model comparison
        print("\nGenerating model comparison visualization...")
        model_selector.visualize_model_comparison(results=results)
        
        print("\nModel training completed successfully!")
    else:
        # Traditional single-model approach
        print("Using traditional single-model approach...")
        
        # Train model
        model, history = train_model(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model_dir=model_dir,
            symbol=symbol,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            use_legacy_model=use_legacy_model,
            reward_system=reward_system
        )
        
        if model is None:
            print("Error: Model training failed")
            return
        
        # Save model metadata
        save_model_metadata(symbol, feature_list, scaler, model_dir=model_dir)
        
        # Evaluate model
        print("\nEvaluating model...")
        evaluate_model(model, X_test, y_test)
        
        # Plot training history
        plot_training_history(history, symbol, model_dir=model_dir)
    
    print("\nModel training completed successfully!")


if __name__ == "__main__":
    main() 