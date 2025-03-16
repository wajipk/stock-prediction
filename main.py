import os
import argparse
import platform
import sys
import traceback
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from src.data_collection import fetch_stock_data, save_stock_data, load_corporate_actions
from src.preprocessing import load_or_calculate_technical_indicators
from src.train_model import main as train_main
from src.predict import main as predict_main
from src.market_analysis import get_market_trend_analysis
from src.prediction_reward_system import PredictionRewardSystem
from src.reinforced_model import ReinforcedStockModel
from src.reinforced_prediction_example import train_reinforced_model


def main():
    """
    Main function to run the entire stock prediction pipeline
    """
    # Print environment information to help diagnose issues
    print("=" * 50)
    print(f"Advanced Stock Prediction Pipeline")
    print(f"Running on: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    # Ensure required directories exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Stock Price Prediction Pipeline')
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol')
    parser.add_argument('--days', type=int, default=365, help='Number of days of historical data to fetch')
    parser.add_argument('--window', type=int, default=10, help='Window size for sequences')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--prediction_days', type=int, default=5, help='Number of days ahead to predict')
    parser.add_argument('--threshold', type=float, default=2.0, help='Threshold percentage for significant movements')
    parser.add_argument('--skip_training', action='store_true', help='Skip training and use existing model')
    parser.add_argument('--no_rules', action='store_true', help='Skip applying financial rules (for testing purposes)')
    parser.add_argument('--smoothing', type=float, default=0.5, help='Smoothing factor for predictions (0.0-1.0)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for model training')
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate for model regularization')
    parser.add_argument('--use_legacy_model', action='store_true', help='Use the simple LSTM model instead of the advanced model')
    parser.add_argument('--force_cpu', action='store_true', help='Force using CPU even if GPU is available')
    parser.add_argument('--market_adjustment', type=float, default=0.015, help='Market trend adjustment factor (0.0-0.1)')
    parser.add_argument('--no_market_trends', action='store_true', help='Skip applying market trend adjustments')
    parser.add_argument('--reward_threshold', type=float, default=0.05, help='Threshold for the reward system')
    parser.add_argument('--no_reward_system', action='store_true', help='Skip using the reward system')
    parser.add_argument('--no_reality_check', action='store_true', help='Disable reality checks for predictions (not recommended)')
    parser.add_argument('--no_model_selection', action='store_true', help='Disable automatic model selection (use single model approach instead)')
    parser.add_argument('--models_to_try', type=str, nargs='+', help='List of models to try (e.g., "lstm bilstm xgboost")')
    parser.add_argument('--priority_metric', type=str, default='mape', 
                       choices=['mse', 'rmse', 'mae', 'mape', 'r2'], 
                       help='Metric to prioritize for model selection')
    parser.add_argument('--accuracy_threshold', type=float, default=0.75,
                       help='Minimum acceptable model accuracy (0.0-1.0, default: 0.75)')
    parser.add_argument('--sequential_selection', action='store_true', default=True,
                       help='Try models sequentially until finding one with acceptable accuracy')
    parser.add_argument('--parallel_selection', action='store_true',
                       help='Train all models in parallel (legacy approach)')
    parser.add_argument('--use_reinforced_model', default=True, action='store_true',
                        help='Use the reinforced model that learns from previous prediction errors')
    parser.add_argument('--base_model_type', type=str, default='lstm',
                        choices=['lstm', 'bilstm', 'gru', 'cnn_lstm', 'advanced'],
                        help='Base model type for reinforced learning')
    parser.add_argument('--error_threshold', type=float, default=0.05,
                        help='Error threshold for reinforced model adaptation')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print(f"Advanced Stock Prediction Pipeline for {args.symbol}")
    print("=" * 50)
    
    # Set environment variables before importing TensorFlow
    if args.force_cpu:
        print("Forcing CPU usage as requested")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # Import modules here to apply environment settings first
    try:
        from src.data_collection import fetch_stock_data, save_stock_data, load_corporate_actions
        from src.preprocessing import load_or_calculate_technical_indicators
        from src.train_model import main as train_main
        from src.predict import main as predict_main
        from src.reinforced_prediction_example import train_reinforced_model
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    
    # Load corporate actions first
    if not args.no_rules:
        print("\nLoading corporate actions (dividends, bonus shares, etc.)...")
        try:
            load_corporate_actions(args.symbol)
        except Exception as e:
            print(f"Warning: Error loading corporate actions: {e}")
            print("Continuing without corporate actions...")
    else:
        print("\nSkipping financial rules as requested")
    
    # Step 1: Fetch data
    print("\n1. Fetching historical stock data...")
    try:
        # Check if data exists in data directory, otherwise fetch it
        data_file = os.path.join('data', f'{args.symbol}.csv')
        if not os.path.exists(data_file):
            print(f"Data file not found, fetching from source...")
            df = fetch_stock_data(args.symbol, days=args.days)
            save_stock_data(df, args.symbol)
        else:
            print(f"Using existing data file: {data_file}")
            df = pd.read_csv(data_file, index_col=0, parse_dates=True)
            
            # Check if we need to get more recent data
            last_date = df.index[-1]
            today = datetime.now().date()
            days_difference = (today - last_date.date()).days
            
            if days_difference > 3:  # Get new data if more than 3 days old
                print(f"Data is {days_difference} days old. Fetching new data...")
                df = fetch_stock_data(args.symbol, days=args.days)
                save_stock_data(df, args.symbol)
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Step 2: Calculate technical indicators
    print("\n2. Calculating technical indicators...")
    try:
        df_with_indicators = load_or_calculate_technical_indicators(df, args.symbol)
    except Exception as e:
        print(f"Error calculating technical indicators: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Initialize reward system if not disabled
    reward_system = None
    if not args.no_reward_system:
        reward_system = PredictionRewardSystem(
            args.symbol, threshold=args.reward_threshold
        )
        print(f"Initialized prediction reward system with threshold {args.reward_threshold}")
        print(f"Predictions will be stored for future accuracy evaluation and model improvement")
    
    # Choose the approach based on user selection
    if args.use_reinforced_model:
        print("\n3. Using reinforced learning model...")
        
        # Train the reinforced model
        model, features_list = train_reinforced_model(
            symbol=args.symbol,
            window_size=args.window,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            dropout_rate=args.dropout,
            base_model_type=args.base_model_type,
            error_threshold=args.error_threshold
        )
        
        # Check if model training was successful
        if model is None:
            print("Failed to train the reinforced model. Exiting.")
            sys.exit(1)
        
        # Make predictions for the next days
        if df_with_indicators is not None:
            print(f"\nMaking predictions for the next {args.prediction_days} days using reinforced model...")
            
            # Ensure we use only the features that were used during training
            print(f"Using {len(features_list)} features from the trained model")
            
            # Make a copy to avoid modifying the original
            prediction_df = df_with_indicators.copy()
            
            # Select only the columns used during training
            if features_list is not None and len(features_list) > 0:
                # Ensure all required features exist in the dataframe
                missing_features = [f for f in features_list if f not in prediction_df.columns]
                if missing_features:
                    print(f"Warning: Missing features from training set: {missing_features}")
                    print("These features will be filled with zeros")
                    for feature in missing_features:
                        prediction_df[feature] = 0.0
                        
                # Use only the features from training
                numeric_df = prediction_df[features_list]
            else:
                print("Warning: No feature list available from training. Using all numeric columns.")
                # Use all numeric columns as fallback
                numeric_df = prediction_df.select_dtypes(include=[np.number])
            
            # Get the most recent window_size rows as input
            try:
                # Convert to numpy array and ensure float32 type
                latest_data = numeric_df.iloc[-args.window:].values.astype(np.float32)
                
                # Debug information
                print(f"Input shape for prediction: {latest_data.shape}")
                print(f"Expected input shape based on training: ({args.window}, {len(features_list)})")
                
                # Check if dimensions match
                if latest_data.shape[1] != len(features_list) and features_list is not None:
                    print(f"Warning: Feature count mismatch. Model expects {len(features_list)} features but input has {latest_data.shape[1]}.")
                    if latest_data.shape[1] > len(features_list):
                        print("Truncating extra features...")
                        latest_data = latest_data[:, :len(features_list)]
                    else:
                        print("Padding missing features with zeros...")
                        padding = np.zeros((latest_data.shape[0], len(features_list) - latest_data.shape[1]), dtype=np.float32)
                        latest_data = np.hstack((latest_data, padding))
                    print(f"Adjusted input shape: {latest_data.shape}")
                
                # Determine which column is the price column (usually the first, but let's make sure)
                price_column_idx = 0
                if 'close' in features_list:
                    price_column_idx = list(features_list).index('close')
                elif 'price' in features_list:
                    price_column_idx = list(features_list).index('price')
                
                print(f"Using column index {price_column_idx} as the price column for predictions")
                
                # Create a list to store prediction dates and values for display
                prediction_results = []
                
                # Get the most recent date from the dataframe for sequential predictions
                if 'date' in df_with_indicators.columns:
                    last_date = pd.to_datetime(df_with_indicators['date'].iloc[-1])
                else:
                    last_date = datetime.now().date()
                
                for i in range(args.prediction_days):
                    try:
                        # Reshape input for prediction
                        input_data = latest_data[-args.window:].reshape(1, args.window, latest_data.shape[1])
                        
                        # Make prediction with the reinforced model
                        prediction = model.predict(input_data, apply_correction=True)
                        
                        # Calculate prediction date
                        if isinstance(last_date, pd.Timestamp):
                            prediction_date = (last_date + pd.Timedelta(days=i+1)).strftime('%Y-%m-%d')
                        else:
                            prediction_date = (datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d')
                        
                        prediction_results.append((prediction_date, prediction))
                        
                        # Create new row based on the last row
                        new_row = latest_data[-1].copy()
                        
                        # Set the price column to the predicted value
                        new_row[price_column_idx] = prediction
                        
                        # Add the new row to latest_data
                        latest_data = np.vstack([latest_data[1:], new_row])  # Remove oldest row, add newest
                    except Exception as e:
                        print(f"Error in prediction loop at iteration {i}: {e}")
                        traceback.print_exc()
                
                # Display all predictions
                print("\nStock Price Predictions:")
                print("-" * 30)
                for date, price in prediction_results:
                    print(f"{date}: ${price:.2f}")
                print("-" * 30)
            except Exception as e:
                print(f"Error making predictions: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Use the standard approach
        
        # Step 3: Train model (if not skipped)
        if not args.skip_training:
            print("\n3. Training prediction model...")
            
            try:
                train_main(
                    symbol=args.symbol,
                    window_size=args.window,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    prediction_days=args.prediction_days,
                    learning_rate=args.learning_rate,
                    dropout_rate=args.dropout,
                    use_legacy_model=args.use_legacy_model,
                    no_rules=args.no_rules,
                    df_with_indicators=df_with_indicators,
                    reward_system=reward_system,
                    use_model_selection=not args.no_model_selection,
                    models_to_try=args.models_to_try,
                    priority_metric=args.priority_metric,
                    accuracy_threshold=args.accuracy_threshold,
                    sequential_selection=args.sequential_selection and not args.parallel_selection
                )
            except Exception as e:
                print(f"Error during model training: {e}")
                traceback.print_exc()
                sys.exit(1)
        else:
            print("\n3. Skipping model training as requested")
        
        # Step 4: Make predictions
        print("\n4. Making price predictions...")
        
        # Get market trend information
        market_trend_info = None
        if not args.no_market_trends:
            try:
                print("Analyzing market trends...")
                market_trend_info = get_market_trend_analysis()
                print(f"Market trend captured: {market_trend_info['trend']} ({market_trend_info['confidence']:.1f}% confidence)")
            except Exception as e:
                print(f"Warning: Unable to get market trend analysis: {e}")
                print("Proceeding without market trend adjustment")
        
        # Make predictions
        try:
            predictions = predict_main(
                symbol=args.symbol,
                days=args.prediction_days,
                smoothing_factor=args.smoothing,
                market_trend_info=None if args.no_market_trends else market_trend_info,
                market_adjustment_factor=args.market_adjustment,
                reality_check=not args.no_reality_check,
                reward_system=reward_system
            )
            
            # Print predictions
            print("\nStock Price Predictions:")
            print("-" * 30)
            for date, price in predictions.items():
                print(f"{date}: ${price:.2f}")
            print("-" * 30)
            
        except Exception as e:
            print(f"Error making predictions: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("\nPrediction pipeline completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main() 