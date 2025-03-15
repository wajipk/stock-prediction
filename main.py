import os
import argparse
import platform
import sys
from datetime import datetime
from src.data_collection import fetch_stock_data, save_stock_data, load_corporate_actions
from src.preprocessing import load_or_calculate_technical_indicators
from src.train_model import main as train_main
from src.predict import main as predict_main
from src.market_analysis import get_market_trend_analysis
from src.prediction_reward_system import PredictionRewardSystem


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
        df = fetch_stock_data(args.symbol, days=args.days)
        if df is not None and len(df) > 0:
            save_stock_data(df, args.symbol)
        else:
            print("Error: Failed to fetch data. Exiting.")
            sys.exit(1)
    except Exception as e:
        print(f"Error during data fetching: {e}")
        sys.exit(1)
    
    # Step 2: Calculate or load technical indicators
    print("\n2. Processing technical indicators...")
    try:
        from src.preprocessing import load_stock_data
        stock_data = load_stock_data(args.symbol, apply_rules=not args.no_rules)
        if stock_data is None or len(stock_data) == 0:
            print("Error: Failed to load stock data. Exiting.")
            sys.exit(1)
            
        # Calculate or load technical indicators
        df_with_indicators = load_or_calculate_technical_indicators(stock_data, args.symbol)
        if df_with_indicators is None or len(df_with_indicators) == 0:
            print("Error: Failed to calculate technical indicators. Exiting.")
            sys.exit(1)
    except Exception as e:
        print(f"Error during technical indicator calculation: {e}")
        sys.exit(1)
        
    # Step 3: Train model (if not skipped)
    if not args.skip_training:
        print("\n3. Training prediction model...")
        
        # Initialize reward system if enabled
        reward_system = None
        if not args.no_reward_system:
            reward_system = PredictionRewardSystem(symbol=args.symbol, threshold=args.reward_threshold)
            print(f"Prediction reward system enabled with threshold {args.reward_threshold}")
            print(f"Self-learning feature active: Previous predictions will be compared with actual prices to improve future forecasts")
        
        # Train model using the data with pre-calculated indicators
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
                df_with_indicators=df_with_indicators,  # Pass the pre-calculated indicators
                reward_system=reward_system,
                use_model_selection=not args.no_model_selection,  # Use model selection by default unless disabled
                models_to_try=args.models_to_try,
                priority_metric=args.priority_metric,
                accuracy_threshold=args.accuracy_threshold,
                sequential_selection=not args.parallel_selection,  # Use sequential selection by default unless parallel is specified
                model_dir='models'  # Add default model_dir parameter
            )
        except Exception as e:
            print(f"Error during model training: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("\n3. Skipping model training as requested")
    
    # Step 4: Make predictions
    print("\n4. Making price predictions...")
    
    # Initialize reward system if enabled
    reward_system = None
    if not args.no_reward_system:
        reward_system = PredictionRewardSystem(symbol=args.symbol, threshold=args.reward_threshold)
        print(f"Prediction reward system enabled with threshold {args.reward_threshold}")
        print(f"Predictions will be stored for future accuracy evaluation and model improvement")
    
    # Get market trend information
    market_trend_info = None
    if not args.no_market_trends:
        try:
            print("Analyzing market trends...")
            market_trend_info = get_market_trend_analysis(args.symbol)
        except Exception as e:
            print(f"Warning: Error getting market trend information: {e}")
            print("Continuing without market trend analysis")
    
    # Make predictions
    try:
        predict_main(
            symbol=args.symbol,
            window_size=args.window,
            days_ahead=args.prediction_days,
            threshold=args.threshold,
            smoothing_factor=args.smoothing,
            apply_market_trends=not args.no_market_trends,
            market_adjustment_factor=args.market_adjustment,
            market_trend_info=market_trend_info,
            reward_system=reward_system,
            df_with_indicators=df_with_indicators,  # Pass the pre-calculated indicators
            reality_check=not args.no_reality_check
        )
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main() 