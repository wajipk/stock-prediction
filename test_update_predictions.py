import os
import sys
import argparse
import pandas as pd
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.append('.')

from src.prediction_reward_system import PredictionRewardSystem
from src.data_collection import fetch_stock_data, save_stock_data
from src.train_model import update_previous_predictions_with_actual_prices
from src.preprocessing import load_stock_data, add_technical_indicators


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test updating predictions with actual prices')
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol')
    parser.add_argument('--days', type=int, default=30, help='Number of days of historical data to fetch')
    parser.add_argument('--threshold', type=float, default=0.05, help='Threshold for the reward system')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print(f"Testing prediction update system for {args.symbol}")
    print("=" * 50)
    
    # Initialize reward system
    reward_system = PredictionRewardSystem(symbol=args.symbol, threshold=args.threshold)
    print(f"Initialized reward system with threshold {args.threshold}")
    
    # Step 1: Check for existing predictions
    predictions_df = reward_system.get_prediction_history()
    if predictions_df.empty:
        print("No existing predictions found. Creating sample predictions...")
        # Create sample predictions for testing
        today = datetime.now()
        
        # Generate predictions for past dates that we can later update with actual prices
        for i in range(1, 11):  # Last 10 days
            date = today - timedelta(days=i)
            # Skip weekends
            if date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
                continue
                
            date_str = date.strftime('%Y-%m-%d')
            # Fake prediction value - just a placeholder
            pred_value = 100.0 + (i * 2.5)
            
            # Save the prediction
            reward_system.save_prediction(date_str, pred_value)
            print(f"Created sample prediction for {date_str}: ${pred_value:.2f}")
    else:
        print(f"Found {len(predictions_df)} existing predictions")
        # Show existing predictions
        print("\nExisting prediction history:")
        pd.set_option('display.max_columns', None)
        print(predictions_df)
    
    # Step 2: Fetch the latest stock data
    print("\nFetching recent stock data...")
    df = fetch_stock_data(args.symbol, days=args.days)
    
    if df is None or len(df) == 0:
        print("Error: Failed to fetch data. Exiting.")
        sys.exit(1)
        
    # Save stock data
    save_stock_data(df, args.symbol)
    
    # Load stock data with processing
    stock_data = load_stock_data(args.symbol, apply_rules=True)
    
    # Add technical indicators
    df_with_indicators = add_technical_indicators(stock_data, rolling_windows=[5, 10, 20])
    
    # Step 3: Update predictions with actual prices
    print("\nUpdating predictions with actual prices...")
    update_count = update_previous_predictions_with_actual_prices(
        args.symbol, 
        df_with_indicators, 
        reward_system, 
        no_rules=False
    )
    
    # Step 4: Show final results
    print("\nFinal prediction history after updates:")
    updated_predictions_df = reward_system.get_prediction_history()
    pd.set_option('display.max_columns', None)
    print(updated_predictions_df)
    
    if update_count > 0:
        # Calculate and show accuracy
        metrics = reward_system.get_overall_accuracy()
        if metrics['count'] > 0:
            print(f"\nPrediction accuracy metrics:")
            print(f"  Total predictions with actual prices: {metrics['count']}")
            print(f"  Mean accuracy: {metrics['mean_accuracy']*100:.2f}%")
            print(f"  Predictions meeting threshold: {metrics['threshold_met_count']} ({metrics['threshold_met_pct']*100:.2f}%)")
    
    print("\nTest completed!")


if __name__ == "__main__":
    main() 