"""
Test script to verify payouts API integration and demonstrate dividend adjustments.
This script fetches dividend data from the API for a specified symbol, 
displays the dividend events, and visualizes how they affect stock prices.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.rules import fetch_and_load_payouts, default_rules

def test_payouts_api(symbol):
    """Test the payouts API integration for a given stock symbol."""
    print(f"Testing payouts API for {symbol}...")
    
    # Clear existing rules data
    default_rules.clear_events()
    
    # Fetch payout data from API
    fetch_and_load_payouts(symbol)
    
    # Print loaded dividend events
    print("\nLoaded Dividend Events:")
    if symbol in default_rules.dividend_events and default_rules.dividend_events[symbol]:
        for date, amount in default_rules.dividend_events[symbol]:
            print(f"  {date.strftime('%Y-%m-%d')}: {amount} PKR")
    else:
        print("  No dividend events found")
    
    # Demonstrate the effect of adjustments
    demonstrate_adjustments(symbol)

def demonstrate_adjustments(symbol):
    """Create a sample dataset to demonstrate the effect of dividend adjustments on stock prices."""
    # Create a date range for the last year
    end_date = pd.Timestamp.now().tz_localize(None)  # Make timezone-naive
    start_date = end_date - pd.Timedelta(days=365)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    
    # Create a sample stock price dataset
    np.random.seed(42)  # For reproducibility
    prices = np.cumsum(np.random.normal(0.1, 1, size=len(dates))) + 100  # Random walk with upward bias
    
    # Create dataframe
    df = pd.DataFrame({
        'Date': dates,
        'Close': prices
    })
    
    # Apply dividend adjustments
    adjusted_prices = prices.copy()
    adjustment_points = []
    
    if symbol in default_rules.dividend_events:
        for date, amount in default_rules.dividend_events[symbol]:
            # Ensure the date is timezone-naive for comparison
            naive_date = date.tz_localize(None) if date.tzinfo is not None else date
            
            if start_date <= naive_date <= end_date:
                # Find all prices before this dividend date and adjust them
                mask = df['Date'] < naive_date
                adjusted_prices[mask] -= amount
                
                # Record the adjustment point for plotting
                idx = df[df['Date'] >= naive_date].index[0] if any(df['Date'] >= naive_date) else None
                if idx is not None:
                    adjustment_points.append((idx, amount))
    
    df['Adjusted'] = adjusted_prices
    
    # Plotting
    plt.figure(figsize=(12, 8))
    plt.plot(df['Date'], df['Close'], label='Original Price', color='blue')
    plt.plot(df['Date'], df['Adjusted'], label='Adjusted Price', color='green')
    
    # Mark dividend dates on the plot
    for idx, amount in adjustment_points:
        plt.axvline(x=df['Date'][idx], color='red', linestyle='--', alpha=0.7)
        plt.text(df['Date'][idx], df['Close'].max(), f"Dividend: {amount} PKR", 
                 rotation=90, verticalalignment='top')
    
    plt.title(f"{symbol} Stock Price - Original vs Dividend-Adjusted")
    plt.xlabel("Date")
    plt.ylabel("Price (PKR)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    output_file = f"{symbol}_adjustment_demonstration.png"
    plt.savefig(output_file)
    print(f"\nAdjustment demonstration plot saved as {output_file}")
    plt.close()

if __name__ == "__main__":
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else "MARI"
    test_payouts_api(symbol)
    print("\nPayouts API test completed.") 