import requests
import pandas as pd
import json
import os
import argparse
from datetime import datetime, timedelta
from src.rules import default_rules, adjust_stock_data, fetch_and_load_payouts


def fetch_stock_data(symbol, days=365):
    """
    Fetch historical stock data for the given symbol from stocks.wajipk.com API
    
    Args:
        symbol (str): Stock symbol
        days (int): Number of days of historical data to fetch
        
    Returns:
        pd.DataFrame: DataFrame containing historical stock data
    """
    print(f"Fetching data for {symbol} for the last {days} days...")
    
    # API endpoint - using the trades endpoint as specified
    url = f"https://stocks.wajipk.com/api/trades?symbol={symbol}"
    
    try:
        # Make API request
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Parse JSON response
        data = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Print available columns to debug
        print(f"API response columns: {df.columns.tolist()}")
        
        # Check if we got expected data format
        if len(df) == 0:
            print(f"Error: No data returned from API for {symbol}")
            return None
        
        # Check and rename columns to match expected format
        # If columns don't exist, create them with placeholder values
        
        # Expected columns: date, open, high, low, close, volume
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
            print("Error: No 'close' price column in API response")
            return None
            
        if 'open' not in df.columns:
            print("Warning: No 'open' price in data, using close price")
            df['open'] = df['close']
            
        if 'high' not in df.columns:
            if 'open' in df.columns:
                df['high'] = df[['open', 'close']].max(axis=1)
            else:
                df['high'] = df['close']
                
        if 'low' not in df.columns:
            if 'open' in df.columns:
                df['low'] = df[['open', 'close']].min(axis=1)
            else:
                df['low'] = df['close']
                
        if 'volume' not in df.columns:
            print("Warning: No volume data available")
            df['volume'] = 0
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date in ascending order (oldest first, newest last)
        df = df.sort_values('date', ascending=True).reset_index(drop=True)
        
        # Limit to the requested number of days
        if len(df) > days:
            df = df.tail(days).reset_index(drop=True)
        
        print(f"Successfully fetched {len(df)} records for {symbol}")
        
        # Print the last date in the DataFrame
        if not df.empty:
            last_date = df['date'].iloc[-1]
            print(f"Last date in fetched data: {last_date.strftime('%Y-%m-%d')}")
        
        # Apply financial rules to adjust stock prices
        print(f"Applying financial rules to adjust stock prices...")
        df = adjust_stock_data(df, symbol)
        
        return df
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error processing data: {e}")
        return None


def save_stock_data(df, symbol):
    """
    Save stock data to CSV file
    
    Args:
        df (pd.DataFrame): DataFrame containing stock data
        symbol (str): Stock symbol
    """
    if df is None or len(df) == 0:
        print("No data to save")
        return
    
    # Create data directory and company subdirectory if they don't exist
    company_dir = os.path.join('data', symbol)
    os.makedirs(company_dir, exist_ok=True)
    
    # Save to CSV
    file_path = os.path.join(company_dir, "historical_data.csv")
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")


def load_corporate_actions(symbol=None):
    """
    Load corporate actions from the API and/or the CSV file
    
    Args:
        symbol (str, optional): Stock symbol. If provided, fetches data from API for this symbol.
    """
    # Try to load from API if symbol is provided
    if symbol:
        print(f"Fetching corporate actions for {symbol} from API...")
        success = fetch_and_load_payouts(symbol)
        if success:
            print(f"Successfully loaded corporate actions for {symbol} from API")
        else:
            print(f"Failed to load corporate actions from API, falling back to local file")
    
    # Check for company-specific corporate actions file
    if symbol:
        company_dir = os.path.join('data', symbol)
        company_actions_file = os.path.join(company_dir, "corporate_actions.csv")
        if os.path.exists(company_actions_file):
            print(f"Loading corporate actions for {symbol} from company-specific file...")
            default_rules.load_events_from_file(company_actions_file)
            print(f"Corporate actions for {symbol} loaded successfully.")
            return
    
    # Also load from CSV file as a fallback or supplement
    actions_file = "data/corporate_actions.csv"
    if os.path.exists(actions_file):
        print("Loading additional corporate actions from general file...")
        default_rules.load_events_from_file(actions_file)
        print("Corporate actions from file loaded successfully.")
    else:
        print("No corporate actions file found.")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Fetch historical stock data')
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol')
    parser.add_argument('--days', type=int, default=1095, help='Number of days of historical data to fetch')
    
    args = parser.parse_args()
    
    # Load corporate actions from API and file
    load_corporate_actions(args.symbol)
    
    # Fetch and save data
    df = fetch_stock_data(args.symbol, args.days)
    save_stock_data(df, args.symbol)


if __name__ == "__main__":
    main() 