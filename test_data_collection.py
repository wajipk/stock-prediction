"""
Simple test script to verify data collection from the API
"""
from src.data_collection import fetch_stock_data, save_stock_data

# Test with the MARI symbol
symbol = 'MARI'
days = 365  # 1 year for testing

print(f"Testing data collection for {symbol}...")
df = fetch_stock_data(symbol, days)

if df is not None and len(df) > 0:
    print(f"Successfully fetched {len(df)} records.")
    print("\nFirst 5 records:")
    print(df.head())
    
    print("\nLast 5 records:")
    print(df.tail())
    
    # Save the data
    save_stock_data(df, symbol)
else:
    print("Failed to fetch data.") 