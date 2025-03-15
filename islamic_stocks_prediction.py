#!/usr/bin/env python
import requests
import json
import subprocess
import time
import argparse
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

def fetch_islamic_companies():
    """Fetch all Islamic companies from the API"""
    url = "https://stocks.wajipk.com/api/companies?type=islamic&avg_volume=50000"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Islamic companies: {e}")
        return []

def run_prediction(symbol):
    """Run prediction for a single company (predict only mode)
    
    Args:
        symbol (str): Stock symbol to predict
        
    Returns:
        bool: True if prediction succeeded, False otherwise
    """
    # Build command - using skip_training to only run prediction
    command = f"python main.py --symbol {symbol}"
    
    # Add timestamp for logging
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Running prediction-only for {symbol}...")
    print(f"[{timestamp}] Executing command: {command}")
    
    try:
        # Run the command with real-time output
        print(f"[{timestamp}] --- Starting subprocess for {symbol} ---")
        print(f"[{timestamp}] --- Output from main.py will appear below ---")
        
        # Run the process with stdout and stderr passed directly to console for real-time viewing
        result = subprocess.run(command, shell=True, text=True)
        
        print(f"[{timestamp}] --- End of subprocess output for {symbol} ---")
        
        # Check if the process succeeded
        if result.returncode == 0:
            print(f"[{timestamp}] Prediction for {symbol} completed successfully")
            return True
        else:
            print(f"[{timestamp}] Error running prediction for {symbol}: Process exited with code {result.returncode}")
            
            # Record the failure
            try:
                with open("islamic_failed_companies.txt", "a+") as f:
                    # Check if symbol is already in the file
                    f.seek(0)
                    failed_companies = [line.strip() for line in f.readlines()]
                    if symbol not in failed_companies:
                        f.write(f"{symbol}\n")
                        print(f"[{timestamp}] Added {symbol} to islamic_failed_companies.txt")
            except Exception as e:
                print(f"[{timestamp}] Warning: Could not update islamic_failed_companies.txt: {e}")
            
            return False
    except Exception as e:
        print(f"[{timestamp}] Error running prediction for {symbol}: {e}")
        
        # Record the failure
        try:
            with open("islamic_failed_companies.txt", "a+") as f:
                # Check if symbol is already in the file
                f.seek(0)
                failed_companies = [line.strip() for line in f.readlines()]
                if symbol not in failed_companies:
                    f.write(f"{symbol}\n")
                    print(f"[{timestamp}] Added {symbol} to islamic_failed_companies.txt")
        except Exception as e:
            print(f"[{timestamp}] Warning: Could not update islamic_failed_companies.txt: {e}")
        
        return False

def main():
    parser = argparse.ArgumentParser(description="Run predictions for all Islamic companies")
    parser.add_argument("--parallel", type=int, default=1, 
                       help="Number of parallel predictions to run (default: 1)")
    parser.add_argument("--delay", type=float, default=0, 
                       help="Delay between predictions in seconds (default: 0)")
    parser.add_argument("--output", type=str, default="islamic_predictions_results.json",
                       help="Output file for saving results (default: islamic_predictions_results.json)")
    args = parser.parse_args()
    
    # Get all Islamic companies from API
    companies = fetch_islamic_companies()
    if not companies:
        print("No Islamic companies found or error fetching data")
        return
    
    print(f"Found {len(companies)} Islamic companies")
    
    # Extract symbols
    symbols = [company["symbol"] for company in companies if company.get("symbol")]
    
    # Create a timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting Islamic stocks prediction run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Processing {len(symbols)} symbols")
    
    # Define worker function for both parallel and sequential execution
    def worker(symbol):
        return run_prediction(symbol)
    
    # Track results
    successful = []
    failed = []
    
    if args.parallel > 1:
        # Run predictions in parallel
        print(f"Running predictions with {args.parallel} parallel workers")
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            # Use map with a list to collect results
            future_to_symbol = {executor.submit(worker, symbol): symbol for symbol in symbols}
            
            # Process results as they complete
            from concurrent.futures import as_completed
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    success = future.result()
                    if success:
                        successful.append(symbol)
                    else:
                        failed.append(symbol)
                except Exception as e:
                    print(f"Error processing {symbol}: {e}")
                    failed.append(symbol)
                
                # Show progress
                total_processed = len(successful) + len(failed)
                print(f"Progress: {total_processed}/{len(symbols)} - Success: {len(successful)}, Failed: {len(failed)}")
    else:
        # Run predictions sequentially
        for i, symbol in enumerate(symbols):
            success = worker(symbol)
            if success:
                successful.append(symbol)
            else:
                failed.append(symbol)
            
            # Show progress
            print(f"Progress: {i+1}/{len(symbols)} - Success: {len(successful)}, Failed: {len(failed)}")
            
            if args.delay > 0 and i < len(symbols) - 1:  # Don't delay after the last one
                print(f"Waiting {args.delay} seconds before next prediction...")
                time.sleep(args.delay)
    
    # Final report
    print("\n" + "=" * 60)
    print("ISLAMIC STOCKS PREDICTION RUN RESULTS")
    print("=" * 60)
    print(f"Total symbols processed: {len(symbols)}")
    print(f"Successfully completed: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print("\nFailed symbols:")
        for symbol in failed:
            print(f" - {symbol}")
    
    # Save results to output file
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_processed": len(symbols),
        "successful": successful,
        "failed": failed
    }
    
    try:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to {args.output}")
    except Exception as e:
        print(f"\nError saving results: {e}")
    
    print("=" * 60)
    print(f"Islamic stocks prediction run completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 