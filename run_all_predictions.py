#!/usr/bin/env python
import requests
import json
import subprocess
import time
import argparse
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

def fetch_companies():
    """Fetch all Islamic companies from the API"""
    url = "https://stocks.wajipk.com/api/companies?type=islamic"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching companies: {e}")
        return []

def get_failed_companies():
    """Get list of companies that failed in previous runs"""
    if not os.path.exists("failed_companies.txt"):
        return []
    
    with open("failed_companies.txt", "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def run_prediction(symbol, retry_failed=False, force_training=False, additional_args=None):
    """Run prediction for a single company with enhanced error handling
    
    Args:
        symbol (str): Stock symbol to predict
        retry_failed (bool): Whether this is a retry for a previously failed symbol
        force_training (bool): Whether to force training even if models exist
        additional_args (list): Additional command line arguments to pass
        
    Returns:
        bool: True if prediction succeeded, False otherwise
    """
    # Build command
    command = f"python main.py --symbol {symbol}"
    
    # Add training flag if needed
    if force_training:
        # Don't skip training
        pass
    else:
        command += " --skip_training"
    
    # Add any additional arguments
    if additional_args:
        command += " " + additional_args
    
    # Add timestamp for logging
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    action = "Retrying" if retry_failed else "Running"
    print(f"[{timestamp}] {action} prediction for {symbol}...")
    
    try:
        # Run the command and capture output
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        # Check if the process succeeded
        if result.returncode == 0:
            print(f"[{timestamp}] Prediction for {symbol} completed successfully")
            
            # If this was a retry and it succeeded, remove from failed_companies.txt
            if retry_failed:
                try:
                    failed_companies = get_failed_companies()
                    if symbol in failed_companies:
                        failed_companies.remove(symbol)
                        with open("failed_companies.txt", "w") as f:
                            f.write("\n".join(failed_companies) + "\n" if failed_companies else "")
                        print(f"[{timestamp}] Removed {symbol} from failed_companies.txt")
                except Exception as e:
                    print(f"[{timestamp}] Warning: Could not update failed_companies.txt: {e}")
            
            return True
        else:
            print(f"[{timestamp}] Error running prediction for {symbol}: Process exited with code {result.returncode}")
            print(f"[{timestamp}] STDOUT: {result.stdout}")
            print(f"[{timestamp}] STDERR: {result.stderr}")
            
            # Record the failure
            try:
                with open("failed_companies.txt", "a+") as f:
                    # Check if symbol is already in the file
                    f.seek(0)
                    failed_companies = [line.strip() for line in f.readlines()]
                    if symbol not in failed_companies:
                        f.write(f"{symbol}\n")
                        print(f"[{timestamp}] Added {symbol} to failed_companies.txt")
            except Exception as e:
                print(f"[{timestamp}] Warning: Could not update failed_companies.txt: {e}")
            
            return False
    except Exception as e:
        print(f"[{timestamp}] Error running prediction for {symbol}: {e}")
        
        # Record the failure
        try:
            with open("failed_companies.txt", "a+") as f:
                # Check if symbol is already in the file
                f.seek(0)
                failed_companies = [line.strip() for line in f.readlines()]
                if symbol not in failed_companies:
                    f.write(f"{symbol}\n")
                    print(f"[{timestamp}] Added {symbol} to failed_companies.txt")
        except Exception as e:
            print(f"[{timestamp}] Warning: Could not update failed_companies.txt: {e}")
        
        return False

def main():
    parser = argparse.ArgumentParser(description="Run predictions for all Islamic companies")
    parser.add_argument("--parallel", type=int, default=1, 
                       help="Number of parallel predictions to run (default: 1)")
    parser.add_argument("--delay", type=float, default=0, 
                       help="Delay between predictions in seconds (default: 0)")
    parser.add_argument("--retry_failed", action="store_true",
                       help="Retry previously failed companies with force_training enabled")
    parser.add_argument("--force_training", action="store_true",
                       help="Force training for all symbols")
    parser.add_argument("--additional_args", type=str, default="",
                       help="Additional arguments to pass to main.py (in quotes)")
    args = parser.parse_args()
    
    # Process symbols based on mode
    if args.retry_failed:
        # Get list of failed companies from file
        symbols = get_failed_companies()
        if not symbols:
            print("No failed companies found to retry")
            return
        print(f"Found {len(symbols)} failed companies to retry")
    else:
        # Get all companies from API
        companies = fetch_companies()
        if not companies:
            print("No companies found or error fetching data")
            return
        
        print(f"Found {len(companies)} Islamic companies")
        # Extract symbols
        symbols = [company["symbol"] for company in companies if company.get("symbol")]
    
    # Create a timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting prediction run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Processing {len(symbols)} symbols")
    
    # Define worker function for both parallel and sequential execution
    def worker(symbol):
        return run_prediction(
            symbol, 
            retry_failed=args.retry_failed,
            force_training=args.force_training or args.retry_failed,
            additional_args=args.additional_args
        )
    
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
            
            if args.delay > 0:
                print(f"Waiting {args.delay} seconds before next prediction...")
                time.sleep(args.delay)
    
    # Final report
    print("\n" + "=" * 60)
    print("PREDICTION RUN RESULTS")
    print("=" * 60)
    print(f"Total symbols processed: {len(symbols)}")
    print(f"Successfully completed: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print("\nFailed symbols:")
        for symbol in failed:
            print(f" - {symbol}")
    
    print("=" * 60)
    print(f"Prediction run completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 