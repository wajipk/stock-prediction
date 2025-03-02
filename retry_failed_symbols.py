import subprocess
import sys
import argparse
import os
import time
from datetime import datetime

def get_failed_companies():
    """Get list of companies that failed in previous runs"""
    if not os.path.exists("failed_companies.txt"):
        return []
    
    with open("failed_companies.txt", "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def run_stock_prediction(symbol, prediction_days=5, threshold=2.0, force_training=True, additional_args=None):
    """
    Run the stock prediction model for a given company symbol
    
    Args:
        symbol (str): Stock symbol to predict
        prediction_days (int): Number of days ahead to predict
        threshold (float): Threshold percentage for significant movements
        force_training (bool): Whether to force training even if models exist
        additional_args (list): Additional command line arguments to pass
        
    Returns:
        bool: True if prediction succeeded, False otherwise
    """
    cmd = [
        sys.executable,  # Python executable
        "main.py",
        "--symbol", symbol,
        "--prediction_days", str(prediction_days),
        "--threshold", str(threshold)
    ]
    
    # If we're skipping training, add the flag
    if not force_training:
        cmd.append("--skip_training")
    
    # Add any additional arguments
    if additional_args:
        cmd.extend(additional_args)
    
    # Add timestamp for logging
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'-' * 60}")
    print(f"[{timestamp}] RUNNING MODEL FOR: {symbol} WITH {'FORCED' if force_training else 'OPTIONAL'} TRAINING")
    print(f"{'-' * 60}")
    
    try:
        # Run the command and stream output in real-time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Print stdout in real-time
        for line in process.stdout:
            print(line.rstrip())
        
        # Wait for process to complete
        return_code = process.wait()
        
        # Check for errors
        if return_code != 0:
            stderr_output = process.stderr.read()
            print(f"[{timestamp}] Error for {symbol}: Process exited with code {return_code}")
            print(f"[{timestamp}] Error details: {stderr_output}")
            
            # Record the failure if not already in failed_companies.txt
            try:
                failed_companies = get_failed_companies()
                if symbol not in failed_companies:
                    with open("failed_companies.txt", "a") as f:
                        f.write(f"{symbol}\n")
                    print(f"[{timestamp}] Added {symbol} to failed_companies.txt")
            except Exception as e:
                print(f"[{timestamp}] Warning: Could not update failed_companies.txt: {e}")
            
            return False
        
        # Success - remove from failed_companies.txt if it's there
        try:
            failed_companies = get_failed_companies()
            if symbol in failed_companies:
                failed_companies.remove(symbol)
                with open("failed_companies.txt", "w") as f:
                    f.write("\n".join(failed_companies) + "\n" if failed_companies else "")
                print(f"[{timestamp}] Removed {symbol} from failed_companies.txt")
        except Exception as e:
            print(f"[{timestamp}] Warning: Could not update failed_companies.txt: {e}")
        
        print(f"{'-' * 60}")
        print(f"[{timestamp}] COMPLETED: {symbol}")
        print(f"{'-' * 60}\n")
        return True
        
    except Exception as e:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Error running prediction for {symbol}: {e}")
        
        # Record the failure if not already in failed_companies.txt
        try:
            failed_companies = get_failed_companies()
            if symbol not in failed_companies:
                with open("failed_companies.txt", "a") as f:
                    f.write(f"{symbol}\n")
                print(f"[{timestamp}] Added {symbol} to failed_companies.txt")
        except Exception as e:
            print(f"[{timestamp}] Warning: Could not update failed_companies.txt: {e}")
        
        return False

def main():
    """
    Main function to retry specific symbols
    """
    parser = argparse.ArgumentParser(description='Retry failed stock predictions with enhanced error handling')
    parser.add_argument('--symbols', type=str, nargs='+', help='Specific symbols to retry')
    parser.add_argument('--file', type=str, help='File containing symbols to retry (one per line)')
    parser.add_argument('--prediction_days', type=int, default=5, help='Number of days ahead to predict')
    parser.add_argument('--threshold', type=float, default=2.0, help='Threshold percentage for significant movements')
    parser.add_argument('--retry_limit', type=int, default=3, help='Maximum number of retry attempts per symbol')
    parser.add_argument('--no_force_training', action='store_true', help='Skip training and use existing models')
    parser.add_argument('--additional_args', type=str, help='Additional arguments to pass to main.py (in quotes)')
    
    args = parser.parse_args()
    
    # Parse additional arguments if provided
    additional_args = args.additional_args.split() if args.additional_args else None
    
    # Get symbols from either command line or file
    symbols = []
    
    if args.symbols:
        symbols.extend(args.symbols)
    
    if args.file and os.path.exists(args.file):
        with open(args.file, 'r') as f:
            file_symbols = [line.strip() for line in f.readlines() if line.strip()]
            symbols.extend(file_symbols)
    
    # If no symbols specified but failed_companies.txt exists, use that
    if not symbols and os.path.exists("failed_companies.txt"):
        symbols = get_failed_companies()
        print(f"No symbols provided, using {len(symbols)} symbols from failed_companies.txt")
    
    if not symbols:
        print("No symbols provided. Please specify symbols directly or through a file.")
        return
    
    # Remove duplicates
    symbols = list(set(symbols))
    
    # Create a timestamp for this run
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Starting retry run at {timestamp}")
    print(f"Attempting to retry {len(symbols)} symbols with {'OPTIONAL' if args.no_force_training else 'FORCED'} training...")
    
    successful = []
    failed = []
    
    # Process each symbol
    for i, symbol in enumerate(symbols):
        print(f"\nProcessing {i+1}/{len(symbols)}: {symbol}")
        
        # Try multiple times
        success = False
        attempts = 0
        
        while not success and attempts < args.retry_limit:
            if attempts > 0:
                print(f"Retry attempt {attempts+1} for {symbol}...")
                time.sleep(2)  # Wait before retry
            
            success = run_stock_prediction(
                symbol,
                prediction_days=args.prediction_days,
                threshold=args.threshold,
                force_training=not args.no_force_training,  # Force training unless explicitly disabled
                additional_args=additional_args
            )
            
            attempts += 1
        
        if success:
            successful.append(symbol)
        else:
            failed.append(symbol)
        
        # Progress update
        print(f"\nProgress: {i+1}/{len(symbols)} processed")
        print(f"Successful: {len(successful)}, Failed: {len(failed)}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("RETRY RESULTS")
    print("=" * 60)
    print(f"Total symbols processed: {len(symbols)}")
    print(f"Successfully fixed: {len(successful)}")
    print(f"Still failing: {len(failed)}")
    
    if successful:
        print("\nSuccessfully fixed symbols:")
        for symbol in successful:
            print(f" - {symbol}")
    
    if failed:
        print("\nStill failing symbols:")
        for symbol in failed:
            print(f" - {symbol}")
        
        # Save still-failing symbols to a new file
        with open("still_failing.txt", "w") as f:
            for symbol in failed:
                f.write(f"{symbol}\n")
        print(f"\nList of still failing symbols saved to 'still_failing.txt'")
    
    print("=" * 60)
    print(f"Retry run completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 