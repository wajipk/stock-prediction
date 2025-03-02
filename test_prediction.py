"""
Test script to run prediction on a trained model
"""
from src.predict import main as predict_main, apply_psx_price_limits
from src.data_collection import load_corporate_actions
import sys

# Symbol to test
symbol = 'MARI'

# Test PSX price limit rules
print("Testing PSX price limit rules...")
previous_price = 100.0  # Example price in PKR

# Test case 1: Price within limits
predicted_price = 105.0  # +5%
limited_price = apply_psx_price_limits(predicted_price, previous_price)
print(f"Test 1: Previous: PKR {previous_price:.2f}, Predicted: PKR {predicted_price:.2f}, After limits: PKR {limited_price:.2f}")

# Test case 2: Price above upper limit
predicted_price = 120.0  # +20%, exceeds +10% limit
limited_price = apply_psx_price_limits(predicted_price, previous_price)
print(f"Test 2: Previous: PKR {previous_price:.2f}, Predicted: PKR {predicted_price:.2f}, After limits: PKR {limited_price:.2f}")

# Test case 3: Price below lower limit
predicted_price = 85.0  # -15%, exceeds -10% limit
limited_price = apply_psx_price_limits(predicted_price, previous_price)
print(f"Test 3: Previous: PKR {previous_price:.2f}, Predicted: PKR {predicted_price:.2f}, After limits: PKR {limited_price:.2f}")

# Test case 4: Low price stock
previous_price = 5.0
predicted_price = 7.0  # +40%, exceeds +10% but is only +2 PKR
limited_price = apply_psx_price_limits(predicted_price, previous_price)
print(f"Test 4: Previous: PKR {previous_price:.2f}, Predicted: PKR {predicted_price:.2f}, After limits: PKR {limited_price:.2f}")

print("\n" + "-" * 50 + "\n")

# Load corporate actions with symbol for API fetching
print(f"Loading corporate actions for {symbol}...")
load_corporate_actions(symbol)

# Set up arguments for prediction
predict_args = [
    '--symbol', symbol,
    '--days', '7',
    '--window', '10',
    '--threshold', '2.0'
    # Remove the '--no_rules' flag to apply financial rules
]

# Set sys.argv and run prediction
old_argv = sys.argv
sys.argv = ['predict.py'] + predict_args

try:
    # Run prediction
    predict_main()
except Exception as e:
    print(f"Error during prediction: {e}")
finally:
    # Restore original sys.argv
    sys.argv = old_argv

print("Prediction test completed with financial rules applied!") 