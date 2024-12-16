import pandas as pd
import numpy as np
from model import StockPredictionModel
from loaddata import StockData
import os
from pandas.tseries.offsets import BDay


class StockPredictor:
    def __init__(self, model_path='./stock_model.keras'):
        print("Loading model and encoders...")
        self.model = StockPredictionModel.load_model(model_path)
        self.stock_data = StockData()
        self.stock_data.load_encoder_and_scaler()
        print("Model and encoders loaded successfully.")

    def get_next_business_day(self, date, days=1):
        """Get the next business day(s) after the given date."""
        return date + BDay(days)

    def retrieve_scaled_symbol(self, symbol):
        """
        Retrieve the encoded value of a symbol and its corresponding scaled value using the complete mapping file.
        """
        try:
            # Load the complete mapping file
            mapping_file = os.path.join(self.stock_data.temp_directory, 'symbol_mapping_complete.csv')
            symbol_mapping = pd.read_csv(mapping_file)

            # Get the row corresponding to the symbol
            matching_row = symbol_mapping[symbol_mapping['symbol'] == symbol]
            if matching_row.empty:
                raise ValueError(f"Symbol '{symbol}' not found in the mapping file.")

            # Retrieve encoded and scaled values
            encoded_symbol = matching_row['symbol_encoded_unscaled'].iloc[0]
            scaled_value = matching_row['symbol_encoded_scaled_y'].iloc[0]

            print(f"Encoded value for '{symbol}': {encoded_symbol}")
            print(f"Scaled value for '{symbol}': {scaled_value}")

            return scaled_value

        except FileNotFoundError:
            print("Error: Mapping file not found. Ensure preprocessing has been run.")
            raise

        except Exception as e:
            print(f"Error retrieving scaled symbol for '{symbol}': {e}")
            raise

    def predict_all_periods(self, symbol: str):
        try:
            print(f"Generating predictions for symbol: {symbol}...")

            # Read the processed data
            df = pd.read_csv(self.stock_data.processed_file)
            df['date'] = pd.to_datetime(df['date'])

            # Define feature columns
            features = ['close', 'volume', 'volatility', 'ma_14', 'ma_30', 'ma_50', 'rsi_14', 'rsi_30', 'rsi_50',
                        'macd', 'obv', 'force_index', 'symbol_encoded']

            # Retrieve the scaled value of the symbol
            scaled_value = self.retrieve_scaled_symbol(symbol)

            # Filter data for the specific symbol (use scaled_value instead of encoded_symbol)
            data = df[df['symbol_encoded'] == scaled_value].copy()

            if data.empty:
                raise ValueError(f"No processed data available for symbol: {symbol}")

            # Check sequence length requirement
            seq_length = self.model.input_shape[1]
            if len(data) < seq_length:
                raise ValueError(f"Insufficient data for symbol {symbol}. "
                                 f"Need at least {seq_length} data points.")

            # Sort data by date
            data = data.sort_values('date')
            maximum_date = data['date'].max()
            latest_date = maximum_date + BDay(30)

            # Prepare input sequence for the model
            X = data[features].values[-seq_length:].reshape(1, seq_length, len(features))

            # Generate predictions
            raw_predictions = self.model.predict(X)

            # Initialize the predictions dictionary
            predictions = {}

            # Create dummy array for inverse scaling
            dummy_array = np.zeros((1, len(self.stock_data.scaler.min_)))  # Match the scaler's fitted shape

            for i in range(30):
                period = f'day{i+1}'

                # Get the predicted date (business day only)
                pred_date = self.get_next_business_day(latest_date, days=i+1)

                # Insert prediction in the correct target column
                target_index = len(features) - 1  # Use the correct target index within scaler dimensions
                dummy_array[0, target_index] = raw_predictions[0][i]

                # Inverse transform to get the unscaled prediction
                unscaled_pred = self.stock_data.scaler.inverse_transform(dummy_array)[0, target_index]

                # Add to predictions dictionary
                predictions[period] = {
                    'date': pred_date.strftime('%Y-%m-%d'),
                    'predicted_price': float(unscaled_pred)
                }

            print("Predictions generated successfully.")
            return predictions

        except Exception as e:
            print(f"Error during prediction: {e}")
            raise
