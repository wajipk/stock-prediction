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
            mapping_file = os.path.join(self.stock_data.data_path, 'symbol_mapping_complete.csv')
            symbol_mapping = pd.read_csv(mapping_file)

            # Get the row corresponding to the symbol
            matching_row = symbol_mapping[symbol_mapping['symbol'] == symbol]
            if matching_row.empty:
                raise ValueError(f"Symbol '{symbol}' not found in the mapping file.")

            # Retrieve encoded and scaled values
            encoded_symbol = matching_row['symbol_encoded_unscaled'].iloc[0]
            scaled_value = matching_row['symbol_encoded_scaled_y'].iloc[0]

            print(f"Encoded value for '{symbol}': {encoded_symbol}")
            print(f"MinMax-scaled value for '{symbol}': {scaled_value}")

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
            features = [
                'close', 'volume', 'volatility', 'ma_20', 'ma_50',
                'rsi', 'macd', 'open', 'symbol_encoded'
            ]

            # Retrieve the encoded and scaled value of the symbol
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
            dummy_array = np.zeros((1, len(features) + 30))

            for i in range(30):
                period = f'day{i+1}'

                # Get the predicted date (business day only)
                pred_date = self.get_next_business_day(latest_date, days=i+1)

                # Scale and inverse transform predictions
                dummy_array[0, len(features) + i] = raw_predictions[0][i]
                unscaled_pred = self.stock_data.scaler.inverse_transform(dummy_array)[0, len(features) + i]

                # Add to predictions dictionary
                predictions[period] = {
                    'date': pred_date.strftime('%Y-%m-%d'),
                    'predicted_price': float(unscaled_pred)
                }

            # Update technical indicators in predictions
            self._update_technical_indicators(predictions, data.iloc[-1])

            print("Predictions generated successfully.")
            return predictions

        except Exception as e:
            print(f"Error during prediction: {e}")
            raise

    def _update_technical_indicators(self, predictions, last_row):
        """Update technical indicators for the predicted prices."""
        try:
            # Create a temporary DataFrame for historical and predicted prices
            temp_df = pd.DataFrame(predictions).T
            temp_df['date'] = pd.to_datetime(temp_df['date'])
            temp_df = temp_df.sort_values('date')

            # Combine historical and predicted prices
            historical_prices = [last_row['close']]
            predicted_prices = [pred['predicted_price'] for pred in predictions.values()]
            all_prices = pd.Series(historical_prices + predicted_prices)

            # Calculate returns
            returns = all_prices.pct_change().dropna()

            # Calculate technical indicators
            volatility = returns.rolling(window=20).std()
            ma_20 = all_prices.rolling(window=20).mean()
            ma_50 = all_prices.rolling(window=50).mean()
            delta = all_prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            exp1 = all_prices.ewm(span=12, adjust=False).mean()
            exp2 = all_prices.ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2

        except Exception as e:
            print(f"Error updating technical indicators: {e}")
            raise
