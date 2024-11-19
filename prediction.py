from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from model import StockPredictionModel
from loaddata import StockData


class StockPredictor:
    def __init__(self, model_path='stock-prediction/stock_model.h5'):
        self.model = StockPredictionModel.load_model(model_path)
        self.stock_data = StockData()
        # Load the LabelEncoder and Scaler from stock-prediction directory
        self.stock_data.load_encoder_and_scaler()

    def predict_all_periods(self, symbol: str):
        try:
            # Load processed data
            df = pd.read_csv(self.stock_data.processed_file)
            features = [
                'close', 'volume', 'volatility', 'ma_20', 'ma_50',
                'rsi', 'macd', 'open', 'symbol_encoded'
            ]

            # Encode symbol
            if symbol not in self.stock_data.label_encoder.classes_:
                raise ValueError(f"Symbol '{symbol}' not found in the dataset.")
            symbol_encoded = self.stock_data.label_encoder.transform([symbol])[0]

            # Filter data for the symbol
            data = df[df['symbol_encoded'] == symbol_encoded]
            if data.empty:
                raise ValueError(f"No data available for symbol: {symbol}")

            seq_length = len(features)

            # Prepare the most recent input sequence for today's prediction
            X = data[features].values[-seq_length:].reshape(1, seq_length, -1)

            # Define prediction periods (1 to 30 days)
            periods = {f'day {i}': i for i in range(1, 31)}

            # Store predictions
            predictions = {}

            # Generate predictions for all periods
            for period, days in periods.items():
                pred = self.model.predict(X)[0][0]
                predictions[period] = {
                    'predicted_price': float(pred),
                    'prediction_date': (datetime.now() + timedelta(days=days)).strftime('%Y-%m-%d')
                }
                # Update the sequence with the predicted price
                X = np.roll(X, -1, axis=1)
                X[0, -1] = pred

            return predictions

        except Exception as e:
            print(f"Error during prediction: {e}")
            raise
