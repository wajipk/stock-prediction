from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from model import StockPredictionModel
from loaddata import StockData


class StockPredictor:
    def __init__(self, model_path='./stock_model.keras'):
        print("Loading model and encoders...")
        self.model = StockPredictionModel.load_model(model_path)
        self.stock_data = StockData()
        self.stock_data.load_encoder_and_scaler()
        print("Model and encoders loaded successfully.")

    def predict_all_periods(self, symbol: str):
        try:
            print(f"Generating predictions for symbol: {symbol}...")
            df = pd.read_csv(self.stock_data.processed_file)
            features = [
                'close', 'volume', 'volatility', 'ma_20', 'ma_50',
                'rsi', 'macd', 'open', 'symbol_encoded'
            ]

            if symbol not in self.stock_data.label_encoder.classes_:
                raise ValueError(f"Symbol '{symbol}' not found in the dataset.")
            symbol_encoded = self.stock_data.label_encoder.transform([symbol])[0]

            data = df[df['symbol_encoded'] == symbol_encoded]
            if data.empty:
                raise ValueError(f"No data available for symbol: {symbol}")

            seq_length = len(features)
            X = data[features].values[-seq_length:].reshape(1, seq_length, -1)

            periods = {f'day{i}': i for i in range(1, 31)}
            predictions = {}

            for period, days in periods.items():
                pred = self.model.predict(X)[0][0]
                predictions[period] = {
                    'predicted_price': float(pred),
                    'prediction_date': (datetime.now() + timedelta(days=days)).strftime('%Y-%m-%d')
                }
                X = np.roll(X, -1, axis=1)
                X[0, -1] = pred

            print("Predictions generated successfully.")
            return predictions

        except Exception as e:
            print(f"Error during prediction: {e}")
            raise
