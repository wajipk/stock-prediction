import pandas as pd
import tensorflow as tf
import numpy as np
from loaddata import StockData
from model import StockPredictionModel


class ModelTrainer:
    def __init__(self, seq_length=365, batch_size=32, epochs=50):
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.epochs = epochs

    def prepare_sequences(self, data, features, periods):
        X, y = [], {period: [] for period in periods}
        for i in range(len(data) - self.seq_length):
            X.append(data[features].values[i:i + self.seq_length])
            for period in periods:
                y[period].append(data[f'target_{period}'].values[i + self.seq_length - 1])
        return np.array(X), {period: np.array(y[period]) for period in periods}

    def train(self, model_path='models/stock_model.h5'):
        try:
            # Load preprocessed data
            stock_data = StockData()
            data = pd.read_csv(stock_data.processed_file)
            features = [
                'close', 'volume', 'volatility', 'ma_20', 'ma_50', 
                'rsi', 'macd', 'open', 'symbol_encoded'
            ]

            # Define prediction periods
            periods = {
                'day1': 1,
                'day2': 2,
                'day3': 3,
                'week1': 7,
                'month1': 30,
                'month2': 60,
                'quarterly': 90,
                'halfyear': 180,
                'year': 365,
            }

            # Prepare data sequences
            X, y = self.prepare_sequences(data, features, periods)

            # Split data
            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train = {period: y[period][:split] for period in periods}
            y_test = {period: y[period][split:] for period in periods}

            # Initialize model
            model = StockPredictionModel(self.seq_length, len(features), len(periods))

            # Train model
            history = model.model.fit(
                X_train, [y_train[period] for period in periods],
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_data=(X_test, [y_test[period] for period in periods]),
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(patience=10),
                    tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True)
                ]
            )
            model.save_model(model_path)
            return history

        except Exception as e:
            print(f"Error during training: {e}")
            raise
