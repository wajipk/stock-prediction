import pandas as pd
import tensorflow as tf
from tensorflow.keras import mixed_precision
import numpy as np
from loaddata import StockData
from model import StockPredictionModel


class ModelTrainer:
    def __init__(self, seq_length=365, batch_size=16, epochs=50):
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.epochs = epochs

    def prepare_sequences(self, data, features, periods):
        X_batch, y_batch = [], {period: [] for period in periods}
        print("Preparing sequences for training...")

        for i in range(len(data) - self.seq_length):
            X_batch.append(data[features].values[i:i + self.seq_length])
            for period in periods:
                y_batch[period].append(data[f'target_{period}'].values[i + self.seq_length - 1])

            if len(X_batch) == self.batch_size:
                y_output = np.array([y_batch[period] for period in periods]).T
                yield np.array(X_batch), y_output
                X_batch, y_batch = [], {period: [] for period in periods}

        if len(X_batch) > 0:
            y_output = np.array([y_batch[period] for period in periods]).T
            yield np.array(X_batch), y_output

    def train(self, model_path='./stock_model.keras'):
        try:
            print("Initializing training process...")

            policy = mixed_precision.Policy('mixed_float16')  # Use 16-bit precision
            mixed_precision.set_global_policy(policy)

            physical_devices = tf.config.experimental.list_physical_devices('GPU')
            if physical_devices:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
                print("GPU detected and enabled for training.")
            else:
                print("No GPU detected, using CPU for training.")

            stock_data = StockData()
            print("Loading preprocessed data...")
            data = pd.read_csv(stock_data.processed_file)
            print(f"Preprocessed data loaded successfully with {len(data)} records.")

            features = [
                'close', 'volume', 'volatility', 'ma_20', 'ma_50',
                'rsi', 'macd', 'open', 'symbol_encoded'
            ]
            periods = {f'day{i}': i for i in range(1, 31)}

            print("Initializing the stock prediction model...")
            model = StockPredictionModel(self.seq_length, len(features), len(periods))
            print("Model initialized successfully.")

            sequence_generator = self.prepare_sequences(data, features, periods)

            steps_per_epoch = max(1, len(data) // self.batch_size)
            validation_steps = max(1, len(data) // self.batch_size)

            print("Starting model training...")
            history = model.model.fit(
                sequence_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=self.epochs,
                validation_data=sequence_generator,
                validation_steps=validation_steps,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                    tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True)
                ]
            )

            print("Saving the trained model...")
            model.save_model(model_path)
            print("Model trained and saved successfully!")

            return history

        except Exception as e:
            print(f"Error during training: {e}")
            raise
