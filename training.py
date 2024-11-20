import pandas as pd
import tensorflow as tf
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
    
        for i in range(len(data) - self.seq_length):
            # Append the sequence for this step
            X_batch.append(data[features].values[i:i + self.seq_length])
        
            # Append the target values for each period (all periods in the dictionary)
            for period in periods:
                y_batch[period].append(data[f'target_{period}'].values[i + self.seq_length - 1])
        
            # Once we reach the batch size, yield the current batch and reset the lists
            if len(X_batch) == self.batch_size:
                # Ensure y_batch contains all periods
                y_output = np.array([y_batch[period] for period in periods]).T
                yield np.array(X_batch), y_output  # We yield the data and multi-output targets
                X_batch, y_batch = [], {period: [] for period in periods}  # Reset the batch after yielding
        
        # Yield the final batch if there are any remaining sequences
        if len(X_batch) > 0:
            y_output = np.array([y_batch[period] for period in periods]).T
            yield np.array(X_batch), y_output


    def train(self, model_path='stock-prediction/stock_model.keras'):
        try:
            # Load preprocessed data
            stock_data = StockData()
            data = pd.read_csv(stock_data.processed_file)
            features = [
            'close', 'volume', 'volatility', 'ma_20', 'ma_50', 
            'rsi', 'macd', 'open', 'symbol_encoded'
            ]

            # Define prediction periods (Day 1 to Day 30)
            periods = {f'day{i}': i for i in range(1, 31)}  # Day 1 to 30

            # Initialize model
            model = StockPredictionModel(self.seq_length, len(features), len(periods))

            # Create a generator for the sequences
            sequence_generator = self.prepare_sequences(data, features, periods)

            # Train model using the generator
            history = model.model.fit(
                sequence_generator,
                steps_per_epoch=len(data) // self.batch_size,  # Number of batches per epoch
                epochs=self.epochs,
                validation_data=sequence_generator,
                validation_steps=len(data) // self.batch_size,
                callbacks=[ 
                    tf.keras.callbacks.EarlyStopping(patience=10),
                    tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True)
                ]
            )

            # Save the trained model in the new `.keras` format
            model.save_model(model_path)
            return history

        except Exception as e:
            print(f"Error during training: {e}")
            raise
