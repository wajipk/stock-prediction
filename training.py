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
        try:
            print("Preparing sequences for training...")
            
            # Ensure we have enough data
            if len(data) < self.seq_length + max(periods.values()):
                raise ValueError("Insufficient data for training. Data length is too short for the specified sequence length and prediction periods.")

            def sequence_generator():
                period_keys = list(periods.keys())
                
                for i in range(len(data) - self.seq_length - max(periods.values()) + 1):
                    # Extract input sequence
                    input_seq = data[features].values[i:i + self.seq_length]
                    
                    # Extract target values for different periods
                    target_values = [data[f'target_{period}'].values[i + self.seq_length - 1] for period in period_keys]
                    
                    yield input_seq, target_values

            # Create tf.data.Dataset from generator
            dataset = tf.data.Dataset.from_generator(
                sequence_generator,
                output_signature=(
                    tf.TensorSpec(shape=(self.seq_length, len(features)), dtype=tf.float32),
                    tf.TensorSpec(shape=(len(periods),), dtype=tf.float32)
                )
            )

            # Process dataset
            dataset = dataset.shuffle(buffer_size=1000)  # Smaller buffer to reduce memory usage
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

            print(f"Prepared dataset for training")
            return dataset

        except Exception as e:
            print(f"Error in sequence preparation: {e}")
            raise

    def train(self, model_path='./stock_model.keras'):
        try:
            # Limit GPU memory growth
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    print("GPU memory growth limited.")
                except RuntimeError as e:
                    print(e)

            # Use lower precision to reduce memory usage
            tf.keras.mixed_precision.set_global_policy('mixed_float16')

            # Load data efficiently
            print("Loading preprocessed data...")
            stock_data = StockData()
            data = pd.read_csv(stock_data.processed_file, low_memory=False)
            print(f"Preprocessed data loaded with {len(data)} records.")

            features = [
                'close', 'volume', 'volatility', 'ma_20', 'ma_50',
                'rsi', 'macd', 'open', 'symbol_encoded'
            ]
            periods = {f'day{i}': i for i in range(1, 31)}

            # Prepare dataset
            print("Preparing training dataset...")
            dataset = self.prepare_sequences(data, features, periods)

            # Model configuration
            print("Initializing stock prediction model...")
            model = StockPredictionModel(self.seq_length, len(features), len(periods))
            
            # Custom learning rate
            learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.001,
                decay_steps=100,
                decay_rate=0.9
            )
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            
            # Compile model
            model.model.compile(
                optimizer=optimizer, 
                loss='mse', 
                metrics=['mae']
            )

            # Callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='loss', 
                    patience=10, 
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='loss', 
                    factor=0.5, 
                    patience=5, 
                    min_lr=1e-6
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    model_path, 
                    save_best_only=True
                )
            ]

            # Training
            print("Starting model training...")
            history = model.model.fit(
                dataset,
                epochs=self.epochs,
                callbacks=callbacks
            )

            # Save model
            print("Saving trained model...")
            model.save_model(model_path)
            print("Model training completed successfully!")

            return history

        except Exception as e:
            print(f"Error during training: {e}")
            raise
