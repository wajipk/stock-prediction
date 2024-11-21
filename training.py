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

            # Prepare input features and target values
            X, y = [], []
            period_keys = list(periods.keys())

            for i in range(len(data) - self.seq_length - max(periods.values()) + 1):
                # Extract input sequence
                input_seq = data[features].values[i:i + self.seq_length]
            
                # Extract target values for different periods
                target_values = [data[f'target_{period}'].values[i + self.seq_length - 1] for period in period_keys]
            
                X.append(input_seq)
                y.append(target_values)

            # Convert to numpy arrays
            X = np.array(X)
            y = np.array(y)

            # Create tf.data.Dataset
            dataset = tf.data.Dataset.from_tensor_slices((X, y))
            dataset = dataset.shuffle(buffer_size=len(X))
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

            print(f"Prepared {len(X)} sequences for training")
            return dataset

        except Exception as e:
            print(f"Error in sequence preparation: {e}")
            raise

    def train(self, model_path='./stock_model.keras'):
        try:
            print("Initializing training process...")

            # Enable mixed precision training
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)

            # GPU configuration
            physical_devices = tf.config.experimental.list_physical_devices('GPU')
            if physical_devices:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
                print("GPU detected and enabled for training.")
            else:
                print("No GPU detected, using CPU for training.")

            # Load and prepare data
            stock_data = StockData()
            print("Loading preprocessed data...")
            data = pd.read_csv(stock_data.processed_file)
            print(f"Preprocessed data loaded successfully with {len(data)} records.")

            features = [
                'close', 'volume', 'volatility', 'ma_20', 'ma_50',
                'rsi', 'macd', 'open', 'symbol_encoded'
            ]
            periods = {f'day{i}': i for i in range(1, 31)}

            # Prepare dataset
            dataset = self.prepare_sequences(data, features, periods)

            # Split dataset into train and validation
            dataset_size = len(list(dataset))
            train_size = int(0.8 * dataset_size)
        
            train_dataset = dataset.take(train_size)
            val_dataset = dataset.skip(train_size)

            # Initialize and compile model
            print("Initializing the stock prediction model...")
            model = StockPredictionModel(self.seq_length, len(features), len(periods))
            print("Model initialized successfully.")

            # Learning rate scheduling
            initial_learning_rate = 0.001
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate,
                decay_steps=100,
                decay_rate=0.9,
                staircase=True
            )
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
            model.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

            # Model training
            print("Starting model training...")
            history = model.model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=self.epochs,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        patience=10, 
                        restore_best_weights=True, 
                        monitor='val_loss'
                    ),
                    tf.keras.callbacks.ModelCheckpoint(
                        model_path, 
                        save_best_only=True, 
                        monitor='val_loss'
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss', 
                        factor=0.5, 
                        patience=5, 
                        min_lr=1e-6
                    )
                ]
            )

            # Save the model
            print("Saving the trained model...")
            model.save_model(model_path)
            print("Model trained and saved successfully!")

            return history

        except Exception as e:
            print(f"Error during training: {e}")
            raise
