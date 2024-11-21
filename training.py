import pandas as pd
import tensorflow as tf
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
            if len(data) < self.seq_length + max(periods.values()):
                raise ValueError("Insufficient data for training.")

            period_keys = list(periods.keys())
            total_sequences = len(data) - self.seq_length - max(periods.values()) + 1

            def sequence_generator():
                for i in range(total_sequences):
                    input_seq = data[features].values[i:i + self.seq_length]
                    target_values = [data[f'target_{period}'].values[i + self.seq_length - 1] for period in period_keys]
                    yield input_seq, target_values

            dataset = tf.data.Dataset.from_generator(
                sequence_generator,
                output_signature=(
                    tf.TensorSpec(shape=(self.seq_length, len(features)), dtype=tf.float32),
                    tf.TensorSpec(shape=(len(periods),), dtype=tf.float32)
                )
            )

            dataset = dataset.repeat().shuffle(buffer_size=min(total_sequences, 10000))
            dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

            print(f"Prepared dataset for training with {total_sequences} sequences.")
            return dataset

        except Exception as e:
            print(f"Error in sequence preparation: {e}")
            raise

    def train(self, model_path='./stock_model.keras'):
        try:
            # GPU configuration
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

            # Data loading
            print("Loading preprocessed data...")
            stock_data = StockData()
            data, features = stock_data.load_stock_data()

            periods = {f'day{i}': i for i in range(1, 31)}
            validation_split = 0.2
            train_size = int(len(data) * (1 - validation_split))
            train_data = data[:train_size]
            val_data = data[train_size:]

            train_dataset = self.prepare_sequences(train_data, features, periods)
            val_dataset = self.prepare_sequences(val_data, features, periods)

            # Model initialization
            print("Initializing stock prediction model...")
            model = StockPredictionModel(self.seq_length, len(features), len(periods))

            # Learning rate scheduler
            learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.001,
                decay_steps=500,
                decay_rate=0.95
            )
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

            model.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

            # Callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
                tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True)
            ]

            # Training
            print("Starting model training...")
            history = model.model.fit(
                train_dataset,
                validation_data=val_dataset,
                steps_per_epoch=len(train_data) // self.batch_size,
                validation_steps=len(val_data) // self.batch_size,
                epochs=self.epochs,
                callbacks=callbacks
            )

            model.save_model(model_path)
            print("Model training completed successfully.")
            return history

        except Exception as e:
            print(f"Error during training: {e}")
            raise
