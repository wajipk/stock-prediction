import os
import tensorflow as tf
from loaddata import StockData
from model import StockPredictionModel


class ModelTrainer:
    def __init__(self, seq_length=60, batch_size=32, epochs=50):
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
            # OS-based GPU configuration check
            system_os = os.name  # 'posix' for Linux, 'nt' for Windows

            if system_os == 'posix':  # This means we're on Linux (including Ubuntu)
                # Use the old code for GPU config in Ubuntu (Linux-based OS)
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                print("Using GPU configuration method for Ubuntu (Linux).")

            elif system_os == 'nt':  # This means we're on Windows
                # Use the new code for GPU config in Windows
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    for gpu in gpus:
                        tf.config.set_memory_growth(gpu, True)
                print("Using GPU configuration method for Windows.")

            else:
                print("Unsupported OS for automatic GPU configuration.")
            
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

            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

            model.model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

            # Training
            print("Starting model training...")
            history = model.model.fit(
                train_dataset,
                validation_data=val_dataset,
                steps_per_epoch=len(train_data) // self.batch_size,
                validation_steps=len(val_data) // self.batch_size,
                epochs=self.epochs,
            )

            model.save_model(model_path)
            print("Model training completed successfully.")
            return history

        except Exception as e:
            print(f"Error during training: {e}")
            raise