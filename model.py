import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


class StockPredictionModel:
    def __init__(self, seq_length=365, n_features=None, n_outputs=None):
        self.seq_length = seq_length
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.model = self._build_model()

    def _build_model(self):
        print("Building the model...")
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(self.seq_length, self.n_features)),
            Dropout(0.3),

            LSTM(64, return_sequences=True),
            Dropout(0.3),

            LSTM(32, return_sequences=False),
            Dropout(0.3),

            Dense(64, activation='relu'),
            Dropout(0.2),

            Dense(self.n_outputs)  # Output layer for multi-output prediction
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        print("Model built successfully.")
        return model

    def save_model(self, path):
        self.model.save(path)
        print(f"Model saved to {path}")

    @classmethod
    def load_model(cls, path):
        print(f"Loading model from {path}...")
        model = tf.keras.models.load_model(path)
        print("Model loaded successfully.")
        return model
