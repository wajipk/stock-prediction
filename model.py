import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


class StockPredictionModel:
    def __init__(self, seq_length=365, n_features=None, n_outputs=None):
        self.seq_length = seq_length
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            # First LSTM layer for long sequence processing
            LSTM(128, return_sequences=True, input_shape=(self.seq_length, self.n_features)),
            Dropout(0.3),  # Dropout to prevent overfitting

            # Second LSTM layer
            LSTM(64, return_sequences=True),
            Dropout(0.3),

            # Third LSTM layer
            LSTM(32, return_sequences=False),
            Dropout(0.3),

            # Dense layer for intermediate representation
            Dense(64, activation='relu'),
            Dropout(0.2),

            # Final output layer for multi-step predictions
            Dense(self.n_outputs)
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def save_model(self, path):
        self.model.save(path)

    @classmethod
    def load_model(cls, path):
        return tf.keras.models.load_model(path)