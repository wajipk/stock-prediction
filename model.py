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
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(self.seq_length, self.n_features)),
            Dropout(0.3),

            LSTM(64, return_sequences=True),
            Dropout(0.3),

            LSTM(32, return_sequences=False),
            Dropout(0.3),

            Dense(64, activation='relu'),
            Dropout(0.2),

            # Output layer with 30 units for multi-output prediction (one for each day)
            Dense(self.n_outputs)
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def save_model(self, path):
        self.model.save(path)

    @classmethod
    def load_model(cls, path):
        return tf.keras.models.load_model(path)