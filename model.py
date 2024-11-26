import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers


class StockPredictionModel:
    def __init__(self, seq_length=365, n_features=None, n_outputs=None):
        self.seq_length = seq_length
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.model = self._build_model()

    def _build_model(self):
        print("Building the model...")
        model = Sequential([
            Input(shape=(self.seq_length, self.n_features)),

            # LSTM layer with L2 regularization
            LSTM(128, return_sequences=True, kernel_regularizer=regularizers.l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),

            # LSTM layer with L2 regularization
            LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),

            # LSTM layer with L2 regularization
            LSTM(32, return_sequences=False, kernel_regularizer=regularizers.l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),

            # Dense layers with L2 regularization
            Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            BatchNormalization(),
            Dropout(0.2),

            Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            BatchNormalization(),
            Dropout(0.2),

            # Output layer
            Dense(self.n_outputs)
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
