import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, GRU, Conv1D
from tensorflow.keras.layers import MaxPooling1D, Flatten, Concatenate, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from src.prediction_reward_system import PredictionRewardSystem


class ReinforcedStockModel:
    """
    A reinforcement learning enhanced stock prediction model that can adapt based on
    historical prediction accuracy and learn from its mistakes.
    """
    
    def __init__(self, symbol, window_size=10, base_model_type='lstm', model_dir='models', 
                 learning_rate=0.001, dropout_rate=0.3, error_threshold=0.05):
        """
        Initialize the reinforced stock model
        
        Args:
            symbol (str): Stock symbol
            window_size (int): Window size for the sequential data
            base_model_type (str): Base model type ('lstm', 'bilstm', 'gru', 'cnn_lstm', 'advanced')
            model_dir (str): Directory to save the model
            learning_rate (float): Base learning rate for the model
            dropout_rate (float): Base dropout rate for regularization
            error_threshold (float): Error threshold to determine adaptation needs
        """
        self.symbol = symbol
        self.window_size = window_size
        self.base_model_type = base_model_type
        self.model_dir = os.path.join(model_dir, symbol)
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.error_threshold = error_threshold
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize reward system
        self.reward_system = PredictionRewardSystem(symbol, threshold=error_threshold, models_dir=model_dir)
        self.reward_system.set_model_info("reinforced", f"reinforced_{base_model_type}")
        
        # Base model (will be initialized when needed)
        self.base_model = None
        
        # Error correction layers (will be initialized when needed)
        self.error_correction_model = None
        
        # Combined model
        self.combined_model = None
        
        # Store last prediction details
        self.last_prediction = {
            'date': None,
            'input_features': None,
            'predicted_price': None,
            'actual_price': None,
            'error': None
        }
        
        # Keep a memory of recent predictions and errors
        self.prediction_memory = []
        self.max_memory_size = 50  # Store last 50 predictions
        
        # Adaptation parameters
        self.adaptation_frequency = 5  # Adapt after this many predictions with actual values
        self.min_adaptations_needed = 10  # Need at least this many actual values to start adapting
        
        # Load models if they exist
        self._load_or_initialize_models()
    
    def _load_or_initialize_models(self):
        """
        Load existing models if they exist, or initialize new ones
        """
        base_model_path = os.path.join(self.model_dir, f'base_{self.base_model_type}_model.keras')
        error_model_path = os.path.join(self.model_dir, 'error_correction_model.keras')
        combined_model_path = os.path.join(self.model_dir, 'combined_reinforced_model.keras')
        
        # Try to load existing models
        try:
            if os.path.exists(base_model_path):
                print(f"Loading base model from {base_model_path}")
                self.base_model = load_model(base_model_path)
            
            if os.path.exists(error_model_path):
                print(f"Loading error correction model from {error_model_path}")
                self.error_correction_model = load_model(error_model_path)
            
            if os.path.exists(combined_model_path):
                print(f"Loading combined reinforced model from {combined_model_path}")
                self.combined_model = load_model(combined_model_path)
                
            print("Models loaded successfully!")
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Will initialize new models when needed.")
    
    def create_base_model(self, input_shape):
        """
        Create the base stock prediction model
        
        Args:
            input_shape (tuple): Shape of input data (window_size, n_features)
            
        Returns:
            tf.keras.Model: Compiled base model
        """
        # Store the input shape for future reference
        self.input_shape = input_shape
        print(f"Creating base model with input shape: {input_shape}")
        
        if self.base_model_type == 'lstm':
            model = tf.keras.Sequential([
                LSTM(units=50, return_sequences=True, input_shape=input_shape),
                Dropout(self.dropout_rate),
                LSTM(units=50, return_sequences=False),
                Dropout(self.dropout_rate),
                Dense(units=25),
                Dense(units=1)
            ])
        elif self.base_model_type == 'bilstm':
            model = tf.keras.Sequential([
                Bidirectional(LSTM(units=50, return_sequences=True), input_shape=input_shape),
                Dropout(self.dropout_rate),
                Bidirectional(LSTM(units=50, return_sequences=False)),
                Dropout(self.dropout_rate),
                Dense(units=25),
                Dense(units=1)
            ])
        elif self.base_model_type == 'gru':
            model = tf.keras.Sequential([
                GRU(units=50, return_sequences=True, input_shape=input_shape),
                Dropout(self.dropout_rate),
                GRU(units=50, return_sequences=False),
                Dropout(self.dropout_rate),
                Dense(units=25),
                Dense(units=1)
            ])
        elif self.base_model_type == 'cnn_lstm':
            model = tf.keras.Sequential([
                Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
                MaxPooling1D(pool_size=2),
                Dropout(self.dropout_rate),
                LSTM(units=50, return_sequences=False),
                Dropout(self.dropout_rate),
                Dense(units=25),
                Dense(units=1)
            ])
        elif self.base_model_type == 'advanced':
            # Input layer
            inputs = Input(shape=input_shape)
            
            # Convolutional branch - capturing local patterns
            conv_layer = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
            conv_layer = BatchNormalization()(conv_layer)
            conv_layer = MaxPooling1D(pool_size=2)(conv_layer)
            conv_layer = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(conv_layer)
            conv_layer = BatchNormalization()(conv_layer)
            conv_layer = MaxPooling1D(pool_size=2)(conv_layer)
            conv_layer = Flatten()(conv_layer)
            
            # LSTM branch - capturing temporal dependencies
            lstm_layer = Bidirectional(LSTM(units=64, return_sequences=True))(inputs)
            lstm_layer = Dropout(self.dropout_rate)(lstm_layer)
            lstm_layer = Bidirectional(LSTM(units=64, return_sequences=False))(lstm_layer)
            lstm_layer = Dropout(self.dropout_rate)(lstm_layer)
            
            # GRU branch - another recurrent approach
            gru_layer = GRU(units=64, return_sequences=True)(inputs)
            gru_layer = Dropout(self.dropout_rate)(gru_layer)
            gru_layer = GRU(units=64, return_sequences=False)(gru_layer)
            gru_layer = Dropout(self.dropout_rate)(gru_layer)
            
            # Merge the branches
            merged = Concatenate()([conv_layer, lstm_layer, gru_layer])
            
            # Dense layers for final prediction
            dense = Dense(units=128, activation='relu')(merged)
            dense = BatchNormalization()(dense)
            dense = Dropout(self.dropout_rate)(dense)
            dense = Dense(units=64, activation='relu')(dense)
            dense = BatchNormalization()(dense)
            dense = Dropout(self.dropout_rate/2)(dense)
            
            # Output layer
            outputs = Dense(units=1)(dense)
            
            # Create and compile model
            model = Model(inputs=inputs, outputs=outputs)
        else:
            raise ValueError(f"Unknown base model type: {self.base_model_type}")
        
        # Compile the model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        
        return model
        
    def create_error_correction_model(self, input_shape, previous_error_shape=(10, 1)):
        """
        Create the error correction model that learns from previous prediction errors
        
        Args:
            input_shape (tuple): Shape of input features (window_size, n_features)
            previous_error_shape (tuple): Shape of previous errors input
            
        Returns:
            tf.keras.Model: Compiled error correction model
        """
        # Feature input branch
        feature_input = Input(shape=input_shape, name='feature_input')
        feature_lstm = LSTM(32, return_sequences=False)(feature_input)
        feature_dense = Dense(16, activation='relu')(feature_lstm)
        
        # Previous error input branch
        error_input = Input(shape=previous_error_shape, name='error_input')
        error_lstm = LSTM(16, return_sequences=False)(error_input)
        error_dense = Dense(8, activation='relu')(error_lstm)
        
        # Merge branches
        merged = Concatenate()([feature_dense, error_dense])
        
        # Dense layers
        dense = Dense(16, activation='relu')(merged)
        dense = Dropout(0.2)(dense)
        
        # Output (error correction factor)
        output = Dense(1)(dense)
        
        # Create model
        model = Model(inputs=[feature_input, error_input], outputs=output)
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate * 0.5)  # Lower learning rate for correction model
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        
        return model
    
    def train_base_model(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, callbacks=None):
        """
        Train the base prediction model
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training target values
            X_val (np.array): Validation features
            y_val (np.array): Validation target values
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            callbacks (list): List of Keras callbacks
            
        Returns:
            tuple: (model, history)
        """
        # Create model if not already created
        if self.base_model is None:
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.base_model = self.create_base_model(input_shape)
        
        # Define default callbacks if none provided
        if callbacks is None:
            # Check if any training values are negative to validate data range
            if np.any(y_train < 0):
                print("Warning: Training data contains negative target values. This may lead to negative predictions.")
                print(f"Range of target values: {np.min(y_train):.2f} to {np.max(y_train):.2f}")
                print("Consider using non-negative data or applying a transformation.")
            
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001),
                ModelCheckpoint(
                    filepath=os.path.join(self.model_dir, f'base_{self.base_model_type}_model.keras'),
                    monitor='val_loss',
                    save_best_only=True
                )
            ]
        
        # Print data statistics for debugging
        print(f"Training data statistics:")
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_train range: {np.min(X_train):.4f} to {np.max(X_train):.4f}")
        print(f"y_train range: {np.min(y_train):.4f} to {np.max(y_train):.4f}")
        
        # Train the model
        history = self.base_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate the model
        # Make predictions on validation data to check for negative values
        val_predictions = self.base_model.predict(X_val)
        if np.any(val_predictions < 0):
            print(f"Warning: {np.sum(val_predictions < 0)} negative predictions found after training.")
            print(f"Prediction range: {np.min(val_predictions):.4f} to {np.max(val_predictions):.4f}")
        
        # Save the model
        self.base_model.save(os.path.join(self.model_dir, f'base_{self.base_model_type}_model.keras'))
        
        return self.base_model, history
    
    def _prepare_error_history(self, X):
        """
        Prepare error history for the error correction model
        
        Args:
            X (np.array): Input features
            
        Returns:
            np.array: Array of previous prediction errors or zeros if not enough history
        """
        # Get prediction history from reward system
        history_df = self.reward_system.get_prediction_history()
        history_df = history_df.dropna(subset=['accuracy'])
        
        if len(history_df) < 10:
            # Not enough error history, use zeros
            print("Not enough error history, using zeros")
            return np.zeros((X.shape[0], 10, 1))
        
        # Use the most recent 10 errors
        recent_errors = history_df.tail(10)['accuracy'].values
        
        # Reshape to (batch_size, 10, 1) and repeat for each sample in the batch
        error_history = np.tile(recent_errors[-10:], (X.shape[0], 1))
        error_history = error_history.reshape((X.shape[0], 10, 1))
        
        return error_history
    
    def adapt_to_errors(self):
        """
        Adapt the model based on previous prediction errors
        
        Returns:
            bool: True if adaptation was performed, False otherwise
        """
        # Get predictions with actual values from the reward system
        predictions_df = self.reward_system.get_prediction_history()
        valid_predictions = predictions_df.dropna(subset=['actual_price'])
        
        # Check if we have enough data to adapt
        if len(valid_predictions) < self.min_adaptations_needed:
            print(f"Not enough data to adapt model. Need at least {self.min_adaptations_needed} samples with actual values.")
            return False
        
        # Check if we need to adapt based on frequency
        if len(valid_predictions) % self.adaptation_frequency != 0:
            return False
        
        print(f"Adapting model based on {len(valid_predictions)} historical predictions...")
        
        # Get the most recent predictions with actual values for training
        train_data = valid_predictions.tail(min(100, len(valid_predictions)))
        
        # Calculate relative prediction errors
        train_data['relative_error'] = (train_data['predicted_price'] - train_data['actual_price']) / train_data['actual_price']
        
        # If we don't have an error correction model yet, create one
        if self.error_correction_model is None:
            # Get input shapes from the last prediction
            if self.last_prediction['input_features'] is not None:
                input_shape = (self.last_prediction['input_features'].shape[1], 
                               self.last_prediction['input_features'].shape[2])
                self.error_correction_model = self.create_error_correction_model(input_shape)
            else:
                print("No prediction has been made yet, cannot initialize error correction model")
                return False
        
        # We need the original features for these predictions to retrain
        # This is a simplified approach - in a real system, you'd need to store and retrieve 
        # the original features that led to each prediction
        
        # For now, let's simulate this with random data just to demonstrate the concept
        # In a real implementation, you'd use the actual historical features
        n_samples = len(train_data)
        
        # Create simulated features
        # These would normally be retrieved from storage
        simulated_features = np.random.random((n_samples, self.window_size, 5))
        
        # Create error history data (10 previous errors for each prediction)
        error_history = np.zeros((n_samples, 10, 1))
        for i in range(n_samples):
            start_idx = max(0, i - 10)
            errors_to_use = train_data['relative_error'].iloc[start_idx:i].values
            # Pad with zeros if needed
            padded_errors = np.pad(errors_to_use, (10 - len(errors_to_use), 0), 'constant')
            error_history[i, :, 0] = padded_errors
        
        # The target is the correction factor (relative error)
        y_correction = train_data['relative_error'].values
        
        # Train the error correction model
        self.error_correction_model.fit(
            [simulated_features, error_history],
            y_correction,
            epochs=50,
            batch_size=16,
            verbose=1
        )
        
        # Save the updated error correction model
        self.error_correction_model.save(os.path.join(self.model_dir, 'error_correction_model.keras'))
        
        print("Error correction model adapted successfully!")
        return True
    
    def predict(self, X_new, apply_correction=True):
        """
        Make prediction with the model, optionally applying error correction
        
        Args:
            X_new (np.array): Input features
            apply_correction (bool): Whether to apply error correction
            
        Returns:
            float: Predicted price
        """
        # Ensure X_new is properly shaped
        if len(X_new.shape) == 2:
            X_new = X_new.reshape(1, X_new.shape[0], X_new.shape[1])
        
        # Ensure the data contains only numeric values
        # Convert any potential non-numeric values to numeric
        try:
            # Check for non-numeric values
            X_new = X_new.astype(np.float32)
        except (ValueError, TypeError) as e:
            print(f"Error converting input data to numeric: {e}")
            print("Attempting to fix data by removing non-numeric values...")
            
            # Create a mask for numeric values (replace non-numeric with NaN then 0)
            X_new_cleaned = np.zeros_like(X_new, dtype=np.float32)
            for i in range(X_new.shape[0]):
                for j in range(X_new.shape[1]):
                    for k in range(X_new.shape[2]):
                        try:
                            X_new_cleaned[i, j, k] = float(X_new[i, j, k])
                        except (ValueError, TypeError):
                            # Replace non-numeric values with 0
                            X_new_cleaned[i, j, k] = 0.0
            
            X_new = X_new_cleaned
            print("Data cleaned and converted to numeric values.")
        
        # Make sure we have a base model
        if self.base_model is None:
            print("Base model not initialized. Creating a new one...")
            input_shape = (X_new.shape[1], X_new.shape[2])
            self.base_model = self.create_base_model(input_shape)
            print("Warning: Using untrained model. Results may be unreliable.")
        
        # Check if input shape matches the model's expected input shape
        if hasattr(self, 'input_shape') and X_new.shape[2] != self.input_shape[1]:
            print(f"Warning: Input feature count ({X_new.shape[2]}) doesn't match model's expected feature count ({self.input_shape[1]})")
            # Adjust the input shape to match what the model expects
            if X_new.shape[2] > self.input_shape[1]:
                print("Truncating extra features")
                X_new = X_new[:, :, :self.input_shape[1]]
            else:
                print("Padding missing features with zeros")
                padding = np.zeros((X_new.shape[0], X_new.shape[1], self.input_shape[1] - X_new.shape[2]), dtype=np.float32)
                X_new = np.concatenate([X_new, padding], axis=2)
            print(f"Adjusted input shape: {X_new.shape}")
        
        # Get base prediction
        base_prediction = self.base_model.predict(X_new, verbose=0)
        predicted_price = base_prediction[0, 0]
        
        # Store input features for possible later adaptation
        self.last_prediction['input_features'] = X_new
        
        # Apply error correction if requested and available
        if apply_correction and self.error_correction_model is not None:
            try:
                # Prepare error history
                error_history = self._prepare_error_history(X_new)
                
                # Get correction factor
                correction_factor = self.error_correction_model.predict([X_new, error_history], verbose=0)[0, 0]
                
                # Apply correction (predicted_price * (1 - correction_factor))
                corrected_price = predicted_price * (1 - correction_factor)
                
                print(f"Base prediction: {predicted_price:.2f}")
                print(f"Correction factor: {correction_factor:.4f}")
                print(f"Corrected prediction: {corrected_price:.2f}")
                
                predicted_price = corrected_price
            except Exception as e:
                print(f"Error applying correction: {e}. Using base prediction.")
                # Continue with the base prediction
        elif apply_correction and self.error_correction_model is None:
            print("No error correction model available yet. Using base prediction only.")
        
        # Ensure prediction is non-negative
        if predicted_price < 0:
            print(f"Warning: Negative price prediction ({predicted_price:.2f}) corrected to minimum value.")
            predicted_price = 0.01  # Set a minimum positive value
        
        # Update last prediction details
        self.last_prediction['predicted_price'] = predicted_price
        current_date = datetime.now().strftime('%Y-%m-%d')
        self.last_prediction['date'] = current_date
        
        # Save prediction to reward system
        self.reward_system.save_prediction(
            date=current_date,
            predicted_price=predicted_price,
            model_version=f"reinforced_{self.base_model_type}"
        )
        
        return predicted_price
    
    def update_with_actual_price(self, date, actual_price):
        """
        Update the model with the actual price for a specific date
        
        Args:
            date (str or datetime): Date of the prediction
            actual_price (float): Actual stock price
            
        Returns:
            bool: True if update was successful
        """
        # Update the reward system
        success = self.reward_system.update_actual_price(date, actual_price)
        
        if success:
            # Get the accuracy
            accuracy, threshold_met = self.reward_system.get_prediction_accuracy(date)
            
            # Update last prediction if it matches this date
            if self.last_prediction['date'] == date or (
                isinstance(date, str) and self.last_prediction['date'] == date):
                self.last_prediction['actual_price'] = actual_price
                self.last_prediction['error'] = accuracy
                
                # Add to prediction memory
                self.prediction_memory.append({
                    'date': date,
                    'predicted_price': self.last_prediction['predicted_price'],
                    'actual_price': actual_price,
                    'error': accuracy,
                    'threshold_met': threshold_met
                })
                
                # Trim memory if needed
                if len(self.prediction_memory) > self.max_memory_size:
                    self.prediction_memory = self.prediction_memory[-self.max_memory_size:]
                
                # Try to adapt the model
                self.adapt_to_errors()
            
            print(f"Updated with actual price: {actual_price:.2f}, prediction accuracy: {accuracy:.4f}")
            
            # If the error is above threshold, consider more aggressive adaptation
            if accuracy is not None and accuracy > self.error_threshold:
                print(f"High prediction error ({accuracy:.4f}). Consider model retraining.")
                
            return True
        
        return False
    
    def get_recent_prediction_metrics(self):
        """
        Get metrics on recent predictions
        
        Returns:
            dict: Metrics on recent predictions
        """
        return self.reward_system.get_overall_accuracy()
    
    def send_predictions_to_api(self, future_prices, future_dates, last_close_price=None, df=None):
        """
        Send predictions to an API endpoint
        
        Args:
            future_prices (list): List of predicted prices for future dates
            future_dates (list): List of dates for the predictions
            last_close_price (float): Last known closing price
            df (pd.DataFrame): DataFrame with historical data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            import requests
            from datetime import datetime
            
            # API endpoint URL
            api_url = "https://stocks.wajipk.com/api/predictions"
            
            # Format dates to strings
            def format_date_safely(date_obj):
                if isinstance(date_obj, str):
                    return date_obj
                elif isinstance(date_obj, datetime) or isinstance(date_obj, pd.Timestamp):
                    return date_obj.strftime('%Y-%m-%d')
                else:
                    # Try to convert to string as fallback
                    return str(date_obj)
            
            date_strings = [format_date_safely(date) for date in future_dates]
            
            # Create payload - include 'date' field at the top level (required by API)
            current_date = datetime.now().strftime('%Y-%m-%d')
            payload = {
                "symbol": self.symbol,
                "date": current_date,  # Add date field at the top level (today's date)
                "predictions": [
                    {"date": date, "price": float(price)} 
                    for date, price in zip(date_strings, future_prices)
                ],
                "last_close_price": float(last_close_price) if last_close_price is not None else None,
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "model_type": f"reinforced_{self.base_model_type}"
            }
            
            # Add metadata if available
            if df is not None:
                try:
                    # Get the last row of data for metadata
                    last_row = df.iloc[-1]
                    metadata = {
                        "last_date": format_date_safely(last_row.get('date', None)),
                        "last_volume": float(last_row.get('volume', 0)),
                        "last_high": float(last_row.get('high', last_close_price)),
                        "last_low": float(last_row.get('low', last_close_price))
                    }
                    payload["metadata"] = metadata
                except Exception as e:
                    print(f"Warning: Error adding metadata to API payload: {e}")
            
            # Print payload for debugging
            print(f"Sending reinforced model predictions to API for {self.symbol}...")
            print(f"Debug - Full payload: {payload}")
            
            # Send to API
            response = requests.post(api_url, json=payload)
            
            if response.status_code >= 400:
                print(f"Error: API returned status code {response.status_code}")
                try:
                    error_details = response.json()
                    print(f"API error details: {error_details}")
                except Exception:
                    print(f"Raw API response: {response.text}")
                return False
                
            print(f"Successfully sent reinforced model predictions to API. Response: {response.status_code}")
            return True
            
        except Exception as e:
            print(f"Error sending reinforced model predictions to API: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict_future_prices(self, X_current, days_ahead=5, df=None, last_close_price=None):
        """
        Predict prices for multiple days ahead and optionally send to API
        
        Args:
            X_current (np.array): Current input window for prediction
            days_ahead (int): Number of days to predict ahead
            df (pd.DataFrame): Historical data frame (optional)
            last_close_price (float): Last known closing price (optional)
            
        Returns:
            tuple: (predictions, future_dates)
        """
        from datetime import datetime, timedelta
        
        # Make sure X_current is in the right shape
        if len(X_current.shape) == 2:
            X_current = X_current.reshape(1, X_current.shape[0], X_current.shape[1])
        
        # Generate future dates
        last_date = datetime.now()
        if df is not None and 'date' in df.columns and len(df) > 0:
            last_date_raw = df['date'].iloc[-1]
            
            if isinstance(last_date_raw, str):
                try:
                    last_date = datetime.strptime(last_date_raw, '%Y-%m-%d')
                except ValueError:
                    try:
                        last_date = datetime.strptime(last_date_raw, '%Y/%m/%d')
                    except ValueError:
                        print(f"Warning: Could not parse date format: {last_date_raw}, using current date")
            elif isinstance(last_date_raw, pd.Timestamp):
                last_date = last_date_raw.to_pydatetime()
        
        # Generate future dates (skipping weekends)
        future_dates = []
        current_date = last_date
        remaining_days = days_ahead
        
        while remaining_days > 0:
            current_date = current_date + timedelta(days=1)
            # Skip weekends (5 = Saturday, 6 = Sunday)
            if current_date.weekday() < 5:
                future_dates.append(current_date)
                remaining_days -= 1
        
        # Make predictions for each future day
        predictions = []
        current_input = X_current.copy()
        
        for i in range(days_ahead):
            # Make prediction for the current date
            price = self.predict(current_input, apply_correction=True)
            predictions.append(price)
            
            # Prepare input for next day by shifting window and adding prediction
            # This assumes the feature at index 0 is the price
            if i < days_ahead - 1:  # Only need to update for next prediction if not the last day
                # Shift the window
                current_input[0, :-1, :] = current_input[0, 1:, :]
                
                # Add prediction as the next day's input (assuming price is first feature)
                # For simplicity, we're just copying the last row and updating the price
                current_input[0, -1, :] = current_input[0, -2, :]
                current_input[0, -1, 0] = price  # Update price component
        
        # Send predictions to API
        self.send_predictions_to_api(predictions, future_dates, last_close_price, df)
        
        return predictions, future_dates
    
    def save_models(self):
        """
        Save all model components
        """
        if self.base_model is not None:
            self.base_model.save(os.path.join(self.model_dir, f'base_{self.base_model_type}_model.keras'))
            
        if self.error_correction_model is not None:
            self.error_correction_model.save(os.path.join(self.model_dir, 'error_correction_model.keras'))
            
        if self.combined_model is not None:
            self.combined_model.save(os.path.join(self.model_dir, 'combined_reinforced_model.keras')) 