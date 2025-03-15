import os
import numpy as np
import pandas as pd
import tensorflow as tf
import time
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, GRU, Conv1D
from tensorflow.keras.layers import MaxPooling1D, Flatten, Concatenate, BatchNormalization, Attention
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
import xgboost as xgb
import lightgbm as lgb
from pathlib import Path
import json
from datetime import datetime

# Model registry to store all available models
MODEL_REGISTRY = {
    # Deep Learning Models
    "lstm": "LSTM Network",
    "bilstm": "Bidirectional LSTM",
    "gru": "GRU Network",
    "cnn_lstm": "CNN-LSTM Hybrid",
    "advanced": "Advanced Multi-Branch Model",
    
    # Traditional ML Models
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "svr": "Support Vector Regression",
    "elastic_net": "Elastic Net Regression",
}


class ModelFactory:
    """Factory class to create different types of models"""
    
    @staticmethod
    def create_model(model_type, input_shape=None, learning_rate=0.001, dropout_rate=0.3, **kwargs):
        """
        Create and return a model based on the specified type
        
        Args:
            model_type (str): Type of model to create
            input_shape (tuple): Shape of input data (required for deep learning models)
            learning_rate (float): Learning rate for model training
            dropout_rate (float): Dropout rate for regularization
            **kwargs: Additional model-specific parameters
            
        Returns:
            object: Created model
        """
        if model_type == "lstm":
            return ModelFactory._create_lstm_model(input_shape, dropout_rate)
        elif model_type == "bilstm":
            return ModelFactory._create_bilstm_model(input_shape, dropout_rate)
        elif model_type == "gru":
            return ModelFactory._create_gru_model(input_shape, dropout_rate)
        elif model_type == "cnn_lstm":
            return ModelFactory._create_cnn_lstm_model(input_shape, dropout_rate)
        elif model_type == "advanced":
            return ModelFactory._create_advanced_model(input_shape, dropout_rate, learning_rate)
        elif model_type == "random_forest":
            return ModelFactory._create_random_forest(**kwargs)
        elif model_type == "gradient_boosting":
            return ModelFactory._create_gradient_boosting(**kwargs)
        elif model_type == "xgboost":
            return ModelFactory._create_xgboost(**kwargs)
        elif model_type == "lightgbm":
            return ModelFactory._create_lightgbm(**kwargs)
        elif model_type == "svr":
            return ModelFactory._create_svr(**kwargs)
        elif model_type == "elastic_net":
            return ModelFactory._create_elastic_net(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def _create_lstm_model(input_shape, dropout_rate=0.2):
        """Create a simple LSTM model"""
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=input_shape),
            Dropout(dropout_rate),
            LSTM(units=50, return_sequences=False),
            Dropout(dropout_rate),
            Dense(units=25),
            Dense(units=1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    @staticmethod
    def _create_bilstm_model(input_shape, dropout_rate=0.2):
        """Create a Bidirectional LSTM model"""
        model = Sequential([
            Bidirectional(LSTM(units=50, return_sequences=True), input_shape=input_shape),
            Dropout(dropout_rate),
            Bidirectional(LSTM(units=50, return_sequences=False)),
            Dropout(dropout_rate),
            Dense(units=25),
            Dense(units=1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    @staticmethod
    def _create_gru_model(input_shape, dropout_rate=0.2):
        """Create a GRU model"""
        model = Sequential([
            GRU(units=50, return_sequences=True, input_shape=input_shape),
            Dropout(dropout_rate),
            GRU(units=50, return_sequences=False),
            Dropout(dropout_rate),
            Dense(units=25),
            Dense(units=1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    @staticmethod
    def _create_cnn_lstm_model(input_shape, dropout_rate=0.2):
        """Create a CNN-LSTM hybrid model"""
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Dropout(dropout_rate),
            LSTM(units=50, return_sequences=False),
            Dropout(dropout_rate),
            Dense(units=25),
            Dense(units=1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    @staticmethod
    def _create_advanced_model(input_shape, dropout_rate=0.3, learning_rate=0.001):
        """Create an advanced multi-branch deep learning model"""
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
        lstm_layer = Dropout(dropout_rate)(lstm_layer)
        lstm_layer = Bidirectional(LSTM(units=64, return_sequences=False))(lstm_layer)
        lstm_layer = Dropout(dropout_rate)(lstm_layer)
        
        # GRU branch - another recurrent approach
        gru_layer = GRU(units=64, return_sequences=True)(inputs)
        gru_layer = Dropout(dropout_rate)(gru_layer)
        gru_layer = GRU(units=64, return_sequences=False)(gru_layer)
        gru_layer = Dropout(dropout_rate)(gru_layer)
        
        # Merge the branches
        merged = Concatenate()([conv_layer, lstm_layer, gru_layer])
        
        # Dense layers for final prediction
        dense = Dense(units=128, activation='relu')(merged)
        dense = BatchNormalization()(dense)
        dense = Dropout(dropout_rate)(dense)
        dense = Dense(units=64, activation='relu')(dense)
        dense = BatchNormalization()(dense)
        dense = Dropout(dropout_rate/2)(dense)
        
        # Output layer
        outputs = Dense(units=1)(dense)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs)
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        
        return model
    
    @staticmethod
    def _create_random_forest(n_estimators=100, max_depth=10, **kwargs):
        """Create a Random Forest regressor"""
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            **{k: v for k, v in kwargs.items() if k in RandomForestRegressor().get_params()}
        )
    
    @staticmethod
    def _create_gradient_boosting(n_estimators=100, max_depth=5, learning_rate=0.1, **kwargs):
        """Create a Gradient Boosting regressor"""
        return GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            **{k: v for k, v in kwargs.items() if k in GradientBoostingRegressor().get_params()}
        )
    
    @staticmethod
    def _create_xgboost(n_estimators=100, max_depth=5, learning_rate=0.1, **kwargs):
        """Create an XGBoost regressor"""
        return xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            **{k: v for k, v in kwargs.items() if k in xgb.XGBRegressor().get_params()}
        )
    
    @staticmethod
    def _create_lightgbm(n_estimators=100, max_depth=5, learning_rate=0.1, **kwargs):
        """Create a LightGBM regressor"""
        return lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            **{k: v for k, v in kwargs.items() if k in lgb.LGBMRegressor().get_params()}
        )
    
    @staticmethod
    def _create_svr(C=1.0, epsilon=0.1, **kwargs):
        """Create a Support Vector Regression model"""
        return SVR(
            C=C,
            epsilon=epsilon,
            **{k: v for k, v in kwargs.items() if k in SVR().get_params()}
        )
    
    @staticmethod
    def _create_elastic_net(alpha=1.0, l1_ratio=0.5, **kwargs):
        """Create an Elastic Net regressor"""
        return ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            random_state=42,
            **{k: v for k, v in kwargs.items() if k in ElasticNet().get_params()}
        )


class ModelEvaluator:
    """Class to evaluate and compare different models"""
    
    @staticmethod
    def evaluate_model(model, X_test, y_test, is_dl_model=True):
        """
        Evaluate model performance
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            is_dl_model (bool): Whether the model is a deep learning model
            
        Returns:
            dict: Performance metrics
        """
        # Make predictions
        if is_dl_model:
            y_pred = model.predict(X_test)
        else:
            # For non-deep learning models, reshape input if needed
            if len(X_test.shape) > 2:
                X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
                y_pred = model.predict(X_test_reshaped)
            else:
                y_pred = model.predict(X_test)
        
        # Ensure y_test and y_pred have the same shape
        if len(y_test.shape) != len(y_pred.shape):
            if len(y_test.shape) > len(y_pred.shape):
                y_pred = y_pred.reshape(y_test.shape)
            else:
                y_test = y_test.reshape(y_pred.shape)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mask = y_test != 0
        y_test_safe = y_test[mask]
        y_pred_safe = y_pred[mask]
        
        if len(y_test_safe) > 0:
            mape = np.mean(np.abs((y_test_safe - y_pred_safe) / y_test_safe)) * 100
        else:
            mape = 0.0
        
        # Calculate R²
        r2 = r2_score(y_test, y_pred)
        
        # Return all metrics in a dictionary
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2
        }


class ModelSelector:
    """Class to select the best model for a given stock symbol"""
    
    def __init__(self, symbol, model_dir='models'):
        """
        Initialize ModelSelector
        
        Args:
            symbol (str): Stock symbol
            model_dir (str): Directory to save model files
        """
        self.symbol = symbol
        self.model_dir = model_dir
        self.company_model_dir = os.path.join(model_dir, symbol)
        self.model_results_file = os.path.join(self.company_model_dir, "model_comparison.json")
        self.best_model_info_file = os.path.join(self.company_model_dir, "best_model_info.json")
        
        # Create directory if it doesn't exist
        os.makedirs(self.company_model_dir, exist_ok=True)
    
    def train_and_evaluate_models(self, X_train, y_train, X_test, y_test, 
                                  models_to_try=None, window_size=10, 
                                  epochs=100, batch_size=32, learning_rate=0.001, 
                                  dropout_rate=0.3, callbacks=None):
        """
        Train and evaluate multiple models
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            models_to_try (list): List of model types to try (if None, all models are tried)
            window_size (int): Window size for sequences
            epochs (int): Number of training epochs for deep learning models
            batch_size (int): Batch size for training deep learning models
            learning_rate (float): Learning rate for model training
            dropout_rate (float): Dropout rate for regularization
            callbacks (list): Callbacks for deep learning models
            
        Returns:
            dict: Results of all models
        """
        # If no models specified, try all
        if models_to_try is None:
            models_to_try = list(MODEL_REGISTRY.keys())
        
        # Prepare non-DL data format (flattened)
        X_train_flat = X_train.reshape(X_train.shape[0], -1) if len(X_train.shape) > 2 else X_train
        X_test_flat = X_test.reshape(X_test.shape[0], -1) if len(X_test.shape) > 2 else X_test
        
        # Dictionary to store results
        results = {}
        
        print(f"Training and evaluating {len(models_to_try)} models for {self.symbol}...")
        
        # Train and evaluate each model
        for model_type in models_to_try:
            try:
                print(f"\nTraining {MODEL_REGISTRY[model_type]} model...")
                
                # Check if model is deep learning or traditional ML
                is_dl_model = model_type in ["lstm", "bilstm", "gru", "cnn_lstm", "advanced"]
                
                # Create and train model
                if is_dl_model:
                    # Create deep learning model
                    model = ModelFactory.create_model(
                        model_type=model_type,
                        input_shape=(X_train.shape[1], X_train.shape[2]),
                        learning_rate=learning_rate,
                        dropout_rate=dropout_rate
                    )
                    
                    # Train deep learning model
                    start_time = time.time()
                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=callbacks,
                        verbose=1
                    )
                    training_time = time.time() - start_time
                    
                    # Save model
                    model_path = os.path.join(self.company_model_dir, f"{model_type}_model.keras")
                    model.save(model_path)
                    
                    # Get training history metrics
                    val_loss = history.history['val_loss'][-1]
                    train_loss = history.history['loss'][-1]
                    
                else:
                    # Create traditional ML model
                    model = ModelFactory.create_model(model_type=model_type)
                    
                    # Train traditional ML model
                    start_time = time.time()
                    model.fit(X_train_flat, y_train)
                    training_time = time.time() - start_time
                    
                    # Save model
                    model_path = os.path.join(self.company_model_dir, f"{model_type}_model.pkl")
                    joblib.dump(model, model_path)
                    
                    # No training history for traditional ML models
                    val_loss = None
                    train_loss = None
                
                # Evaluate model
                eval_metrics = ModelEvaluator.evaluate_model(
                    model=model,
                    X_test=X_test if is_dl_model else X_test_flat,
                    y_test=y_test,
                    is_dl_model=is_dl_model
                )
                
                # Add results to dictionary
                results[model_type] = {
                    'name': MODEL_REGISTRY[model_type],
                    'metrics': eval_metrics,
                    'training_time': training_time,
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                    'is_dl_model': is_dl_model,
                    'model_path': model_path
                }
                
                print(f"  {MODEL_REGISTRY[model_type]} evaluation:")
                print(f"    MSE: {eval_metrics['mse']:.4f}")
                print(f"    RMSE: {eval_metrics['rmse']:.4f}")
                print(f"    MAE: {eval_metrics['mae']:.4f}")
                print(f"    MAPE: {eval_metrics['mape']:.2f}%")
                print(f"    R²: {eval_metrics['r2']:.4f}")
                print(f"    Training time: {training_time:.2f} seconds")
                
            except Exception as e:
                print(f"Error training {model_type} model: {e}")
                import traceback
                traceback.print_exc()
        
        # Save results to file
        with open(self.model_results_file, 'w') as f:
            # Convert numpy values to native Python types for JSON serialization
            serializable_results = {}
            for model_type, model_data in results.items():
                serializable_metrics = {}
                for metric, value in model_data['metrics'].items():
                    serializable_metrics[metric] = float(value)
                
                serializable_results[model_type] = {
                    **model_data,
                    'metrics': serializable_metrics,
                    'val_loss': float(model_data['val_loss']) if model_data['val_loss'] is not None else None,
                    'train_loss': float(model_data['train_loss']) if model_data['train_loss'] is not None else None,
                    'training_time': float(model_data['training_time'])
                }
            
            json.dump(serializable_results, f, indent=4)
        
        return results
    
    def select_best_model(self, results=None, priority_metric='mape'):
        """
        Select the best model based on evaluation metrics
        
        Args:
            results (dict): Results from train_and_evaluate_models
            priority_metric (str): Metric to prioritize ('mse', 'rmse', 'mae', 'mape', 'r2')
            
        Returns:
            tuple: (best_model_type, best_model_metrics)
        """
        # Load results from file if not provided
        if results is None:
            if os.path.exists(self.model_results_file):
                with open(self.model_results_file, 'r') as f:
                    results = json.load(f)
            else:
                raise FileNotFoundError(f"Model comparison file not found: {self.model_results_file}")
        
        # No results available
        if not results:
            return None, None
        
        # Select best model based on priority metric
        best_model_type = None
        best_metric_value = float('inf')  # Lower is better for MSE, RMSE, MAE, MAPE
        
        # For R², higher is better
        if priority_metric == 'r2':
            best_metric_value = float('-inf')
        
        for model_type, model_data in results.items():
            metric_value = model_data['metrics'][priority_metric]
            
            if priority_metric == 'r2':
                # For R², higher is better
                if metric_value > best_metric_value:
                    best_metric_value = metric_value
                    best_model_type = model_type
            else:
                # For error metrics, lower is better
                if metric_value < best_metric_value:
                    best_metric_value = metric_value
                    best_model_type = model_type
        
        # Save best model info
        best_model_info = {
            'best_model_type': best_model_type,
            'model_name': MODEL_REGISTRY[best_model_type],
            'priority_metric': priority_metric,
            'metric_value': float(best_metric_value),
            'model_path': results[best_model_type]['model_path'],
            'is_dl_model': results[best_model_type]['is_dl_model'],
            'selection_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(self.best_model_info_file, 'w') as f:
            json.dump(best_model_info, f, indent=4)
        
        return best_model_type, results[best_model_type]
    
    def load_best_model(self):
        """
        Load the best model for this symbol
        
        Returns:
            tuple: (model, model_info)
        """
        # Check if best model info exists
        if not os.path.exists(self.best_model_info_file):
            raise FileNotFoundError(f"Best model info file not found: {self.best_model_info_file}")
        
        # Load best model info
        with open(self.best_model_info_file, 'r') as f:
            best_model_info = json.load(f)
        
        # Load the model
        model_path = best_model_info['model_path']
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if best_model_info['is_dl_model']:
            model = tf.keras.models.load_model(model_path)
        else:
            model = joblib.load(model_path)
        
        return model, best_model_info
    
    def visualize_model_comparison(self, results=None):
        """
        Visualize the comparison of different models
        
        Args:
            results (dict): Results from train_and_evaluate_models
        """
        # Load results from file if not provided
        if results is None:
            if os.path.exists(self.model_results_file):
                with open(self.model_results_file, 'r') as f:
                    results = json.load(f)
            else:
                raise FileNotFoundError(f"Model comparison file not found: {self.model_results_file}")
        
        # Extract model names and metrics
        model_names = [results[model_type]['name'] for model_type in results]
        mse_values = [results[model_type]['metrics']['mse'] for model_type in results]
        rmse_values = [results[model_type]['metrics']['rmse'] for model_type in results]
        mae_values = [results[model_type]['metrics']['mae'] for model_type in results]
        mape_values = [results[model_type]['metrics']['mape'] for model_type in results]
        r2_values = [results[model_type]['metrics']['r2'] for model_type in results]
        
        # Create figure with multiple subplots
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))
        
        # Plot MSE
        axs[0, 0].bar(model_names, mse_values)
        axs[0, 0].set_title('Mean Squared Error (MSE)')
        axs[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
        
        # Plot RMSE
        axs[0, 1].bar(model_names, rmse_values)
        axs[0, 1].set_title('Root Mean Squared Error (RMSE)')
        axs[0, 1].set_xticklabels(model_names, rotation=45, ha='right')
        
        # Plot MAE
        axs[1, 0].bar(model_names, mae_values)
        axs[1, 0].set_title('Mean Absolute Error (MAE)')
        axs[1, 0].set_xticklabels(model_names, rotation=45, ha='right')
        
        # Plot MAPE
        axs[1, 1].bar(model_names, mape_values)
        axs[1, 1].set_title('Mean Absolute Percentage Error (MAPE)')
        axs[1, 1].set_xticklabels(model_names, rotation=45, ha='right')
        
        # Plot R²
        axs[2, 0].bar(model_names, r2_values)
        axs[2, 0].set_title('R² Score')
        axs[2, 0].set_xticklabels(model_names, rotation=45, ha='right')
        
        # Plot training time
        training_times = [results[model_type]['training_time'] for model_type in results]
        axs[2, 1].bar(model_names, training_times)
        axs[2, 1].set_title('Training Time (seconds)')
        axs[2, 1].set_xticklabels(model_names, rotation=45, ha='right')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(self.company_model_dir, f"{self.symbol}_model_comparison.png"))
        plt.close()

    def try_models_sequentially(self, X_train, y_train, X_test, y_test, 
                               models_to_try=None, accuracy_threshold=0.75,
                               window_size=10, epochs=100, batch_size=32, 
                               learning_rate=0.001, dropout_rate=0.3, 
                               priority_metric='mape', callbacks=None,
                               reward_system=None):
        """
        Train and evaluate models one by one until finding one with acceptable accuracy
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            models_to_try (list): Ordered list of model types to try
            accuracy_threshold (float): Acceptable accuracy threshold (0.0-1.0)
            window_size (int): Window size for sequences
            epochs (int): Number of training epochs for deep learning models
            batch_size (int): Batch size for training deep learning models
            learning_rate (float): Learning rate for model training
            dropout_rate (float): Dropout rate for regularization
            priority_metric (str): Metric to use for model evaluation
            callbacks (list): Callbacks for deep learning models
            reward_system: Optional reward system to check historical accuracy
            
        Returns:
            tuple: (best_model_type, best_model_data, results)
        """
        # If no models specified, use default order
        if models_to_try is None:
            # Start with deep learning models, then traditional ML models
            models_to_try = ['lstm', 'bilstm', 'gru', 'cnn_lstm', 'advanced', 
                            'random_forest', 'gradient_boosting', 'xgboost', 
                            'lightgbm', 'svr', 'elastic_net']
        
        # Prepare non-DL data format (flattened)
        X_train_flat = X_train.reshape(X_train.shape[0], -1) if len(X_train.shape) > 2 else X_train
        X_test_flat = X_test.reshape(X_test.shape[0], -1) if len(X_test.shape) > 2 else X_test
        
        # Dictionary to store results of all models
        results = {}
        
        # Variables to track best model so far
        best_model_type = None
        best_accuracy = 0.0
        best_is_acceptable = False
        
        print(f"Sequential model evaluation for {self.symbol}...")
        print(f"Trying models in sequence until finding one with {accuracy_threshold*100}% accuracy")
        
        # Try each model in sequence
        for model_type in models_to_try:
            try:
                print(f"\nTrying {MODEL_REGISTRY[model_type]} model...")
                
                # Determine if this is a deep learning model
                is_dl_model = model_type in ['lstm', 'bilstm', 'gru', 'cnn_lstm', 'advanced']
                
                if is_dl_model:
                    # Create DL model
                    input_shape = (X_train.shape[1], X_train.shape[2])
                    model = ModelFactory.create_model(
                        model_type=model_type, 
                        input_shape=input_shape,
                        learning_rate=learning_rate,
                        dropout_rate=dropout_rate
                    )
                    
                    # Set up model checkpoint callback
                    checkpoint_path = os.path.join(self.company_model_dir, f"{model_type}_checkpoint.keras")
                    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                        filepath=checkpoint_path,
                        save_best_only=True,
                        monitor='val_loss',
                        mode='min',
                        verbose=1
                    )
                    
                    # Add checkpoint to callbacks
                    model_callbacks = [checkpoint_callback]
                    if callbacks:
                        model_callbacks.extend(callbacks)
                    
                    # Train deep learning model
                    start_time = time.time()
                    history = model.fit(
                        X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.2,
                        callbacks=model_callbacks,
                        verbose=1
                    )
                    training_time = time.time() - start_time
                    
                    # Load best model from checkpoint
                    if os.path.exists(checkpoint_path):
                        model = tf.keras.models.load_model(checkpoint_path)
                    
                    # Save model to final location
                    model_path = os.path.join(self.company_model_dir, f"{model_type}_model.keras")
                    model.save(model_path)
                    
                    # Extract training metrics
                    val_loss = min(history.history['val_loss'])
                    train_loss = history.history['loss'][-1]
                    
                else:
                    # Create traditional ML model
                    model = ModelFactory.create_model(model_type=model_type)
                    
                    # Train traditional ML model
                    start_time = time.time()
                    model.fit(X_train_flat, y_train)
                    training_time = time.time() - start_time
                    
                    # Save model
                    model_path = os.path.join(self.company_model_dir, f"{model_type}_model.pkl")
                    joblib.dump(model, model_path)
                    
                    # No training history for traditional ML models
                    val_loss = None
                    train_loss = None
                
                # Evaluate model
                eval_metrics = ModelEvaluator.evaluate_model(
                    model=model,
                    X_test=X_test if is_dl_model else X_test_flat,
                    y_test=y_test,
                    is_dl_model=is_dl_model
                )
                
                # Add results to dictionary
                results[model_type] = {
                    'name': MODEL_REGISTRY[model_type],
                    'metrics': eval_metrics,
                    'training_time': training_time,
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                    'is_dl_model': is_dl_model,
                    'model_path': model_path
                }
                
                # Print evaluation results
                print(f"  {MODEL_REGISTRY[model_type]} evaluation:")
                print(f"    MSE: {eval_metrics['mse']:.4f}")
                print(f"    RMSE: {eval_metrics['rmse']:.4f}")
                print(f"    MAE: {eval_metrics['mae']:.4f}")
                print(f"    MAPE: {eval_metrics['mape']:.2f}%")
                print(f"    R²: {eval_metrics['r2']:.4f}")
                print(f"    Training time: {training_time:.2f} seconds")
                
                # Convert to accuracy score (100% - MAPE) for easier interpretation
                if priority_metric == 'mape':
                    accuracy = 1.0 - (eval_metrics['mape'] / 100.0)
                    print(f"    Accuracy (100% - MAPE): {accuracy*100:.2f}%")
                else:
                    # For other metrics, use a normalized approach
                    if priority_metric == 'r2':
                        # For R², higher is better, and values can be >1
                        accuracy = min(1.0, max(0.0, (eval_metrics['r2'] + 1) / 2))
                    else:
                        # For error metrics like MSE/RMSE/MAE, use a simple threshold based approach
                        normalized_error = min(1.0, eval_metrics[priority_metric] / (y_test.mean() * 2))
                        accuracy = 1.0 - normalized_error
                    
                    print(f"    Estimated accuracy based on {priority_metric}: {accuracy*100:.2f}%")
                
                # Check historical prediction accuracy if reward system provided
                historical_accuracy = None
                if reward_system:
                    metrics = reward_system.get_overall_accuracy()
                    if metrics['count'] > 0 and metrics['mean_accuracy'] is not None:
                        historical_accuracy = 1.0 - metrics['mean_accuracy']
                        print(f"    Historical prediction accuracy: {historical_accuracy*100:.2f}%")
                
                # Determine if this model is acceptable
                model_accuracy = historical_accuracy if historical_accuracy is not None else accuracy
                is_acceptable = model_accuracy >= accuracy_threshold
                
                # Update best model if this is better
                if best_model_type is None or model_accuracy > best_accuracy:
                    best_model_type = model_type
                    best_accuracy = model_accuracy
                    best_is_acceptable = is_acceptable
                
                # Save results so far to file
                with open(self.model_results_file, 'w') as f:
                    json.dump(results, f, indent=4, default=str)
                
                # If model is acceptable, stop trying
                if is_acceptable:
                    print(f"\n✓ {MODEL_REGISTRY[model_type]} achieved acceptable accuracy: {model_accuracy*100:.2f}%")
                    print(f"Stopping sequential model evaluation early")
                    break
                else:
                    print(f"✗ {MODEL_REGISTRY[model_type]} did not meet accuracy threshold: {model_accuracy*100:.2f}% < {accuracy_threshold*100:.2f}%")
                    print(f"Trying next model...")
                
            except Exception as e:
                print(f"Error training/evaluating {model_type}: {e}")
                import traceback
                traceback.print_exc()
                # Continue with next model
        
        # If we've tried all models, select best one
        if not best_is_acceptable and len(results) > 0:
            print(f"\nNo model met the accuracy threshold of {accuracy_threshold*100:.2f}%")
            print(f"Using best model found: {MODEL_REGISTRY[best_model_type]} with accuracy {best_accuracy*100:.2f}%")
        
        # Save best model info
        if best_model_type is not None:
            best_model_info = {
                'best_model_type': best_model_type,
                'model_name': MODEL_REGISTRY[best_model_type],
                'priority_metric': priority_metric,
                'metric_value': float(results[best_model_type]['metrics'][priority_metric]),
                'accuracy': float(best_accuracy),
                'model_path': results[best_model_type]['model_path'],
                'is_dl_model': results[best_model_type]['is_dl_model'],
                'selection_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(self.best_model_info_file, 'w') as f:
                json.dump(best_model_info, f, indent=4)
        
        return best_model_type, results.get(best_model_type), results

# Import from this module in new versions of train_model.py 