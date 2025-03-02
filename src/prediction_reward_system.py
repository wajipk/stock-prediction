import os
import pandas as pd
from pandas import Timestamp
import numpy as np
from datetime import datetime
import warnings

# Suppress specific FutureWarning about DataFrame concatenation
warnings.filterwarnings('ignore', category=FutureWarning, 
                      message='.*The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*')

class PredictionRewardSystem:
    """
    A reward system for stock predictions that tracks prediction accuracy over time
    and helps the model learn from past predictions.
    """
    
    def __init__(self, symbol, threshold=0.05, models_dir='models'):
        """
        Initialize the prediction reward system
        
        Args:
            symbol (str): Stock symbol
            threshold (float): Accuracy threshold for determining if a prediction was successful
            models_dir (str): Base directory for model and prediction storage
        """
        self.symbol = symbol
        self.threshold = threshold
        
        # Create model directory path
        self.model_dir = os.path.join(models_dir, symbol)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Set predictions file path in the model directory
        self.predictions_file = os.path.join(self.model_dir, "predictions.csv")
        
        # Load existing predictions if the file exists
        self.predictions_df = self._load_predictions()
        
    def _load_predictions(self):
        """
        Load predictions history from CSV file
        
        Returns:
            pd.DataFrame: DataFrame with predictions or empty DataFrame if none exists
        """
        if os.path.exists(self.predictions_file):
            return pd.read_csv(self.predictions_file)
        else:
            # Create empty DataFrame with appropriate columns
            return pd.DataFrame(columns=[
                'date', 'predicted_price', 'actual_price', 'accuracy', 
                'threshold_met', 'model_version', 'timestamp', 'notes'
            ])
    
    def save_prediction(self, date, predicted_price, model_version='default', overwrite=False):
        """
        Save a new prediction to the predictions history, or overwrite an existing one
        
        Args:
            date (str or datetime or pd.Timestamp): Date for the prediction (in format 'YYYY-MM-DD')
            predicted_price (float): Predicted stock price
            model_version (str): Version of the model used for prediction
            overwrite (bool): Whether to overwrite an existing prediction for this date
            
        Returns:
            bool: True if the prediction was saved, False otherwise
        """
        # Convert date to string format depending on the type
        if isinstance(date, datetime):
            date_str = date.strftime('%Y-%m-%d')
        elif isinstance(date, pd.Timestamp):
            date_str = date.strftime('%Y-%m-%d')
        elif isinstance(date, str):
            date_str = date
        else:
            # Try to convert to string as a fallback
            try:
                date_str = str(date)
                # Check if it's in a reasonable date format
                if not (len(date_str) >= 8 and '-' in date_str or '/' in date_str):
                    print(f"Warning: Date format unusual: {date_str}, attempting to use as is")
            except:
                print(f"Error: Could not convert date {date} to string")
                return False
            
        # Check if this date already has a prediction
        if date_str in self.predictions_df['date'].values:
            if not overwrite:
                print(f"Warning: A prediction for {date_str} already exists. Skipping.")
                return False
            else:
                # Find the existing prediction and overwrite it
                idx = self.predictions_df[self.predictions_df['date'] == date_str].index[0]
                old_prediction = self.predictions_df.loc[idx, 'predicted_price']
                
                # Store old values for logging
                self.predictions_df.loc[idx, 'predicted_price'] = predicted_price
                self.predictions_df.loc[idx, 'timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.predictions_df.loc[idx, 'model_version'] = model_version
                
                # Add note about overwrite
                if 'notes' not in self.predictions_df.columns:
                    self.predictions_df['notes'] = ""
                self.predictions_df.loc[idx, 'notes'] = f"Manually overwritten. Previous value: {old_prediction}"
                
                # Recalculate accuracy if actual price exists
                if pd.notna(self.predictions_df.loc[idx, 'actual_price']):
                    actual_price = self.predictions_df.loc[idx, 'actual_price']
                    accuracy = abs(predicted_price - actual_price) / actual_price
                    self.predictions_df.loc[idx, 'accuracy'] = accuracy
                    self.predictions_df.loc[idx, 'threshold_met'] = accuracy <= self.threshold
                
                print(f"Prediction for {date_str} overwritten: {old_prediction} -> {predicted_price}")
        else:
            # Create a new prediction entry
            new_prediction = {
                'date': date_str,
                'predicted_price': predicted_price,
                'actual_price': None,  # Will be filled when available
                'accuracy': None,      # Will be calculated when actual price is available
                'threshold_met': None, # Will be determined when actual price is available
                'model_version': model_version,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'notes': ""
            }
            
            # Add to DataFrame
            new_prediction_df = pd.DataFrame([new_prediction])
            # Use the more modern pd.concat approach with explicit dtype preservation
            self.predictions_df = pd.concat(
                [self.predictions_df, new_prediction_df], 
                ignore_index=True,
                axis=0
            )
            
            print(f"New prediction saved for {date_str}: {predicted_price}")
        
        # Save to CSV
        self.predictions_df.to_csv(self.predictions_file, index=False)
        
        return True
    
    def update_actual_price(self, date, actual_price):
        """
        Update a prediction with the actual price and calculate accuracy
        
        Args:
            date (str or datetime or pd.Timestamp): Date for the prediction (in format 'YYYY-MM-DD')
            actual_price (float): Actual stock price
            
        Returns:
            bool: True if the prediction was updated, False if no prediction exists for this date
        """
        # Convert date to string format depending on the type
        if isinstance(date, datetime):
            date_str = date.strftime('%Y-%m-%d')
        elif isinstance(date, pd.Timestamp):
            date_str = date.strftime('%Y-%m-%d')
        elif isinstance(date, str):
            date_str = date
        else:
            # Try to convert to string as a fallback
            try:
                date_str = str(date)
                # Check if it's in a reasonable date format
                if not (len(date_str) >= 8 and '-' in date_str or '/' in date_str):
                    print(f"Warning: Date format unusual: {date_str}, attempting to use as is")
            except:
                print(f"Error: Could not convert date {date} to string")
                return False
            
        # Check if this date has a prediction
        if date_str not in self.predictions_df['date'].values:
            print(f"Warning: No prediction found for {date_str}. Skipping.")
            return False
            
        # Find the prediction
        idx = self.predictions_df[self.predictions_df['date'] == date_str].index[0]
        
        # Calculate accuracy (as percentage error)
        predicted_price = self.predictions_df.loc[idx, 'predicted_price']
        accuracy = abs(predicted_price - actual_price) / actual_price
        
        # Update the prediction
        self.predictions_df.loc[idx, 'actual_price'] = actual_price
        self.predictions_df.loc[idx, 'accuracy'] = accuracy
        self.predictions_df.loc[idx, 'threshold_met'] = accuracy <= self.threshold
        
        # Save to CSV
        self.predictions_df.to_csv(self.predictions_file, index=False)
        
        return True
    
    def get_prediction_accuracy(self, date):
        """
        Get the accuracy of a prediction for a specific date
        
        Args:
            date (str or datetime or pd.Timestamp): Date for the prediction (in format 'YYYY-MM-DD')
            
        Returns:
            tuple: (accuracy, threshold_met) or (None, None) if no prediction or actual price exists
        """
        # Convert date to string format depending on the type
        if isinstance(date, datetime):
            date_str = date.strftime('%Y-%m-%d')
        elif isinstance(date, pd.Timestamp):
            date_str = date.strftime('%Y-%m-%d')
        elif isinstance(date, str):
            date_str = date
        else:
            # Try to convert to string as a fallback
            try:
                date_str = str(date)
                # Check if it's in a reasonable date format
                if not (len(date_str) >= 8 and '-' in date_str or '/' in date_str):
                    print(f"Warning: Date format unusual: {date_str}, attempting to use as is")
            except:
                print(f"Error: Could not convert date {date} to string")
                return None, None
            
        # Check if this date has a prediction with an actual price
        if date_str not in self.predictions_df['date'].values:
            return None, None
            
        # Find the prediction
        prediction = self.predictions_df[self.predictions_df['date'] == date_str].iloc[0]
        
        if pd.isna(prediction['actual_price']):
            return None, None
            
        return prediction['accuracy'], prediction['threshold_met']
    
    def get_overall_accuracy(self):
        """
        Get overall prediction accuracy metrics
        
        Returns:
            dict: Dictionary with accuracy metrics
        """
        # Filter to only predictions with actual prices
        valid_predictions = self.predictions_df.dropna(subset=['actual_price'])
        
        if len(valid_predictions) == 0:
            return {
                'count': 0,
                'mean_accuracy': None,
                'threshold_met_count': 0,
                'threshold_met_pct': None
            }
            
        mean_accuracy = valid_predictions['accuracy'].mean()
        threshold_met_count = valid_predictions['threshold_met'].sum()
        threshold_met_pct = threshold_met_count / len(valid_predictions)
        
        return {
            'count': len(valid_predictions),
            'mean_accuracy': mean_accuracy,
            'threshold_met_count': threshold_met_count,
            'threshold_met_pct': threshold_met_pct
        }
    
    def get_prediction_history(self, last_n=None):
        """
        Get prediction history
        
        Args:
            last_n (int, optional): Get only the last N predictions. If None, returns all.
            
        Returns:
            pd.DataFrame: DataFrame with prediction history
        """
        if last_n is not None and last_n > 0:
            return self.predictions_df.tail(last_n)
        return self.predictions_df
    
    def get_recent_accuracy_trend(self, window=10):
        """
        Calculate the trend in prediction accuracy over the most recent predictions
        
        Args:
            window (int): Number of most recent predictions to analyze
            
        Returns:
            float: Slope of the trend line (negative means improving accuracy)
        """
        # Filter to only predictions with actual prices
        valid_predictions = self.predictions_df.dropna(subset=['actual_price'])
        
        if len(valid_predictions) < window:
            return 0  # Not enough data for a trend
            
        # Get the most recent predictions
        recent = valid_predictions.tail(window).copy()
        recent['index'] = range(len(recent))
        
        if len(recent) <= 1:
            return 0  # Not enough points for a trend
            
        # Calculate trend slope using numpy polyfit
        x = recent['index'].values
        y = recent['accuracy'].values
        
        # A negative slope means improving accuracy (smaller error)
        if len(x) > 1 and len(y) > 1:
            slope = np.polyfit(x, y, 1)[0]
            return slope
        
        return 0

    def suggest_model_adjustments(self, adjustment_threshold=0.1):
        """
        Suggest model adjustments based on prediction accuracy
        
        Args:
            adjustment_threshold (float): Threshold for suggesting adjustments
            
        Returns:
            dict: Dictionary with adjustment suggestions
        """
        # Get overall accuracy metrics
        metrics = self.get_overall_accuracy()
        
        # Get recent trend
        trend = self.get_recent_accuracy_trend()
        
        # Initialize suggestions
        suggestions = {
            'needs_adjustment': False,
            'learning_rate_adjustment': 0,
            'window_size_adjustment': 0,
            'smoothing_adjustment': 0,
            'reason': 'No adjustment needed'
        }
        
        # Not enough data for meaningful suggestions
        if metrics['count'] < 5:
            suggestions['reason'] = 'Not enough historical predictions'
            return suggestions
            
        # Check if accuracy is below threshold
        if metrics['threshold_met_pct'] < 0.5:
            suggestions['needs_adjustment'] = True
            suggestions['learning_rate_adjustment'] = 0.001  # Increase learning rate
            suggestions['reason'] = 'Overall accuracy is below 50%'
            
        # Check for worsening trend (positive slope means increasing error)
        if trend > 0.005:
            suggestions['needs_adjustment'] = True
            suggestions['window_size_adjustment'] = 2  # Increase window size
            suggestions['smoothing_adjustment'] = 0.1  # Increase smoothing
            suggestions['reason'] = 'Accuracy is trending worse'
            
        # Check for very poor recent performance
        recent = self.predictions_df.dropna(subset=['actual_price']).tail(3)
        if len(recent) >= 3 and (recent['threshold_met'] == False).all():
            suggestions['needs_adjustment'] = True
            suggestions['learning_rate_adjustment'] = 0.002  # Larger increase in learning rate
            suggestions['reason'] = 'Recent predictions all missed threshold'
            
        return suggestions 