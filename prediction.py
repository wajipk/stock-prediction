import pandas as pd
import numpy as np
from model import StockPredictionModel
from loaddata import StockData
from pandas.tseries.offsets import BDay


class StockPredictor:
    def __init__(self, model_path='./stock_model.keras'):
        print("Loading model and encoders...")
        self.model = StockPredictionModel.load_model(model_path)
        self.stock_data = StockData()
        self.stock_data.load_encoder_and_scaler()
        print("Model and encoders loaded successfully.")

    def get_next_business_day(self, date, days=1):
        """Get the next business day(s) after the given date."""
        return date + BDay(days)

    def predict_all_periods(self, symbol: str):
        try:
            print(f"Generating predictions for symbol: {symbol}...")
            # Read the processed data
            df = pd.read_csv(self.stock_data.processed_file)
            df['date'] = pd.to_datetime(df['date'])
            
            features = [
                'close', 'volume', 'volatility', 'ma_20', 'ma_50',
                'rsi', 'macd', 'open', 'symbol_encoded'
            ]

            # Load the original data to get the actual symbol mapping
            original_df = pd.read_csv(self.stock_data.data_file)
            available_symbols = original_df['symbol'].unique()

            if symbol not in available_symbols:
                raise ValueError(f"Symbol '{symbol}' not found in the dataset. Available symbols: {', '.join(available_symbols[:5])}...")

            # Get the encoded value for the symbol
            symbol_encoded = float(self.stock_data.label_encoder.transform([symbol])[0])
            
            # Filter data for the specific symbol
            data = df[df['symbol_encoded'] == symbol_encoded].copy()
            
            if len(data) == 0:
                raise ValueError(f"No processed data available for symbol: {symbol}")

            # Get the sequence length from the model's input shape
            seq_length = self.model.layers[0].input_shape[1]
            
            if len(data) < seq_length:
                raise ValueError(f"Insufficient data for symbol {symbol}. Need at least {seq_length} data points.")

            # Sort data by date and get the latest date
            data = data.sort_values('date')
            latest_date = data['date'].max()

            # Prepare input sequence
            X = data[features].values[-seq_length:].reshape(1, seq_length, len(features))

            # Make predictions for all periods
            raw_predictions = self.model.predict(X)

            # Initialize predictions dictionary
            predictions = {}
            
            # Get the last row of actual data for reference values
            last_row = data.iloc[-1]
            
            # Create a dummy array for inverse scaling
            dummy_array = np.zeros((1, len(features) + 30))  # features + 30 target columns
            
            for i in range(30):
                period = f'day{i+1}'
                
                # Calculate the prediction date (excluding weekends)
                pred_date = self.get_next_business_day(latest_date, days=i+1)
                
                # Get the scaled prediction for this period
                scaled_pred = raw_predictions[0][i]
                
                # Put the prediction in the correct position for inverse scaling
                dummy_array[0, len(features) + i] = scaled_pred
                
                # Inverse transform to get the actual price
                unscaled_pred = self.stock_data.scaler.inverse_transform(dummy_array)[0, len(features) + i]
                
                predictions[period] = {
                    'date': pred_date.strftime('%Y-%m-%d'),
                    'predicted_price': float(unscaled_pred),
                }

                # For day30, this is the closing price
                if i == 29:  # day30
                    predictions[period]['price_type'] = 'close'
                # For day29, this is the opening price for day30
                elif i == 28:  # day29
                    predictions[period]['price_type'] = 'next_day_open'
                else:
                    predictions[period]['price_type'] = 'intermediate'

                # Copy over the last known technical indicators
                # These will be updated in post-processing
                predictions[period].update({
                    'volume': float(last_row['volume']),
                    'volatility': float(last_row['volatility']),
                    'ma_20': float(last_row['ma_20']),
                    'ma_50': float(last_row['ma_50']),
                    'rsi': float(last_row['rsi']),
                    'macd': float(last_row['macd'])
                })

            # Post-process predictions to update technical indicators
            self._update_technical_indicators(predictions, last_row)

            print("Predictions generated successfully.")
            return predictions

        except Exception as e:
            print(f"Error during prediction: {e}")
            raise

    def _update_technical_indicators(self, predictions, last_row):
        """Update technical indicators for the predicted prices."""
        try:
            # Create a temporary dataframe with historical and predicted prices
            temp_df = pd.DataFrame(predictions).T
            temp_df['date'] = pd.to_datetime(temp_df['date'])
            temp_df = temp_df.sort_values('date')

            # Calculate returns
            prices = [last_row['close']] + [p['predicted_price'] for p in predictions.values()]
            returns = pd.Series(prices).pct_change().values[1:]
            
            # Update volatility (20-day rolling standard deviation of returns)
            historical_returns = pd.Series(last_row['returns'])
            rolling_returns = pd.concat([historical_returns, pd.Series(returns)])
            volatility = rolling_returns.rolling(window=20).std().values[-len(returns):]

            # Update moving averages
            historical_prices = pd.Series(last_row['close'])
            all_prices = pd.concat([historical_prices, pd.Series([p['predicted_price'] for p in predictions.values()])])
            ma_20 = all_prices.rolling(window=20).mean().values[-len(returns):]
            ma_50 = all_prices.rolling(window=50).mean().values[-len(returns):]

            # Update RSI
            delta = all_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).values[-len(returns):]

            # Update MACD
            exp1 = all_prices.ewm(span=12, adjust=False).mean()
            exp2 = all_prices.ewm(span=26, adjust=False).mean()
            macd = (exp1 - exp2).values[-len(returns):]

            # Update the predictions dictionary with new technical indicators
            for i, (period, pred) in enumerate(predictions.items()):
                pred['volatility'] = float(volatility[i]) if i < len(volatility) else float(volatility[-1])
                pred['ma_20'] = float(ma_20[i]) if i < len(ma_20) else float(ma_20[-1])
                pred['ma_50'] = float(ma_50[i]) if i < len(ma_50) else float(ma_50[-1])
                pred['rsi'] = float(rsi[i]) if i < len(rsi) else float(rsi[-1])
                pred['macd'] = float(macd[i]) if i < len(macd) else float(macd[-1])

        except Exception as e:
            print(f"Error updating technical indicators: {e}")
            raise