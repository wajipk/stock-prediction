import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import os
import logging
from typing import Optional, Tuple
import joblib


class StockData:
    def __init__(self, data_path: str = 'stock-prediction/'):
        self.data_path = data_path
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()
        self.data_file = os.path.join(data_path, 'trade_data.csv')
        self.processed_file = os.path.join(data_path, 'trade_preprocessed_data.csv')

        # File paths for saving encoders and scalers in the stock-prediction directory
        self.encoder_file = os.path.join(data_path, 'label_encoder.pkl')
        self.scaler_file = os.path.join(data_path, 'scaler.pkl')
        self.model_file = os.path.join(data_path, 'stock_model.h5')  # For saving the trained model

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Create data directory if it doesn't exist
        if not os.path.exists(data_path):
            os.makedirs(data_path)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').drop_duplicates(subset=['date', 'symbol'], keep='last')
            df['close'] = df['close'].interpolate(method='linear')
            df['volume'] = df['volume'].fillna(df['volume'].mean())
            df['open'] = df['close'].shift(1).fillna(df['close'])
            df.dropna(inplace=True)

            for col in ['close', 'volume']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]

            return df

        except Exception as e:
            self.logger.error(f"Error in clean_data: {str(e)}")
            raise

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std()
            df['ma_20'] = df['close'].rolling(window=20).mean()
            df['ma_50'] = df['close'].rolling(window=50).mean()
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()
            return df

        except Exception as e:
            self.logger.error(f"Error in calculate_technical_indicators: {str(e)}")
            raise

    def create_multistep_labels(self, df: pd.DataFrame, periods: dict) -> pd.DataFrame:
        try:
            # Generate target labels for each period within the same symbol
            for period, days in periods.items():
                df[f'target_{period}'] = (
                    df.groupby('symbol')['close']
                    .shift(-days)  # Shift close prices upward for each symbol group
                )

            # Drop rows with NaN values in any target column
            df.dropna(inplace=True)

            return df

        except Exception as e:
            self.logger.error(f"Error in create_multistep_labels: {str(e)}")
            raise

    def load_stock_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None,
                        use_incremental: bool = False) -> Tuple[pd.DataFrame, list]:
        try:
            if not os.path.exists(self.data_file):
                raise FileNotFoundError(f"{self.data_file} does not exist.")

            if start_date is None:
                start_date = (datetime.now() - timedelta(days=5 * 365)).strftime('%Y-%m-%d')
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')

            df = pd.read_csv(self.data_file)
            if use_incremental:
                self.logger.info("Using incremental data.")
            else:
                self.logger.info("Using full dataset.")
                # If not incremental, consider only the main dataset
                df = df[df['is_incremental'] == 0] if 'is_incremental' in df.columns else df

            df['date'] = pd.to_datetime(df['date'])
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            df = self.clean_data(df)
            df = self.calculate_technical_indicators(df)

            # Define prediction periods for day 1 to day 30
            periods = {f'day{i}': i for i in range(1, 31)}  # Day 1 to 30
            df = self.create_multistep_labels(df, periods)

            # Label encode 'symbol' and scale it
            df['symbol_encoded'] = self.label_encoder.fit_transform(df['symbol'])
            df['symbol_encoded'] = self.scaler.fit_transform(df[['symbol_encoded']])

            # Save the LabelEncoder and Scaler for future use
            joblib.dump(self.label_encoder, self.encoder_file)
            joblib.dump(self.scaler, self.scaler_file)

            # Drop unnecessary columns
            df.drop(columns=['symbol', 'name'], inplace=True)

            features = ['close', 'volume', 'volatility', 'ma_20', 'ma_50', 'rsi', 'macd', 'open', 'symbol_encoded']
            df[features] = self.scaler.fit_transform(df[features])

            # Save processed data as `trade_preprocessed.csv`
            df.to_csv(self.processed_file, index=False)
            self.logger.info(f"Successfully processed and saved data to {self.processed_file}.")
            return df, features

        except Exception as e:
            self.logger.error(f"Error in load_stock_data: {str(e)}")
            raise

    def load_encoder_and_scaler(self):
        if os.path.exists(self.encoder_file) and os.path.exists(self.scaler_file):
            self.label_encoder = joblib.load(self.encoder_file)
            self.scaler = joblib.load(self.scaler_file)
        else:
            raise FileNotFoundError("LabelEncoder or MinMaxScaler not found. Ensure preprocessing has been done.")
