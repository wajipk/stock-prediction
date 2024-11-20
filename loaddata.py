import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import os
import logging
from typing import Optional, Tuple
import joblib


class StockData:
    def __init__(self, data_path: str = './'):
        # Set up the file paths
        self.data_path = os.path.abspath(data_path)

        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()
        self.data_file = os.path.join(self.data_path, 'trade_data.csv')
        self.processed_file = os.path.join(self.data_path, 'trade_preprocessed_data.csv')

        # Save paths for encoder, scaler, and model
        self.encoder_file = os.path.join(self.data_path, 'label_encoder.pkl')
        self.scaler_file = os.path.join(self.data_path, 'scaler.pkl')
        self.model_file = os.path.join(self.data_path, 'stock_model.keras')

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            self.logger.info("Starting data cleaning...")
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

            self.logger.info("Data cleaning completed.")
            return df

        except Exception as e:
            self.logger.error(f"Error in clean_data: {e}")
            raise

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            self.logger.info("Calculating technical indicators...")
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

            self.logger.info("Technical indicators calculated.")
            return df

        except Exception as e:
            self.logger.error(f"Error in calculate_technical_indicators: {e}")
            raise

    def create_multistep_labels(self, df: pd.DataFrame, periods: dict) -> pd.DataFrame:
        try:
            self.logger.info("Creating multi-step target labels...")
            for period, days in periods.items():
                df[f'target_{period}'] = df.groupby('symbol')['close'].shift(-days)

            df.dropna(inplace=True)
            self.logger.info("Multi-step target labels created.")
            return df

        except Exception as e:
            self.logger.error(f"Error in create_multistep_labels: {e}")
            raise

    def load_stock_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None,
                        use_incremental: bool = False) -> Tuple[pd.DataFrame, list]:
        try:
            self.logger.info("Loading raw stock data...")
            if not os.path.exists(self.data_file):
                raise FileNotFoundError(f"{self.data_file} does not exist.")

            df = pd.read_csv(self.data_file)
            self.logger.info(f"Raw data loaded with {len(df)} records.")

            if start_date:
                start_date = pd.to_datetime(start_date)
                df = df[df['date'] >= start_date]
            if end_date:
                end_date = pd.to_datetime(end_date)
                df = df[df['date'] <= end_date]

            df = self.clean_data(df)
            df = self.calculate_technical_indicators(df)

            periods = {f'day{i}': i for i in range(1, 31)}
            df = self.create_multistep_labels(df, periods)

            df['symbol_encoded'] = self.label_encoder.fit_transform(df['symbol'])
            df[['symbol_encoded']] = self.scaler.fit_transform(df[['symbol_encoded']])

            joblib.dump(self.label_encoder, self.encoder_file)
            joblib.dump(self.scaler, self.scaler_file)

            features = ['close', 'volume', 'volatility', 'ma_20', 'ma_50', 'rsi', 'macd', 'open', 'symbol_encoded']
            df[features] = self.scaler.fit_transform(df[features])

            df.to_csv(self.processed_file, index=False)
            self.logger.info(f"Preprocessed data saved to {self.processed_file}.")
            return df, features

        except Exception as e:
            self.logger.error(f"Error in load_stock_data: {e}")
            raise

    def load_encoder_and_scaler(self):
        try:
            if os.path.exists(self.encoder_file) and os.path.exists(self.scaler_file):
                self.label_encoder = joblib.load(self.encoder_file)
                self.scaler = joblib.load(self.scaler_file)
                self.logger.info("Encoders and scalers loaded successfully.")
            else:
                raise FileNotFoundError("LabelEncoder or MinMaxScaler not found.")
        except Exception as e:
            self.logger.error(f"Error loading encoder and scaler: {e}")
            raise
