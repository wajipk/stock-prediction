import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import os
import logging
from typing import Optional, Tuple


class StockData:
    def __init__(self, data_path: str = 'data/'):
        self.data_path = data_path
        self.scaler = MinMaxScaler()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.csv_path = 'C:/Users/DELL/Downloads/StockPrediction/trade_data.csv'
        self.incremental_csv_path = 'C:/Users/DELL/Downloads/StockPrediction/incremental_data.csv'

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

    def load_stock_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Tuple[pd.DataFrame, list]:
        try:
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=5 * 365)).strftime('%Y-%m-%d')
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')

            df = pd.read_csv(self.csv_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            df = self.clean_data(df)
            df = self.calculate_technical_indicators(df)

            # One-hot encode 'symbol' and 'name'
            encoded = self.encoder.fit_transform(df[['symbol', 'name']])
            encoded_cols = self.encoder.get_feature_names_out(['symbol', 'name'])
            encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
            df = pd.concat([df, encoded_df], axis=1)
            df.drop(columns=['symbol', 'name'], inplace=True)

            features = ['close', 'volume', 'volatility', 'ma_20', 'ma_50', 'rsi', 'macd', 'open'] + list(encoded_cols)
            df[features] = self.scaler.fit_transform(df[features])

            processed_file = f'{self.data_path}/processed_stock_data.csv'
            df.to_csv(processed_file, index=False)
            self.logger.info("Successfully loaded and processed stock data.")
            return df, features

        except Exception as e:
            self.logger.error(f"Error in load_stock_data: {str(e)}")
            raise

    def get_incremental_data(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.incremental_csv_path)
            df['date'] = pd.to_datetime(df['date'])
            if len(df) == 0:
                self.logger.info("No new incremental data available.")
                return None
            df = self.clean_data(df)
            df = self.calculate_technical_indicators(df)

            # One-hot encode 'symbol' and 'name'
            encoded = self.encoder.transform(df[['symbol', 'name']])
            encoded_cols = self.encoder.get_feature_names_out(['symbol', 'name'])
            encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
            df = pd.concat([df, encoded_df], axis=1)
            df.drop(columns=['symbol', 'name'], inplace=True)

            incremental_file = f'{self.data_path}/incremental_processed_data.csv'
            df.to_csv(incremental_file, index=False)
            self.logger.info("Successfully loaded incremental data.")
            return df

        except Exception as e:
            self.logger.error(f"Error in get_incremental_data: {str(e)}")
            raise


if __name__ == "__main__":
    stock_data = StockData()
    try:
        df, features = stock_data.load_stock_data()
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error: {str(e)}")
