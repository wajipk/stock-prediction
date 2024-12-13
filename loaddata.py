import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import logging
import joblib
from pandas.tseries.offsets import BDay
from multiprocessing import cpu_count
from joblib import Parallel, delayed

class StockData:
    def __init__(self, data_path: str = './'):
        # File paths
        self.data_path = os.path.abspath(data_path)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.data_file = os.path.join(self.data_path, 'trade_data.csv')

        # Set processed file path to Temp directory
        self.temp_directory = os.path.join(self.data_path, 'Temp')
        if not os.path.exists(self.temp_directory):
            os.makedirs(self.temp_directory)

        self.processed_file = os.path.join(self.temp_directory, 'trade_preprocessed_data.csv')
        self.encoder_file = os.path.join(self.temp_directory, 'label_encoder.pkl')
        self.scaler_file = os.path.join(self.temp_directory, 'scaler.pkl')

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            self.logger.info("Starting data cleaning...")

            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').drop_duplicates(subset=['date', 'symbol'], keep='last')
            df['close'] = df['close'].interpolate(method='linear')
            df['volume'] = df['volume'].fillna(df['volume'].mean())
            df.dropna(inplace=True)

            # Drop 'name' column if it exists
            if 'name' in df.columns:
                df.drop(columns=['name'], inplace=True)

            self.logger.info("Data cleaning completed.")
            return df

        except Exception as e:
            self.logger.error(f"Error in clean_data: {e}")
            raise

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            self.logger.info("Calculating technical indicators...")

            # Use vectorized pandas operations instead of loops
            df['returns'] = df['close'].pct_change()

            # Volatility (rolling standard deviation of returns)
            df['volatility'] = df['returns'].ewm(span=20, adjust=False).std()

            # Moving Averages (MA)
            df['ma_14'] = df['close'].ewm(span=14, adjust=False).mean()
            df['ma_30'] = df['close'].ewm(span=30, adjust=False).mean()
            df['ma_50'] = df['close'].ewm(span=50, adjust=False).mean()

            # RSI (Relative Strength Index) for different periods
            for period in [14, 30, 50]:
                delta = df['close'].diff()
                gain = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
                loss = -delta.clip(upper=0).ewm(span=period, adjust=False).mean()
                rs = gain / loss
                df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

            # MACD (Moving Average Convergence Divergence)
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()

            # OBV (On-Balance Volume)
            df['obv'] = np.where(df['close'] > df['close'].shift(1), df['volume'],
                                 np.where(df['close'] < df['close'].shift(1), -df['volume'], 0))
            df['obv'] = df['obv'].cumsum()

            # Force Index (FI) = (close - previous close) * volume
            df['force_index'] = df['close'].diff().mul(df['volume'], fill_value=0)

            self.logger.info("Technical indicators calculated.")
            return df

        except Exception as e:
            self.logger.error(f"Error in calculate_technical_indicators: {e}")
            raise

    def create_multistep_labels(self, df: pd.DataFrame, periods: dict) -> pd.DataFrame:
        try:
            self.logger.info("Creating multi-step target labels...")

            # Ensure the date column is in datetime format
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values(by=['symbol', 'date'], ascending=[True, True])

            # Function to create multi-step labels for each symbol
            def process_symbol_data(symbol_data, periods):
                # Set 'date' column as the index and convert to DatetimeIndex
                symbol_data.set_index('date', inplace=True)

                # Ensure the index is a DatetimeIndex for shift operation
                if not isinstance(symbol_data.index, pd.DatetimeIndex):
                    symbol_data.index = pd.to_datetime(symbol_data.index)

                # Apply the shift for each period
                for period, days in periods.items():
                    # Shift close prices by the specified business days
                    symbol_data[f'target_{period}'] = symbol_data['close'].shift(-days, freq=BDay())
                
                # Reset index after processing
                symbol_data.reset_index(inplace=True)
                return symbol_data

            # Parallelize the operation for each symbol
            symbols = df['symbol'].unique()
            df_list = Parallel(n_jobs=cpu_count())(
                delayed(process_symbol_data)(df[df['symbol'] == symbol], periods)
                for symbol in symbols
            )

            # Combine the processed data
            df = pd.concat(df_list)

            # Drop rows with NaN values in any target column
            target_columns = [f'target_{period}' for period in periods]
            df.dropna(subset=target_columns, inplace=True)

            # Reset index
            df.reset_index(drop=True, inplace=True)

            self.logger.info("Multi-step target labels created successfully.")
            return df

        except Exception as e:
            self.logger.error(f"Error in create_multistep_labels: {e}")
            raise

    def preprocess_data(self):
        try:
            self.logger.info("Starting preprocessing of stock data...")

            if not os.path.exists(self.data_file):
                raise FileNotFoundError(f"{self.data_file} not found.")

            # Load raw data
            df = pd.read_csv(self.data_file, low_memory=False)

            # Clean data
            df = self.clean_data(df)

            # Add technical indicators
            df = self.calculate_technical_indicators(df)

            # Create multi-step target labels based on business days
            periods = {f'day{i}': i for i in range(1, 31)}
            df = self.create_multistep_labels(df, periods)

            # Encode the symbol column
            df['symbol_encoded'] = self.label_encoder.fit_transform(df['symbol'])

            # Save complete symbol mapping
            symbol_mapping = pd.DataFrame({
                'symbol': df['symbol'],
                'symbol_encoded_unscaled': self.label_encoder.transform(df['symbol']),
                'symbol_encoded_scaled': None  # Placeholder, will be filled after scaling
            }).drop_duplicates()

            # Features to scale
            features = ['close', 'volume', 'volatility', 'ma_14', 'ma_30', 'ma_50', 'rsi_14', 'rsi_30', 'rsi_50', 
                        'macd', 'obv', 'force_index', 'symbol_encoded']

            # Check the DataFrame size before scaling
            self.logger.debug(f"DataFrame shape before scaling: {df.shape}")

            # Check if DataFrame has any rows after target creation
            if df.shape[0] == 0:
                raise ValueError("The DataFrame is empty after target creation. No data available for scaling.")

            # Scale all features
            df[features + [f'target_day{i}' for i in range(1, 31)]] = self.scaler.fit_transform(
                df[features + [f'target_day{i}' for i in range(1, 31)]]
            )

            # Update scaled values in the symbol mapping
            scaled_mapping = df[['symbol_encoded', 'symbol']].drop_duplicates().rename(
                columns={'symbol_encoded': 'symbol_encoded_scaled'}
            )
            symbol_mapping = symbol_mapping.merge(scaled_mapping, on='symbol', how='left')

            # Save updated symbol mapping
            mapping_file = os.path.join(self.temp_directory, 'symbol_mapping_complete.csv')
            symbol_mapping.to_csv(mapping_file, index=False)
            self.logger.info(f"Complete symbol mapping saved to {mapping_file}.")

            # Drop the symbol column
            df.drop(columns=['symbol'], inplace=True)

            # Save processed data to the Temp directory
            processed_file = self.processed_file
            df.to_csv(processed_file, index=False)
            self.logger.info(f"Processed data saved at {processed_file}.")

            return df, features

        except Exception as e:
            self.logger.error(f"Error in preprocess_data: {e}")
            raise

    def load_stock_data(self, use_incremental=False):
        try:
            self.logger.info("Loading preprocessed stock data...")

            if not os.path.exists(self.processed_file):
                raise FileNotFoundError(f"Preprocessed data not found. Please run preprocess_data first.")

            # Load preprocessed data
            df = pd.read_csv(self.processed_file, low_memory=False)

            # Extract features used in model
            features = ['close', 'volume', 'volatility', 'ma_14', 'ma_30', 'ma_50', 'rsi_14', 'rsi_30', 'rsi_50', 
                        'macd', 'obv', 'force_index', 'symbol_encoded']

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
                self.logger.warning("Encoder or scaler file not found. Initializing new instances.")
                self.label_encoder = LabelEncoder()
                self.scaler = StandardScaler()
        except Exception as e:
            self.logger.error(f"Error loading encoder and scaler: {e}")
            raise
