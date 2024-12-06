import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, LabelEncoder
import os
import logging
import joblib
from pandas.tseries.offsets import BDay


class StockData:
    def __init__(self, data_path: str = './'):
        # File paths
        self.data_path = os.path.abspath(data_path)
        self.scaler = RobustScaler()
        self.label_encoder = LabelEncoder()
        self.data_file = os.path.join(self.data_path, 'trade_data.csv')
        
        # Set processed file path to Temp directory
        self.temp_directory = os.path.join(self.data_path, 'Temp')
        if not os.path.exists(self.temp_directory):
            os.makedirs(self.temp_directory)
        
        self.processed_file = os.path.join(self.temp_directory, 'trade_preprocessed_data.csv')
        self.encoder_file = os.path.join(self.data_path, 'label_encoder.pkl')
        self.scaler_file = os.path.join(self.data_path, 'scaler.pkl')

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

            # Returns (percentage change in close price)
            df['returns'] = df['close'].pct_change()

            # Volatility (rolling standard deviation of returns)
            df['volatility'] = df['returns'].rolling(window=20).std()

            # Moving Averages (MA)
            df['ma_14'] = df['close'].rolling(window=14).mean()
            df['ma_30'] = df['close'].rolling(window=30).mean()
            df['ma_50'] = df['close'].rolling(window=50).mean()

            # RSI (Relative Strength Index) for different periods
            for period in [14, 30, 50]:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
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
            df['force_index'] = (df['close'] - df['close'].shift(1)) * df['volume']

            self.logger.info("Technical indicators calculated.")
            return df

        except Exception as e:
            self.logger.error(f"Error in calculate_technical_indicators: {e}")
            raise

    def create_multistep_labels(self, df: pd.DataFrame, periods: dict) -> pd.DataFrame:
        try:
            self.logger.info("Creating multi-step target labels based on business days...")

            # Ensure the dataframe is sorted by symbol and date in ascending order (oldest date first)
            df = df.sort_values(by=['symbol', 'date'], ascending=[True, True])

            # Convert 'date' column to datetime if it's not already
            df['date'] = pd.to_datetime(df['date'])

            # Set 'date' as the index to enable time-based operations
            df.set_index('date', inplace=True)

            # Initialize the columns for the target days (target_day1 to target_day30)
            target_columns = [f'target_day{i}' for i in range(1, 31)]

            # Loop through each symbol and create the target columns
            for symbol in df['symbol'].unique():
                symbol_data = df[df['symbol'] == symbol].copy()

                # For each target day (target_day1 to target_day30)
                for i in range(1, 31):
                    target_col = f'target_day{i}'

                    # Loop through each row of the symbol's data and calculate the target value
                    for index, row in symbol_data.iterrows():
                        # Find the next business day for each row
                        next_business_day = row.name + BDay(i)

                        # Check if the next business day exists in the dataset
                        if next_business_day in symbol_data.index:
                            # Assign the close price of the target day (next business day)
                            df.loc[index, target_col] = symbol_data.loc[next_business_day, 'close']
                        else:
                            # If no data is available for the target day, assign NaN
                            df.loc[index, target_col] = np.nan

                # After shifting, log whether the columns are populated
                missing_cols = [col for col in target_columns if col not in df.columns or df[col].isna().all()]
                if missing_cols:
                    self.logger.warning(f"Missing or empty columns for {symbol}: {', '.join(missing_cols)}")

            # Check if target columns were successfully created
            missing_cols = [col for col in target_columns if col not in df.columns]
            if missing_cols:
                raise KeyError(f"Missing target columns: {', '.join(missing_cols)}")

            # Drop rows where any target is NaN (those without a valid future target)
            df.dropna(subset=target_columns, inplace=True)

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
            mapping_file = os.path.join(self.data_path, 'symbol_mapping_complete.csv')
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
                self.scaler = RobustScaler()
        except Exception as e:
            self.logger.error(f"Error loading encoder and scaler: {e}")
            raise
