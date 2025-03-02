"""
Financial rules module for stock price adjustments

This module contains rules and functions to adjust stock prices for corporate actions
like dividends and bonus shares. These adjustments are crucial for accurate 
model training and prediction.
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime


class StockAdjustmentRules:
    """
    Class to handle stock price adjustments based on corporate actions
    """
    
    def __init__(self):
        """Initialize with known corporate actions"""
        # Store dividend events: {symbol: [(date, amount)]}
        self.dividend_events = {}
        
        # Store bonus share events: {symbol: [(date, ratio)]}
        # ratio is expressed as N:M where you get N shares for every M you hold
        self.bonus_events = {}
        
        # API base URL
        self.api_base_url = "https://stocks.wajipk.com"
    
    def load_events_from_file(self, file_path):
        """
        Load corporate action events from a CSV file
        
        Args:
            file_path (str): Path to the CSV file with corporate action data
            
        CSV format expected:
        symbol,date,action_type,value
        MARI,2023-01-15,dividend,2.5
        MARI,2023-05-10,bonus,3:1
        """
        try:
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                symbol = row['symbol']
                date = pd.to_datetime(row['date']).tz_localize(None)  # Ensure timezone-naive
                
                # Convert action_type to string before calling lower()
                action_type = str(row['action_type']).lower()
                value = row['value']
                
                if action_type == 'dividend':
                    self.add_dividend_event(symbol, date, float(value))
                elif action_type == 'bonus':
                    self.add_bonus_event(symbol, date, value)
                    
            return True
        except Exception as e:
            print(f"Error loading corporate actions: {e}")
            return False
    
    def fetch_payouts_from_api(self, symbol):
        """
        Fetch dividend payout information from the API for a given symbol
        
        Args:
            symbol (str): Stock symbol to fetch payouts for
            
        Returns:
            tuple: (dividend_events, None) where dividend_events is a list of (date, amount) tuples
            or (None, None) if the API call fails
        """
        url = f"{self.api_base_url}/api/payouts?symbol={symbol}"
        
        try:
            print(f"Fetching dividend data for {symbol} from API...")
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                print(f"Error fetching payout data: HTTP {response.status_code}")
                return None, None
            
            data = response.json()
            if not data:
                print(f"No payout data found for {symbol}")
                return [], None
            
            # Process dividend events
            dividend_events = []
            
            for payout in data:
                # Use xdate (ex-dividend date) if available, otherwise use announcedate
                if payout.get('xdate'):
                    date_str = payout['xdate']
                elif payout.get('announcedate'):
                    date_str = payout['announcedate']
                else:
                    print(f"Skipping record with no date: {payout}")
                    continue
                
                # Parse the date
                try:
                    date = pd.to_datetime(date_str).tz_localize(None)  # Ensure timezone-naive
                    
                    # Get the dividend amount directly from the 'dividend' field
                    if payout.get('dividend') is not None and payout.get('dividend') != "":
                        try:
                            dividend_amount = float(payout['dividend'])
                            if dividend_amount > 0:
                                dividend_events.append((date, dividend_amount))
                        except (ValueError, TypeError):
                            print(f"Invalid dividend amount: {payout.get('dividend')}")
                except (ValueError, KeyError) as e:
                    print(f"Error processing payout record: {e}")
                    continue
            
            print(f"Loaded {len(dividend_events)} dividend events from API for {symbol}")
            return dividend_events, None
            
        except Exception as e:
            print(f"Exception while fetching payout data: {e}")
            return None, None
    
    def add_dividend_event(self, symbol, date, amount):
        """
        Add a dividend event
        
        Args:
            symbol (str): Stock symbol
            date (datetime): Date of dividend
            amount (float): Dividend amount in currency
        """
        if symbol not in self.dividend_events:
            self.dividend_events[symbol] = []
        
        self.dividend_events[symbol].append((date, amount))
        # Sort by date to ensure chronological processing
        self.dividend_events[symbol].sort(key=lambda x: x[0])
    
    def add_bonus_event(self, symbol, date, ratio):
        """
        Add a bonus share event
        
        Args:
            symbol (str): Stock symbol
            date (datetime): Date of bonus issue
            ratio (str): Bonus ratio in format "N:M"
        """
        if symbol not in self.bonus_events:
            self.bonus_events[symbol] = []
        
        self.bonus_events[symbol].append((date, ratio))
        # Sort by date to ensure chronological processing
        self.bonus_events[symbol].sort(key=lambda x: x[0])
    
    def apply_rules(self, df, symbol):
        """
        Apply all rules to adjust stock prices
        
        Args:
            df (pd.DataFrame): DataFrame with stock price data
            symbol (str): Stock symbol
            
        Returns:
            pd.DataFrame: Adjusted DataFrame
        """
        # Ensure the dataframe is sorted by date
        df = df.sort_values('date', ascending=True).reset_index(drop=True)
        
        # Apply dividend adjustments
        df = self.adjust_for_dividends(df, symbol)
        
        # Apply bonus share adjustments - only if we have bonus data
        if symbol in self.bonus_events and len(self.bonus_events[symbol]) > 0:
            df = self.adjust_for_bonus_shares(df, symbol)
        
        return df
    
    def adjust_for_dividends(self, df, symbol):
        """
        Adjust stock prices for dividend events
        
        Args:
            df (pd.DataFrame): DataFrame with stock price data
            symbol (str): Stock symbol
            
        Returns:
            pd.DataFrame: Adjusted DataFrame
        """
        if symbol not in self.dividend_events:
            return df
        
        # Create a copy to avoid modifying the original
        adjusted_df = df.copy()
        
        # Convert date column to datetime if it's not already
        adjusted_df['date'] = pd.to_datetime(adjusted_df['date']).dt.tz_localize(None)
        
        for dividend_date, dividend_amount in self.dividend_events[symbol]:
            # Ensure the dividend_date is timezone-naive
            naive_dividend_date = dividend_date.tz_localize(None) if dividend_date.tzinfo is not None else dividend_date
            
            # Find the index in dataframe where the date is the dividend date
            mask = adjusted_df['date'] == naive_dividend_date
            
            if mask.any():
                dividend_idx = adjusted_df.index[mask][0]
                
                # Apply Rule 1: Subtract dividend amount from prices before dividend date
                # This adjusts historical prices to account for the dividend drop
                adjusted_df.loc[:dividend_idx, 'open'] -= dividend_amount
                adjusted_df.loc[:dividend_idx, 'high'] -= dividend_amount
                adjusted_df.loc[:dividend_idx, 'low'] -= dividend_amount
                adjusted_df.loc[:dividend_idx, 'close'] -= dividend_amount
        
        return adjusted_df
    
    def adjust_for_bonus_shares(self, df, symbol):
        """
        Adjust stock prices for bonus share events
        
        Args:
            df (pd.DataFrame): DataFrame with stock price data
            symbol (str): Stock symbol
            
        Returns:
            pd.DataFrame: Adjusted DataFrame
        """
        if symbol not in self.bonus_events:
            return df
        
        # Create a copy to avoid modifying the original
        adjusted_df = df.copy()
        
        # Convert date column to datetime if it's not already
        adjusted_df['date'] = pd.to_datetime(adjusted_df['date']).dt.tz_localize(None)
        
        for bonus_date, bonus_ratio in self.bonus_events[symbol]:
            # Ensure the bonus_date is timezone-naive
            naive_bonus_date = bonus_date.tz_localize(None) if bonus_date.tzinfo is not None else bonus_date
            
            # Parse the bonus ratio (N:M)
            numerator, denominator = map(int, bonus_ratio.split(':'))
            adjustment_factor = (1 + numerator / denominator)
            
            # Find the index in dataframe where the date is the bonus date
            mask = adjusted_df['date'] == naive_bonus_date
            
            if mask.any():
                bonus_idx = adjusted_df.index[mask][0]
                
                # Apply Rule 2: Adjust prices before bonus date
                # Formula: Current share price (cum-bonus) / (1 + Bonus issue ratio)
                adjusted_df.loc[:bonus_idx, 'open'] /= adjustment_factor
                adjusted_df.loc[:bonus_idx, 'high'] /= adjustment_factor
                adjusted_df.loc[:bonus_idx, 'low'] /= adjustment_factor
                adjusted_df.loc[:bonus_idx, 'close'] /= adjustment_factor
        
        return adjusted_df
    
    def clear_events(self):
        """
        Clear all loaded dividend and bonus events
        """
        self.dividend_events = {}
        self.bonus_events = {}


# Create a default instance that can be imported
default_rules = StockAdjustmentRules()


def adjust_stock_data(df, symbol):
    """
    Wrapper function to apply all stock adjustment rules
    
    Args:
        df (pd.DataFrame): DataFrame with stock price data
        symbol (str): Stock symbol
        
    Returns:
        pd.DataFrame: Adjusted DataFrame
    """
    return default_rules.apply_rules(df, symbol)


def fetch_and_load_payouts(symbol):
    """
    Fetch payout records from the API and load them into rules
    
    Args:
        symbol (str): Stock symbol to fetch and load payouts for
        
    Returns:
        bool: True if successful, False otherwise
    """
    dividend_events, _ = default_rules.fetch_payouts_from_api(symbol)
    
    if dividend_events is None:
        print("Using fallback local corporate actions data")
        return False
    
    # Add the events to our rules
    for date, amount in dividend_events:
        default_rules.add_dividend_event(symbol, date, amount)
    
    return True 