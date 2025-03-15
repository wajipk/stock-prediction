import numpy as np

def get_column_case_insensitive(df, column_name):
    """
    Get column from DataFrame regardless of case sensitivity.
    
    Args:
        df (pd.DataFrame): DataFrame to search
        column_name (str): Column name to find (case insensitive)
        
    Returns:
        pd.Series: The requested column or raises a more helpful error
    """
    # Check for exact match first
    if column_name in df.columns:
        return df[column_name]
    
    # Try case-insensitive match
    col_lower = column_name.lower()
    for col in df.columns:
        if col.lower() == col_lower:
            print(f"Column '{column_name}' not found, using '{col}' instead")
            return df[col]
    
    # If we get here, the column doesn't exist in any case variation
    available_cols = ", ".join(list(df.columns))
    raise KeyError(f"Column '{column_name}' not found (case-insensitive). Available columns: {available_cols}")

def validate_predictions_against_reality(predictions, previous_price, historical_volatility=None, df=None, max_days=5):
    """
    Enhanced validation of predictions against historical reality with more aggressive checks.
    
    Args:
        predictions: Array of predicted prices
        previous_price: Last known closing price
        historical_volatility: Historical daily volatility (if None, will be calculated from df)
        df: DataFrame with historical data for volatility calculation
        max_days: Maximum number of days to check
        
    Returns:
        tuple: (is_realistic, warnings, adjusted_predictions)
    """
    if predictions is None or len(predictions) == 0:
        return True, [], predictions
    
    # Limit validation to max_days
    num_days = min(len(predictions), max_days)
    warnings = []
    adjusted_predictions = np.array(predictions).copy() if not isinstance(predictions, np.ndarray) else predictions.copy()
    is_realistic = True
    
    # Calculate historical volatility if not provided
    if historical_volatility is None and df is not None and len(df) > 30:
        try:
            close_prices = get_column_case_insensitive(df, 'Close').values[-30:]
            returns = np.diff(close_prices) / close_prices[:-1]
            historical_volatility = np.std(returns)
            print(f"Calculated historical volatility: {historical_volatility:.4f}")
        except Exception as e:
            print(f"Warning: Error calculating historical volatility: {e}")
            historical_volatility = 0.02  # Default if calculation fails
            print(f"Using default historical volatility: {historical_volatility}")
    elif historical_volatility is None:
        historical_volatility = 0.02  # Default 2% if we can't calculate
        print(f"Using default historical volatility: {historical_volatility}")
    
    # Calculate cumulative return
    cumulative_return = (predictions[num_days-1] / previous_price) - 1
    
    # Calculate daily returns
    daily_returns = np.zeros(num_days)
    daily_returns[0] = (predictions[0] / previous_price) - 1
    for i in range(1, num_days):
        daily_returns[i] = (predictions[i] / predictions[i-1]) - 1
    
    # Calculate standard deviation of predicted daily returns
    std_daily_returns = np.std(daily_returns)
    
    # Check if all predictions are in the same direction (all up or all down)
    all_up = all(pred >= previous_price for pred in predictions[:num_days])
    all_down = all(pred <= previous_price for pred in predictions[:num_days])
    
    # Count consecutive increases or decreases
    consecutive_increases = 0
    consecutive_decreases = 0
    current_increases = 0
    current_decreases = 0
    
    for i in range(num_days):
        if i == 0:
            if predictions[i] > previous_price:
                current_increases += 1
            elif predictions[i] < previous_price:
                current_decreases += 1
        else:
            if predictions[i] > predictions[i-1]:
                current_increases += 1
                current_decreases = 0
            elif predictions[i] < predictions[i-1]:
                current_decreases += 1
                current_increases = 0
            else:
                current_increases = 0
                current_decreases = 0
        
        consecutive_increases = max(consecutive_increases, current_increases)
        consecutive_decreases = max(consecutive_decreases, current_decreases)
    
    # 1. Check if the cumulative return is unrealistic
    if abs(cumulative_return) > historical_volatility * 5 * np.sqrt(num_days):
        warnings.append(f"WARNING: Cumulative {num_days}-day return of {cumulative_return:.2%} is unrealistically large")
        is_realistic = False
        
        # Apply a corrective adjustment - move toward the realistic range
        realistic_max_return = historical_volatility * 4 * np.sqrt(num_days) * np.sign(cumulative_return)
        correction_factor = (1 + realistic_max_return) / (1 + cumulative_return)
        
        # Apply gradual correction (stronger for later days)
        for i in range(num_days):
            # Gradual correction factor (from 0.3 to 1.0 of the full correction)
            day_factor = 0.3 + 0.7 * (i / (num_days - 1)) if num_days > 1 else 1.0
            adjusted_factor = 1.0 + (correction_factor - 1.0) * day_factor
            adjusted_predictions[i] *= adjusted_factor
    
    # 2. Check for unrealistic consecutive moves in the same direction
    max_realistic_consecutive = 3  # Maximum number of consecutive days in same direction
    if consecutive_increases > max_realistic_consecutive:
        warnings.append(f"WARNING: Unrealistic streak of {consecutive_increases} consecutive price increases")
        is_realistic = False
    elif consecutive_decreases > max_realistic_consecutive:
        warnings.append(f"WARNING: Unrealistic streak of {consecutive_decreases} consecutive price decreases")
        is_realistic = False
    
    # 3. Check for unrealistic uniformity - same percent change every day
    if std_daily_returns < historical_volatility * 0.5:
        warnings.append(f"WARNING: Daily returns have unnaturally low volatility ({std_daily_returns:.2%} vs historical {historical_volatility:.2%})")
        is_realistic = False
    
    # 4. Check for ALL predictions in same direction (stronger check for 5-day predictions)
    if num_days >= 4:
        if all_up and cumulative_return > 0.05:
            warnings.append(f"WARNING: All predictions are increasing ({cumulative_return:.2%} over {num_days} days)")
            is_realistic = False
            
            # For all-up predictions, apply a mean reversion to later days
            for i in range(1, num_days):
                reversion_factor = 0.1 + (i * 0.1)  # Increasing reversion (0.2 to 0.5)
                original_return = (adjusted_predictions[i] / previous_price) - 1
                reverted_return = original_return * (1 - reversion_factor)
                adjusted_predictions[i] = previous_price * (1 + reverted_return)
                
        elif all_down and cumulative_return < -0.05:
            warnings.append(f"WARNING: All predictions are decreasing ({cumulative_return:.2%} over {num_days} days)")
            is_realistic = False
            
            # For all-down predictions, apply mean reversion to later days
            for i in range(1, num_days):
                reversion_factor = 0.1 + (i * 0.1)  # Increasing reversion (0.2 to 0.5)
                original_return = (adjusted_predictions[i] / previous_price) - 1  # Negative
                reverted_return = original_return * (1 - reversion_factor)  # Less negative
                adjusted_predictions[i] = previous_price * (1 + reverted_return)
    
    # Print summary
    if len(warnings) > 0:
        print("\n==== PREDICTION REALITY CHECK ====")
        for warning in warnings:
            print(warning)
        print(f"Original prediction: {previous_price:.2f} -> {predictions[num_days-1]:.2f} ({cumulative_return:.2%})")
        adjusted_return = (adjusted_predictions[num_days-1] / previous_price) - 1
        print(f"Adjusted prediction: {previous_price:.2f} -> {adjusted_predictions[num_days-1]:.2f} ({adjusted_return:.2%})")
        print("================================\n")
    
    return is_realistic, warnings, adjusted_predictions 