# Stock Prediction System

This system predicts stock prices for multiple companies using historical data, machine learning models, and market trend analysis.

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Scripts

The repository contains the following scripts:

- `main.py` - The main stock prediction pipeline for a single company
- `run_all_companies.py` - Script to run predictions for all Islamic companies

## Running Predictions for All Companies

To run predictions for all Islamic companies listed on the API:

```bash
python run_all_companies.py
```

### Optional parameters:

- `--skip_training` - Skip model training and use existing models
- `--prediction_days` - Number of days ahead to predict (default: 5)
- `--threshold` - Threshold percentage for significant movements (default: 2.0)
- `--max_retries` - Maximum number of retry attempts for failed companies (default: 2)
- `--start_index` - Index to start from in the company list (for resuming interrupted runs)
- `--market_adjustment` - Market trend adjustment factor (default: 0.03, range: 0.0-0.1)
- `--no_market_trends` - Skip applying market trend adjustments

Example:

```bash
python run_all_companies.py --skip_training --prediction_days 7 --threshold 1.5 --market_adjustment 0.05
```

### Resuming Failed Runs

If the process is interrupted or some companies fail, you can resume:

```bash
# Resume from a specific index
python run_all_companies.py --start_index 50

# Run only failed companies from a previous run
cat failed_companies.txt | xargs -I{} python main.py --symbol {}
```

## Overwriting Predictions

The prediction reward system now allows overwriting existing predictions. You can use this functionality in your code by setting the `overwrite` parameter to `True` when calling the `save_prediction` method:

```python
from src.prediction_reward_system import PredictionRewardSystem

# Initialize the reward system
reward_system = PredictionRewardSystem(symbol='AAPL')

# Overwrite an existing prediction
reward_system.save_prediction(
    date='2023-10-15',
    predicted_price=150.75,
    model_version='manual_override',
    overwrite=True
)
```

When a prediction is overwritten:
1. The previous prediction value is stored in the 'notes' column
2. Accuracy is recalculated if actual data is available
3. The updated data is saved to CSV in the model directory for that stock (models/{symbol}/predictions.csv)

## Output

For each company, the script will:
1. Fetch historical stock data
2. Train or load a prediction model
3. Analyze market and sector trends
4. Adjust predictions based on market sentiment
5. Make predictions for future stock prices
6. Save visualizations in the `models/{symbol}/future_prediction.png` files

A progress tracker shows how many companies have been processed and the success rate. The script displays live output from the training and prediction process for each company.

Failed companies will be saved to `failed_companies.txt` for later retry.

## Market Trend Analysis

The system now incorporates market trend analysis to enhance prediction accuracy. It:

1. Fetches market index data (KSE100 by default)
2. Analyzes market direction (bullish/bearish)
3. Calculates market strength, momentum, and volatility
4. Generates a market sentiment score (-1 to +1)
5. Adjusts predictions based on overall market sentiment

You can control the impact of market trends with the `--market_adjustment` parameter (default: 0.03):
- Higher values (e.g., 0.05-0.10) make predictions more sensitive to market trends
- Lower values (e.g., 0.01-0.02) make predictions less sensitive to market trends
- Setting to 0 or using `--no_market_trends` disables market trend analysis

### Fallback Behavior

If KSE100 market data is unavailable:
- The system will automatically detect this and skip market trend analysis
- Predictions will continue without market adjustments
- A warning message will be displayed indicating market data couldn't be retrieved
- The program will run as if `--no_market_trends` flag was used

This ensures the system remains robust and can still make stock predictions even when market index data is temporarily unavailable.

## API Source

The list of Islamic companies is fetched from:
```
https://stocks.wajipk.com/api/companies?type=islamic
```

## Project Structure

- `src/`: Source code for the project
  - `data_collection.py`: Scripts for fetching historical stock data
  - `preprocessing.py`: Data preprocessing and feature engineering
  - `model.py`: LSTM model definition and training
  - `predict.py`: Using trained models for prediction
  - `rules.py`: Financial rules for adjusting stock prices
  - `market_analysis.py`: Market trend analysis and sentiment scoring
- `data/`: Stored data files
- `models/`: Trained models and visualizations
- `notebooks/`: Jupyter notebooks for exploration and analysis

## Running the Pipeline

```bash
# Full pipeline (data fetching, training, and prediction)
python main.py --symbol MARI --epochs 50

# With market trend analysis (higher adjustment factor)
python main.py --symbol MARI --market_adjustment 0.05

# Without market trend analysis
python main.py --symbol MARI --no_market_trends

# Test predictions only
python test_prediction.py

# Test dividend data integration
python test_payouts_api.py MARI
```

## Financial Rules

This model incorporates financial rules to adjust stock prices for dividend events. 
For more details on the financial rules system, see [README_FINANCIAL_RULES.md](README_FINANCIAL_RULES.md).

## Requirements

- Python 3.10+
- TensorFlow 2.10+
- Pandas, NumPy, Matplotlib
- scikit-learn
- Requests (for API calls)
- beautifulsoup4 (for web scraping market news)
- yfinance (alternative data source)

## Features

- Historical stock data retrieval with technical indicators
- Advanced deep learning model architecture for accurate prediction
- Technical indicator-based data enrichment
- Multiple prediction horizons
- Visualization of predictions with confidence intervals
- Identification of significant price movements
- Market trend analysis for enhanced predictions
- Prediction reward system for continuous model improvement
- Support for custom model parameters and training configurations

## New Feature: Prediction Reward System

The stock prediction model now includes a reward system that tracks the accuracy of predictions over time and helps the model learn from past predictions:

- Stores predictions in a CSV file with dates
- Evaluates prediction accuracy when actual prices become available
- Adjusts the model's approach when accuracy is below threshold
- Avoids duplicate predictions for the same date
- Provides suggestions for model parameter adjustments based on historical performance

### How the Reward System Works

1. **Prediction Tracking**: When a prediction is made, it's saved along with the date in a CSV file.

2. **Accuracy Evaluation**: During subsequent runs, the system checks for existing predictions and updates them with actual prices when available.

3. **Model Adjustment**: If prediction accuracy is below the specified threshold, the system will suggest adjustments to model parameters like learning rate, window size, and smoothing factor.

4. **Performance Analysis**: The system provides analytics about prediction accuracy over time, including trend analysis and success rate.

### Command Line Parameters

| Parameter | Description |
|-----------|-------------|
| `--reward_threshold` | Threshold for determining if a prediction is accurate (default: 0.05 or 5%) |
| `--no_reward_system` | Skip using the reward system |

### Example

```bash
python main.py --symbol AAPL --reward_threshold 0.03 
```

This will run the pipeline with the reward system enabled and set the accuracy threshold to 3%.
