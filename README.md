# Stock Prediction System

This system predicts stock prices for multiple companies using historical data, machine learning models, and market trend analysis.

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
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

- Advanced deep learning model using multi-branch architecture (CNN, LSTM, GRU)
- Technical indicators calculation
- Adjustment for corporate actions (dividends, splits, etc.)
- Market trend analysis and adjustment
- Prediction confidence intervals
- Self-learning reward system that improves over time
- Reality check validation for predictions
- Support for both legacy and advanced models
- Detailed visualization of predictions

## Self-Learning Prediction System

The stock prediction model includes a reward system that tracks prediction accuracy over time to improve future forecasts:

1. When predictions are made, they are stored in a history file with the predicted price
2. During the next training cycle, previous predictions are automatically compared with actual prices
3. The model learns from its mistakes by giving higher weight to training examples where it performed poorly
4. Over time, the model adjusts its parameters based on observed prediction accuracy
5. The reward system can also suggest adjustments to hyperparameters like learning rate

This creates a feedback loop that allows the model to continuously improve its predictions based on real-world performance. You can run the test script to see this in action:

```bash
python test_update_predictions.py --symbol YOUR_STOCK_SYMBOL
```

## Usage

To run the pipeline with default settings:

```bash
python main.py --symbol AAPL
```

## Multi-Model Selection for Stock Prediction

The system now includes a powerful multi-model selection feature that automatically determines the best model for each stock:

### Available Models

The system can now train and evaluate multiple types of models for each stock:

#### Deep Learning Models:
- **LSTM Network**: Standard Long Short-Term Memory network
- **Bidirectional LSTM**: LSTM with bidirectional layers
- **GRU Network**: Gated Recurrent Unit network
- **CNN-LSTM Hybrid**: Convolutional Neural Network combined with LSTM
- **Advanced Multi-Branch Model**: Complex model combining CNN, LSTM, and GRU branches

#### Traditional Machine Learning Models:
- **Random Forest**: Ensemble learning method using decision trees
- **Gradient Boosting**: Boosting algorithm for regression
- **XGBoost**: Optimized gradient boosting implementation
- **LightGBM**: Gradient boosting framework using tree-based learning
- **Support Vector Regression**: Regression using support vector machines
- **Elastic Net**: Linear regression with L1 and L2 regularization

### How It Works

1. The system trains multiple model types on the same stock data
2. Each model is evaluated using several metrics (MSE, RMSE, MAE, MAPE, R²)
3. The best model is selected based on the chosen priority metric (default: MAPE)
4. A visualization comparing all models is generated
5. The best model is saved and used for future predictions for that stock

### Benefits

- Different stocks behave differently and require different modeling approaches
- Some stocks may follow linear patterns while others have complex non-linear relationships
- Traditional ML models may work better for stable, established companies
- Deep learning models often perform better for volatile or cyclical stocks
- This approach finds the optimal model for each individual stock

### Using Multi-Model Selection

The system automatically uses model selection by default to find the best model for each stock. You can customize the behavior with the following options:

```bash
# Model selection is enabled by default (no need for additional flags)
python main.py --symbol TSLA

# Try specific models only
python main.py --symbol MSFT --models_to_try lstm xgboost random_forest

# Change priority metric (default is mape)
python main.py --symbol AAPL --priority_metric r2

# Disable model selection and use only the default model
python main.py --symbol GOOGL --no_model_selection
```

The system will generate a visualization comparing all models in `models/{symbol}/{symbol}_model_comparison.png` and store information about the best model in `models/{symbol}/best_model_info.json`.
