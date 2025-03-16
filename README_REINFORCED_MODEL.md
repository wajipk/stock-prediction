# Reinforced Learning for Stock Price Prediction

This document explains how to use the reinforced learning model for stock price prediction, which can adapt to improve its accuracy over time by learning from its past prediction errors.

## Overview

The reinforced stock model enhances traditional prediction models by adding an error correction component that learns from the differences between predicted and actual prices. This allows the model to:

1. Make predictions using a base model (LSTM, BiLSTM, GRU, CNN-LSTM, or advanced hybrid model)
2. Track prediction accuracy over time using a reward system
3. Adapt and learn from its mistakes through a dedicated error correction model
4. Apply learned corrections to future predictions

This approach is especially useful for symbols where a standard model may achieve decent accuracy (such as 80%) but needs further refinement to handle specific patterns or behaviors for that particular stock.

## Using the Reinforced Model

### Command Line Usage

To use the reinforced model from the command line:

```bash
python main.py --symbol AAPL --use_reinforced_model --base_model_type lstm --error_threshold 0.05
```

#### Key Parameters:

- `--use_reinforced_model`: Activates the reinforced learning approach
- `--base_model_type`: Specifies the base model architecture (`lstm`, `bilstm`, `gru`, `cnn_lstm`, or `advanced`)
- `--error_threshold`: Sets the error threshold for determining when adaptation is needed (default: 0.05, or 5%)
- `--window`: Window size for sequential data (default: 10)
- `--epochs`: Training epochs (default: 100)
- `--batch_size`: Batch size for training (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)
- `--dropout`: Dropout rate for regularization (default: 0.4)

### Running the Example Script Directly

You can also run the example script directly:

```bash
python src/reinforced_prediction_example.py --symbol AAPL --base_model lstm --epochs 50
```

## How the Reinforced Model Works

1. **Base Model**: A standard prediction model (LSTM, BiLSTM, etc.) is trained on historical data
2. **Reward System**: Predictions are tracked along with actual prices when they become available
3. **Error Correction Model**: A second model learns to predict and correct the errors made by the base model
4. **Adaptation**: The model periodically retrains the error correction component using recent prediction errors
5. **Reinforced Prediction**: New predictions combine the base prediction with learned error corrections

## Implementation Details

### ReinforcedStockModel Class

The core of the implementation is the `ReinforcedStockModel` class in `src/reinforced_model.py`. This class:

- Maintains both a base prediction model and an error correction model
- Tracks prediction history and errors
- Automatically adapts to improve accuracy over time
- Provides functions for training, predicting, and updating with actual prices

### Key Methods:

- `train_base_model()`: Trains the base prediction model
- `predict()`: Makes predictions, applying error corrections if available
- `update_with_actual_price()`: Updates the model with actual prices when they become available
- `adapt_to_errors()`: Retrains the error correction model based on recent prediction errors

## Performance and Adaptation

The model's performance improves over time as it collects more actual prices and learns from its errors. Key performance indicators are tracked automatically:

- **Prediction Accuracy**: How close predictions are to actual prices
- **Error Patterns**: Systematic errors that the model can learn to correct
- **Adaptation Effect**: How much the error correction improves predictions

## Advanced Usage

For advanced usage, you can directly use the `ReinforcedStockModel` class in your own code:

```python
from src.reinforced_model import ReinforcedStockModel

# Create a reinforced model
model = ReinforcedStockModel(
    symbol='AAPL',
    window_size=10,
    base_model_type='lstm',
    learning_rate=0.001,
    dropout_rate=0.3,
    error_threshold=0.05
)

# Train the model
model.train_base_model(X_train, y_train, X_val, y_val, epochs=100, batch_size=32)

# Make a prediction
prediction = model.predict(X_new, apply_correction=True)

# Update with actual price when available
model.update_with_actual_price('2023-06-15', 185.27)
```

## Best Practices

1. **Data Quality**: Ensure high-quality historical data with appropriate technical indicators
2. **Initial Training**: Start with a well-trained base model for best results
3. **Regular Updates**: Consistently update the model with actual prices when they become available
4. **Track Metrics**: Monitor the model's improvement over time using the reward system metrics
5. **Patience**: Allow the model to collect enough error data before expecting significant improvements 