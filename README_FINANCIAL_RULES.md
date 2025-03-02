# Financial Rules for Stock Price Prediction

This document outlines the financial rules applied to adjust stock prices in our prediction model. These adjustments are crucial for accurate model training and prediction as they account for corporate actions that affect stock prices but do not reflect actual changes in company value.

## Implemented Rules

### 1. Dividend Adjustments

When a company distributes dividends to shareholders, the stock price typically drops by the dividend amount. Our model adjusts historical prices before the dividend date by subtracting the dividend amount:

```
Price before dividend = Original price - Dividend amount (PKR)
```

This adjustment prevents the model from interpreting dividend-related price drops as negative trends in the stock's value.

### 2. Bonus Shares Adjustments (For Future Implementation)

Our system includes code for bonus share adjustments, though this feature is not currently active in the API integration. When implemented, the formula will be:

```
Adjusted price = Current price / (1 + Bonus ratio)
```

For example, if a company issues a 3:1 bonus with a pre-bonus price of 100 PKR, the adjusted price would be:
```
100 / (1 + 3/1) = 100 / 4 = 25 PKR
```

### 3. Pakistan Stock Exchange (PSX) Price Limit Rules

Our prediction model now implements the PSX daily price limit rules:

- **Upper cap**: The maximum a stock can increase in one day is either 10% of the previous day's closing price or 1 PKR, whichever is higher.
- **Lower cap**: The maximum a stock can decrease in one day is either 10% of the previous day's closing price or 1 PKR, whichever is higher in absolute terms.

These rules are applied to all predictions to ensure they comply with the PSX trading rules, and therefore represent more realistic price movements.

```
Upper limit = max(previous_price * 1.10, previous_price + 1)
Lower limit = max(previous_price * 0.90, previous_price - 1)
```

## How Rules Are Applied

The system applies financial adjustments in three ways:

### 1. Automatic API Retrieval (Currently Dividends Only)

The system automatically fetches dividend payout records from the API:
```
https://stocks.wajipk.com/api/payouts?symbol=SYMBOL
```

This API returns payout information with the following fields:
- `companyid`: Identifier for the company
- `symbol`: Stock symbol
- `announcedate`: Date the payout was announced
- `spotdate`: Date of spot trading
- `xdate`: Ex-dividend date (when price adjusts)
- `bookclosure`: Book closure date
- `facevalue`: Face value of the share
- `percentage`: Face value percentage of the dividend
- `dividend`: Actual dividend amount in PKR

The system uses the `xdate` (if available) or `announcedate` to determine when to apply price adjustments, and uses the `dividend` field for the amount to adjust.

### 2. Local Configuration File

As a fallback or supplement, the system also loads corporate actions from `data/corporate_actions.csv` which contains records in the format:
```
symbol,date,action_type,value
MARI,2023-01-15,dividend,2.5
MARI,2023-05-10,bonus,3:1
```

This file can contain both dividend and bonus share information, though only dividend information is currently used from the API.

### 3. Data Sorting and Adjustment

Before applying any rules, the data is sorted chronologically by date to ensure proper application of adjustments. The rules are automatically applied during data loading in both training and prediction pipelines unless the `--no_rules` flag is used.

## Adding New Manual Rules

For additional corporate actions not available through the API:

1. Edit the `data/corporate_actions.csv` file to add new events.
2. For new types of corporate actions, update the `src/rules.py` file to handle the new action type.

## Extending the Rule System

Future rule additions may include:

1. **Bonus Shares**: Adding API integration for bonus share information when available.
2. **Stock Splits**: Similar to bonus shares but with a different mechanism.
3. **Rights Issues**: When companies raise additional capital by offering existing shareholders the right to buy new shares.
4. **Mergers and Acquisitions**: Special adjustments for when companies merge or are acquired.

## Bypassing Rules

For testing or other purposes, you can bypass the application of financial rules by using the `--no_rules` flag:

```bash
python main.py --symbol MARI --epochs 50 --no_rules
```

This will run the pipeline without applying any financial adjustments to the stock data. 