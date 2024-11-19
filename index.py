# Parameter symbol for prediction
# Parameter preprocessing for cleaning data by default 0
# Training 1/0 by default 0
# Return data in json form
# Parameter plot graph by default 0 for plotting prediction price and previous price 

import argparse
from training import ModelTrainer
from prediction import StockPredictor
from loaddata import StockData
import matplotlib.pyplot as plt
import json


def main():
    parser = argparse.ArgumentParser(description="Stock Prediction System")
    parser.add_argument("--train", type=int, default=0, choices=[0, 1], help="1 for training, 0 for prediction")
    parser.add_argument("--preprocess", type=int, default=0, choices=[0, 1], help="1 for preprocessing data")
    parser.add_argument("--symbol", type=str, help="Stock symbol for prediction")
    parser.add_argument("--plot", type=int, default=0, choices=[0, 1], help="1 to plot graph")
    
    args = parser.parse_args()

    if args.preprocess == 1:
        stock_data = StockData()
        stock_data.load_stock_data(use_incremental=False)

    if args.train == 1:
        trainer = ModelTrainer()
        trainer.train()

    if args.symbol:
        predictor = StockPredictor()
        predictions = predictor.predict_all_periods(args.symbol)

        # Output predictions as JSON
        print(json.dumps(predictions, indent=4))

        if args.plot == 1:
            dates = [prediction['prediction_date'] for prediction in predictions.values()]
            prices = [prediction['predicted_price'] for prediction in predictions.values()]

            plt.plot(dates, prices)
            plt.xlabel('Prediction Date')
            plt.ylabel('Predicted Price')
            plt.title(f'Predictions for {args.symbol}')
            plt.xticks(rotation=45)
            plt.show()


if __name__ == "__main__":
    main()
