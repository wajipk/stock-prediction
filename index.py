import argparse
from training import ModelTrainer
from prediction import StockPredictor
from loaddata import StockData
import matplotlib.pyplot as plt
import json


def main():
    parser = argparse.ArgumentParser(description="Stock Prediction System")
    parser.add_argument("--train", type=int, default=0, choices=[0, 1], help="1 for training, 0 for prediction")
    parser.add_argument("--preprocess", type=int, default=0, choices=[0, 1], help="1 to preprocess data")
    parser.add_argument("--symbol", type=str, help="Stock symbol for prediction")
    parser.add_argument("--plot", type=int, default=0, choices=[0, 1], help="1 to plot predictions")

    args = parser.parse_args()

    # Data preprocessing step
    if args.preprocess == 1:
        print("Starting data preprocessing...")
        stock_data = StockData()
        # Preprocess the data (clean, calculate technical indicators, and generate labels)
        stock_data.preprocess_data()
        print("Data preprocessing completed.")

    # Model training step
    if args.train == 1:
        print("Starting model training...")
        trainer = ModelTrainer()
        trainer.train()
        print("Model training completed.")

    # Prediction step
    if args.symbol:
        print(f"Predicting for stock symbol: {args.symbol}...")
        predictor = StockPredictor()
        predictions = predictor.predict_all_periods(args.symbol)
        print(json.dumps(predictions, indent=4))

        # If plotting is enabled, generate and display the prediction plot
        if args.plot == 1:
            print("Generating prediction plot...")
            dates = [prediction['date'] for prediction in predictions.values()]
            prices = [prediction['price'] for prediction in predictions.values()]

            plt.plot(dates, prices)
            plt.xlabel('Prediction Date')
            plt.ylabel('Predicted Price')
            plt.title(f'Predictions for {args.symbol}')
            plt.xticks(rotation=45)
            plt.show()
            print("Prediction plot displayed.")


if __name__ == "__main__":
    main()
