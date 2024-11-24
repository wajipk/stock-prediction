# stock-prediction

modelfile (folder) (all files of model or trained model)
scripts
  - loaddata.py (responsible for loading data + preprocessing)
      - loadstockdata()
  - model.py (responsible for model loading)
      - return model class
  - server.py (all apis for comunication to server)
  - training.py (for training the model)
  - prediction.py (for prediction)
      - this file will be called form laravel with paramaters (symbol)
      - return {day1='Share Price of tomorrow',day2='Share Price after 2 days',day3='Share Price after 3 days',day4='Share Price after 4 days',day5='Share Price after 5 days',week1='Share Price after 1 week',week2='Share Price after 2 weeks',month1='Share Price after 1 month',month2='Share Price after 2 months',quarterly='Share Price after 3 months',halfyear='Share Price after 6 months',year='Share Price after 1 year'}

1. Load data (CSV: column: companyid, name, symbol, date (dd/mm/yyyy), close, volume)
2. Define model
3. evalution (back data testing)

- loaddata.py
   - Class - Stock Data
     - Method Load Stcok Data
       - Paramter Start Date
       - By default 5 year
       - Save Clean Data

- model.py
  - Model Architecture

- index.py
    - Parameter symbol for prediction
    - Parameter preprocessing for cleaning data by default 0
    - Training 1/0 by default 0
    - Return data in json form
    - Parameter plot graph by default 0 for plotting prediction price and previous price 

### Required Libraries

pip install pandas numpy scikit-learn matplotlib
pip install tensorflow==2.10.0

python index.py --train=1 --preprocess=1 --symbol=MARI --plot=1

Improvements
- [] Data File should be passed from index.py to data preprocessing

- Dask (https://www.dask.org/) allows for parallel and out-of-core computation, handling datasets larger than available RAM.  I will also add logging statements to track progress. Processing is also sequential; each step operates on the entire DataFrame.  Parallel processing or chunking could significantly improve performance.

- Attention Mechanism: Incorporate an attention mechanism. Attention allows the model to focus on the most relevant parts of the input sequence, improving performance, especially for long sequences.

Regularization: While dropout is used, consider adding L1 or L2 regularization to the dense layers to further prevent overfitting.

Bollinger Bands: Add Bollinger Bands to measure price volatility and potential reversals.

Relative Strength Index (RSI) variations: Experiment with different RSI periods (e.g., 10-period, 20-period RSI) or variations of the RSI calculation.

Moving Average Convergence Divergence (MACD) variations: Explore different MACD settings (e.g., different fast and slow EMA periods) or use the MACD histogram as a separate feature.

Average True Range (ATR): Incorporate ATR to measure market volatility.

Volume-based indicators: Add volume-weighted moving averages or other volume-based indicators to capture the relationship between price and volume.



Hyperparameter Tuning: The hyperparameters (seq_length, batch_size, epochs) are hardcoded.  A more robust approach would involve hyperparameter tuning using techniques like grid search, random search, or Bayesian optimization. This would help find optimal settings for the model.

Early Stopping: The training runs for a fixed number of epochs.  Adding early stopping based on a validation metric (e.g., MAE or MSE) would prevent overfitting and improve generalization.

Learning Rate Scheduling:  Using a constant learning rate throughout training might not be optimal.  Consider using a learning rate scheduler (e.g., ReduceLROnPlateau) to adjust the learning rate dynamically based on the training progress.

Data Augmentation:  To improve model robustness and generalization, consider data augmentation techniques.  This could involve adding noise to the input data or creating synthetic data points.

Different Model Architectures: Explore alternative model architectures, such as different types of recurrent neural networks (RNNs) (e.g., GRU) or convolutional neural networks (CNNs) combined with RNNs.  These might capture different patterns in the data and lead to better performance.

 Evaluation Metrics: While MAE is used, consider adding other relevant evaluation metrics like RMSE (Root Mean Squared Error) and R-squared to get a more comprehensive evaluation of the model's performance.


