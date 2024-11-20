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

- pandas
- numpy
- scikit-learn
- tensorflow
- matplotlib