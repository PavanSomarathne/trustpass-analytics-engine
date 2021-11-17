from flask import Flask, request, render_template, send_file
from flask_cors import CORS
import datetime as DT
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
import pandas as pd
import pickle

today = DT.date.today()

start_date = today + DT.timedelta(days=1)
end_date = today + DT.timedelta(days=90)

app = Flask(__name__)
CORS(app)
# loaded = ARIMAResults.load('trained_models/logins.pkl')
# loaded = pickle.load(open('app/trained_models/logins.pkl', 'rb'))
@app.route('/')
def index():
    try:
        output = {'logins':[]  }
        df = pd.read_csv('app/data/logins.csv', index_col='Date', parse_dates=True)
        df = df.dropna()

        loaded = ARIMAResults.load('app/models/logins.pkl')
        index_future_dates = pd.date_range(start=str(start_date), end=str(end_date))

        pred = loaded.predict(start=len(df), end=len(df) + 89, typ='levels').rename('ARIMA Predictions')
        pred.index = index_future_dates
        print(pred)
        return_value = []
        for i in range(0, 90):
            return_value.append(float(str(pred[i])[0:5]))
        output = {'logins':return_value  }
    except Exception as e:
        print(e)
    
    return output


@app.route('/purpose')
def purpose():
    try:
        output = {'banking':[],'education':[],'tourism':[]}
        df = pd.read_csv('app/data/purpose.csv', index_col='Date', parse_dates=True)
        df = df.dropna()

        banking = ARIMAResults.load('app/models/Banking.pkl')
        education = ARIMAResults.load('app/models/Education.pkl')
        tourism = ARIMAResults.load('app/models/Tourism.pkl')

        index_future_dates = pd.date_range(start=str(start_date), end=str(end_date))

        banking_pred = banking.predict(start=len(df), end=len(df) + 89, typ='levels').rename('ARIMA Predictions')
        education_pred = education.predict(start=len(df), end=len(df) + 89, typ='levels').rename('ARIMA Predictions')
        tourism_pred = tourism.predict(start=len(df), end=len(df) + 89, typ='levels').rename('ARIMA Predictions')

        banking_pred.index = index_future_dates
        education_pred.index = index_future_dates
        tourism_pred.index = index_future_dates

        return_banking = []
        return_education = []
        return_tourism = []

        for i in range(0, 90):
            return_banking.append(float(str(banking_pred[i])[0:5]))
            return_education.append(float(str(education_pred[i])[0:5]))
            return_tourism.append(float(str(tourism_pred[i])[0:5]))

        output = {'banking':return_banking,'education':return_education,'tourism':return_tourism}
    except Exception as e:
        print(e)
    return output


@app.route('/feedback')
def feedback():
    output = {'data':[]}
    df = pd.read_csv('app/data/customer_feedback.csv', index_col='Date', parse_dates=True)
    df = df.dropna()

    loaded = ARIMAResults.load('app/models/rate.pkl')
    index_future_dates = pd.date_range(start=str(start_date), end=str(end_date))

    pred = loaded.predict(start=len(df), end=len(df) + 89, typ='levels').rename('ARIMA Predictions')
    pred.index = index_future_dates
    return_value = []
    for i in range(0, 90):
        return_value.append(float(str(pred[i])[0:5]))

    output = {'data':return_value}
    return output


@app.route('/irregular')
def irregular():
    try:
        output = {'UID0001':0,'UID0002':0,'UID0003':0}
        df = pd.read_csv('app/data/user_data.csv', index_col='Date', parse_dates=True)
        df = df.dropna()
    
        UID0001 = ARIMAResults.load('app/models/UID0001.pkl')
        UID0002 = ARIMAResults.load('app/models/UID0002.pkl')
        UID0003 = ARIMAResults.load('app/models/UID0003.pkl')
    
        index_future_dates = pd.date_range(start=str(start_date), end=str(end_date))
    
        UID0001_pred = UID0001.predict(start=len(df), end=len(df) + 89, typ='levels').rename('ARIMA Predictions')
        UID0002_pred = UID0002.predict(start=len(df), end=len(df) + 89, typ='levels').rename('ARIMA Predictions')
        UID0003_pred = UID0003.predict(start=len(df), end=len(df) + 89, typ='levels').rename('ARIMA Predictions')
    
        UID0001_pred.index = index_future_dates
        UID0002_pred.index = index_future_dates
        UID0003_pred.index = index_future_dates
    
        return_UID0001 = []
        return_UID0002 = []
        return_UID0003 = []
    
        for i in range(0, 90):
            return_UID0001.append(float(str(UID0001_pred[i])[0:5]))
            return_UID0002.append(float(str(UID0002_pred[i])[0:5]))
            return_UID0003.append(float(str(UID0003_pred[i])[0:5]))
    
        output = {'UID0001':return_UID0001,'UID0002':return_UID0002,'UID0003':return_UID0003}
    except Exception as e:
        print(e)
    return output


if __name__ == '__main__':
    app.run(debug=True)
