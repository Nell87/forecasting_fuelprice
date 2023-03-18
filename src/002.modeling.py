# ------------------------ Libraries & Credentials ------------------------ #
# Others
# ==============================================================================
import mlflow
import os
import boto3
from io import StringIO
import datetime
from datetime import date
import mlflow
from IPython.display import display

# Data manipulation
# ==============================================================================
import numpy as np
import pandas as pd

# Modeling and Forecasting
# ==============================================================================
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
import statsmodels.tsa.arima as ARIMA
import statsmodels.tsa.statespace.sarimax as SARIMAX
from statsmodels.tools.eval_measures import rmse

# Credentials AWS
os.environ["AWS_PROFILE"] = ("mlops") # fill in with your AWS profile. 

# ------------------------ FUNCTIONS ------------------------ #
# Download Data
def download_s3(bucket, data):
    client = boto3.client('s3')
    bucket_name = bucket
    object_key = data
    csv_obj = client.get_object(Bucket=bucket_name, Key=object_key)
    body = csv_obj['Body']
    csv_string = body.read().decode('utf-8')

    data = pd.read_csv(StringIO(csv_string))
    return data

# Preprocess data
def preprocess_fuelprice(data):
    data['Diesel'] = data['Diesel'] .astype(float)
    data['Date'] = pd.to_datetime(data['Date'],format="%Y-%m-%d")
    data.sort_values(by='Date', inplace = True) 
    data.drop_duplicates(data, inplace = True)
    data= data.groupby([ pd.Grouper(key='Date', freq = 'W-MON')])['Diesel'].mean()

    return data

# Split in train/test
def split_train_test(data, days_test):
    train = data[:-days_test]
    test = data[-days_test:]

    return train,test

# ------------------------ WORKFLOW ------------------------ #
# Download, preprocess and split data
data = download_s3('gas-prices-project','data.csv')
data = preprocess_fuelprice(data)
train,test = split_train_test(data, 4)

# Modeling
# MLFlow 
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("SARIMAX")

with mlflow.start_run(run_name='arima_param'):

    q = 1
    d = 1
    p = 1

    mlflow.set_tag("developer", "sara")
    mlflow.set_tag("model", "SARIMAX")
    mlflow.log_param("q", q)
    mlflow.log_param("d", q)
    mlflow.log_param("p", q)

    model_sarimax = SARIMAX.SARIMAX(train,order=(p,d,q),enforce_invertibility=False)
    results = model_sarimax.fit()

    start=len(train)
    end=len(train)+len(test)-1

    predictions = results.predict(start=start, end=end, dynamic=False)
    rmse_ = rmse(test, predictions)
    mlflow.log_metric("rmse", rmse_)    