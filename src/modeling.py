# ------------------------ Libraries & Credentials ------------------------ #
# Others
# ==============================================================================
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import os
import boto3
from io import StringIO
import datetime
from datetime import date
import mlflow
from IPython.display import display
import itertools
from prefect import flow,task
from prefect.task_runners import SequentialTaskRunner
from prefect.deployments import Deployment
from prefect.filesystems import S3

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
from sklearn.metrics import mean_absolute_percentage_error

# Credentials and configuration
# ==============================================================================
# Check if the code is run locally to set up the environment configuration
@task
def where_am_i():
    hostname=os.popen('hostname').read()
    desktop = "DESKTOP"

    if desktop in hostname:
        os.environ["AWS_PROFILE"] = "mlops" # fill in with your AWS profile.
        os.environ['AWS_DEFAULT_REGION'] = "eu-west-1"

# ------------------------ FUNCTIONS ------------------------ #
# Download Data
@task
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
@task
def preprocess_fuelprice(data):
    data['Diesel'] = data['Diesel'] .astype(float)
    data['Date'] = pd.to_datetime(data['Date'],format="%Y-%m-%d")
    data.sort_values(by='Date', inplace = True) 
    #data.drop_duplicates(data, inplace = True)
    data= data.groupby([ pd.Grouper(key='Date', freq = 'W-MON')])['Diesel'].mean()

    return data

# Split in train/test
@task
def split_train_test(data, days_test):
    train = data[:-days_test]
    test = data[-days_test:]

    return train,test

# MLFlow: Modeling SARIMA
@task
def train_sarimax_model_mlflow(train,test, run_name):

    # Paramters
    p = range(0,2)
    d = range(0,1)
    q = range(0,1)

    P = range(0,1)
    D = range(0,1)
    Q = range(0,1)

    parameters = itertools.product(p, d, q, P, D, Q)
    parameters_list = list(parameters)
    len(parameters_list)

    # Model loop
    for param in parameters_list:
        with mlflow.start_run(run_name=run_name):

            # Log 
            mlflow.set_tag("model", "SARIMA")
            mlflow.log_param('order-p', param[0])
            mlflow.log_param('order-d', param[1])
            mlflow.log_param('order-q', param[2])
            mlflow.log_param('order-P', param[3])
            mlflow.log_param('order-D', param[4])
            mlflow.log_param('order-Q', param[5])     
            
            # SARIMAX
            try:
                model_sarimax = SARIMAX.SARIMAX(train, order=(param[0], param[1],param[2]), seasonal_order=(param[3], param[4], param[5], 52))

            except ValueError:
                print('bad parameter combination:', param)    
                            
                continue
            results = model_sarimax.fit()
            start=len(train)
            end=len(train)+len(test)-1
            predictions = results.predict(start=start, end=end, dynamic=False)

            # metrics
            rmse_ = rmse(test, predictions) 
            mape_ = mean_absolute_percentage_error(test, predictions)   
            ACI_ = results.aic
                    
            mlflow.log_metric("rmse", rmse_)  
            mlflow.log_metric("mape", mape_) 
            mlflow.log_metric("ACI", ACI_) 

            # model
            #mlflow.statsmodels.log_model(results, artifact_path = "model")
            mlflow.sklearn.log_model(results, artifact_path = "model")


# Main function
@flow
def pipeline():
    # mlflow
    TRACKING_SERVER_HOST = "ec2-52-17-129-153.eu-west-1.compute.amazonaws.com" # fill in with the public DNS of the EC2 instance
    mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")
    mlflow.set_experiment("SARIMA_param")

    data = download_s3('gas-prices-project','data.csv')
    data = preprocess_fuelprice(data)
    train,test = split_train_test(data, 4)
    train_sarimax_model_mlflow(train,test, "SARIMA_param")

# ------------------------ WORKFLOW ------------------------ #
if __name__ == "__main__":
    pipeline()