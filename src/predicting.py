# ------------------------ Libraries & Credentials ------------------------ #
# Others
# ==============================================================================
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from mlflow.deployments import get_deploy_client
from mlflow.sagemaker import SageMakerDeploymentClient
import mlflow.sagemaker as mfs
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
import tempfile
import coremltools as ct
from datetime import datetime
import json

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

# Credentials and configuration
# ==============================================================================
os.environ["AWS_PROFILE"] = ("mlops") # fill in with your AWS profile. 
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
    data= data.groupby([ pd.Grouper(key='Date', freq = 'W-MON', label='left',closed='left')])['Diesel'].mean()

    return data

# Load model in production (mlflow)
@task
def load_model_production(bucket):
    # set up the S3 client
    s3 = boto3.client('s3')

    # specify the name of the bucket and the file path
    logged_model = 's3://gas-prices-project/production/'
    model = mlflow.sklearn.load_model(logged_model)
    
    return model

## Apply best model
@task
def apply_model(model, data):
    pred =  model.predict(start=len(data), end=len(data)+4)
    return pred

# Save predictions into the s3
@task
def upload_s3(bucket, new_data):  
    timestamp = datetime.now()
    str_date_time = timestamp.strftime("%d-%m-%Y, %H:%M:%S")

    timestamp = datetime.now()
    str_date_time = timestamp.strftime("%d-%m-%Y, %H:%M:%S")

    s3 = boto3.client('s3')
    csv_buffer = StringIO()
    new_data.to_csv(csv_buffer, index=False)
    bucket_name = bucket
    
    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket, f'predictions/predictions-'+str_date_time+'.csv').put(Body=csv_buffer.getvalue())

# Main function
@flow
def pipeline():
    global app_name
    global region
    app_name = 'test'
    region = 'eu-west-1'

    # mlflow
    # Set client
    exp_name =  "SARIMA_param"
    MLFLOW_TRACKING_URI = mlflow.set_tracking_uri("http://ec2-52-17-129-153.eu-west-1.compute.amazonaws.com:5000")
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    mlflow.set_experiment(exp_name)

    data = download_s3('gas-prices-project','data.csv')
    data = preprocess_fuelprice(data)
    model = load_model_production('gas-prices-project')
    predictions = apply_model(model, data)
    upload_s3('gas-prices-project', predictions)

# ------------------------ WORKFLOW ------------------------ #
if __name__ == "__main__":
    pipeline()