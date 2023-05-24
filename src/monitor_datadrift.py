# ------------------------ Libraries & Credentials ------------------------ #
# Others
# ==============================================================================
import os
import boto3
from io import StringIO
import datetime
from datetime import date
from prefect import flow,task
from prefect.task_runners import SequentialTaskRunner
from prefect.deployments import Deployment
from prefect.filesystems import S3
from datetime import datetime

# Data manipulation
# ==============================================================================
import numpy as np
import pandas as pd

# Monitoring
# ==============================================================================
import evidently
from evidently.report import Report
from evidently.metrics import DataDriftTable

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
    data= data.groupby([ pd.Grouper(key='Date', freq = 'W-MON', label='left',closed='left')])['Diesel'].mean()

    return data

# Report evidently
@task
def check_data_drift(data):

    # Remove seasonality and trend by differencing Technique (by 1 week)
    datadiff=data-data.shift(1)
    datadiff = datadiff.dropna()
    datadiff = datadiff.to_frame()

    # Kolmogorov–Smirnov (K-S) test to evaluate data drift ()
    data_drift_report = Report(metrics=[
        DataDriftTable(num_stattest='ks'),
    ])

    reference_data = int(datadiff.shape[0]*0.95)
    data_drift_report.run(reference_data=datadiff[:reference_data], current_data=datadiff[reference_data:])
    data_drift_report = data_drift_report.as_dict()

    drift = data_drift_report['metrics'][0]['result']['dataset_drift']
    drift_score = data_drift_report['metrics'][0]['result']['drift_by_columns']['Diesel']['drift_score']

    return(drift_score)

# Main function
@flow
def pipeline():
    data = download_s3('gas-prices-project','data.csv')
    data = preprocess_fuelprice(data)
    drift_score = check_data_drift(data)

    return(drift_score)

# ------------------------ WORKFLOW ------------------------ #
if __name__ == "__main__":
    print(pipeline())
