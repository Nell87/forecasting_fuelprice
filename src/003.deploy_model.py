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

# Load model in production (mlflow)
@task
def load_model_production(bucket):
    # set up the S3 client
    s3 = boto3.client('s3')

    # specify the name of the bucket and the file path
    logged_model = 's3://gas-prices-project/production/'
    model = mlflow.sklearn.load_model(logged_model)
    
    return model

# Deploy model
@task
def deploy_model():
    config=dict(region_name="eu-west-1", execution_role_arn = "arn:aws:iam::331069391452:role/EC2-S3-Access",
    image_url = "331069391452.dkr.ecr.eu-west-1.amazonaws.com/mlflow-pyfunc:2.1.1")
    location = 's3://gas-prices-project/production'
    mlflow.models.build_docker(model_uri=location, name='mlflow-pyfunc')
   
    client = get_deploy_client("sagemaker")
    client.create_deployment(
        name = "test", 
        model_uri = location,
        config = config)
    
# Main function
@flow
def pipeline():
    # mlflow
    # Set client
    exp_name =  "SARIMA_param"
    MLFLOW_TRACKING_URI = mlflow.set_tracking_uri("http://ec2-52-17-129-153.eu-west-1.compute.amazonaws.com:5000")
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    mlflow.set_experiment(exp_name)

    model = load_model_production('gas-prices-project')
    deploy_model()

# ------------------------ WORKFLOW ------------------------ #
if __name__ == "__main__":
    pipeline()