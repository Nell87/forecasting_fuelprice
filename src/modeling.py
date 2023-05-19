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

# MLFlow: Registry & Production best model
@task
def register_best_model(exp_name):
    # Set client
    MLFLOW_TRACKING_URI = mlflow.set_tracking_uri("http://ec2-52-17-129-153.eu-west-1.compute.amazonaws.com:5000")
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    mlflow.set_experiment(exp_name)
   
    # Get experiment id
    current_experiment=dict(mlflow.get_experiment_by_name(exp_name))
    experiment_id=current_experiment['experiment_id']

    # Select model with the lowest RMSE
    runs = client.search_runs(
        experiment_ids= experiment_id,
        run_view_type= ViewType.ACTIVE_ONLY,
        max_results= 1,
        order_by=["metrics.rmse ASC"]
    )

    newmodel_run_id = runs[0].info.run_id    

    # Model to compare: new one vs old one
    model_name = "fuel-price-predict"
    rmse_new = runs[0].data.metrics['rmse']
    tags = {"rmse": rmse_new}
    model_uri = f"runs:/{newmodel_run_id}/model"

    registered_models = client.search_registered_models()
    
    # If there is a model in production
    if len(registered_models) >0:
        prod_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/Production") 
        prod_model_id = prod_model.metadata.run_id
        prod_last_version = client.get_latest_versions(model_name, stages = ["Production"])[0].version
        rmse_prod = client.get_metric_history(prod_model_id, "rmse")[0].value
    
        if(rmse_prod > rmse_new):
            mlflow.register_model(model_uri=model_uri, name=model_name, tags=tags)
            new_last_version = client.get_latest_versions(model_name, stages = ["None"])[0].version
            
            client.transition_model_version_stage(
                name=model_name, version=new_last_version, stage="Production"            
            )
            
            client.transition_model_version_stage(
                name=model_name, version=prod_last_version, stage="Archived"
            )

    # If there is no model in production
    else:
        model_name = "fuel-price-predict"
        mlflow.register_model(model_uri=model_uri, name=model_name, tags=tags)
        new_last_version = client.get_latest_versions(model_name, stages = ["None"])[0].version        
        client.transition_model_version_stage(
            name=model_name, version=new_last_version, stage="Production"            
        )

# MLFlow & S3: Delete no register models
@task
def delete_no_registered_models_s3(exp_name, bucket_name, subfolder):
   # Set client
    MLFLOW_TRACKING_URI = mlflow.set_tracking_uri("http://ec2-52-17-129-153.eu-west-1.compute.amazonaws.com:5000")
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # Get experiment id
    current_experiment=dict(mlflow.get_experiment_by_name(exp_name))
    experiment_id=current_experiment['experiment_id']

    # Registered models id list
    registered_models=[]
    registered_models_id =[]

    for rm in client.search_registered_models():
        registered_models.append(dict(rm)["latest_versions"])

    registered_models= registered_models[0]

    for model in range(len(registered_models)):
        registered_models_id.append(registered_models[model].run_id)

    # Delete model from S3
    subfolder_path = f'{subfolder}/{experiment_id}/'
    s3 = boto3.resource('s3')

    # Iterate through the list of objects in the subfolder
    for obj in s3.Bucket(bucket_name).objects.filter(Prefix=subfolder_path):
        if not any(word in obj.key for word in registered_models_id):
            # Object is a subfolder and does not contain any
            s3.Object(bucket_name, obj.key).delete()

@task
def model_prod_s3(exp_name, bucket):
    MLFLOW_TRACKING_URI = mlflow.set_tracking_uri("http://ec2-52-17-129-153.eu-west-1.compute.amazonaws.com:5000")
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # DELETE THE OLD MODEL
    # set up the S3 client
    s3_resource = boto3.resource('s3')

    # check if the folder is empty
    bucket_ = s3_resource.Bucket(bucket)
    objects = list(bucket_.objects.filter(Prefix="production/"))    

    if len(objects) > 0:
        # Iterate through all objects in the folder and delete them
        for obj in objects:
            s3_resource.Object(bucket, obj.key).delete()  

    # ADD THE NEW MODEL
    # Get experiment id
    current_experiment=dict(mlflow.get_experiment_by_name(exp_name))
    experiment_id=current_experiment['experiment_id']

    # Get run_id from production model
    registered_models = client.search_registered_models()
    model_production_id =registered_models[0].latest_versions["current_stage"=="Production"].run_id

    # Define the source and destination subfolders and filenames
    bucket_ = s3_resource.Bucket(bucket)
    source_subfolder = f'mlflow/{experiment_id}/{model_production_id}/artifacts/model'
    destination_subfolder = 'production'

    # Copy the file to the destination subfolder
    s3_client = boto3.client('s3')

    for obj in bucket_.objects.filter(Prefix=source_subfolder):
        # Create new key name by replacing source folder name with destination folder name
        new_key = obj.key.replace(source_subfolder, destination_subfolder, 1)

        # Copy the object to the new key in the destination folder
        s3_resource.Object(bucket, new_key).copy_from(CopySource={'Bucket': bucket, 'Key': obj.key})
  
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
    register_best_model(exp_name = "SARIMA_param")
    delete_no_registered_models_s3("SARIMA_param", 'gas-prices-project', 'mlflow')
    model_prod_s3(exp_name = "SARIMA_param", bucket= "gas-prices-project")

# ------------------------ WORKFLOW ------------------------ #
if __name__ == "__main__":
    pipeline()