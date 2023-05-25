# Fuel price project
[![turn_on_ec2](https://github.com/Nell87/forecasting_fuelprice/actions/workflows/turn_on_ec2.yml/badge.svg)](https://github.com/Nell87/forecasting_fuelprice/actions/workflows/turn_on_ec2.yml)
[![linter-pylint](https://github.com/Nell87/forecasting_fuelprice/actions/workflows/linter.yaml/badge.svg)](https://github.com/Nell87/forecasting_fuelprice/actions/workflows/linter.yaml)
[![testing](https://github.com/Nell87/forecasting_fuelprice/actions/workflows/test.yaml/badge.svg)](https://github.com/Nell87/forecasting_fuelprice/actions/workflows/test.yaml)
[![mlops](https://github.com/Nell87/forecasting_fuelprice/actions/workflows/mlops.yaml/badge.svg)](https://github.com/Nell87/forecasting_fuelprice/actions/workflows/mlops.yaml)
[![turn_off_ec2](https://github.com/Nell87/forecasting_fuelprice/actions/workflows/turn_off_ec2.yaml/badge.svg)](https://github.com/Nell87/forecasting_fuelprice/actions/workflows/turn_off_ec2.yaml)

## Table of Contents
- [Objective](#objective)
- [Project structure](#project-structure)
- [Architecture](#architecture)
- [GitHub Actions](#github-actions)
- [Tools and technologies used](#tools-and-technologies-used)
- [To do list](#to-do-list)
- [Credits](#credits)

## Objective
The objective of the project is to build an end-to-end machine learning project that aims to predict fuel prices. This project involves all the necessary steps, starting from data collection, followed by training, deploying, and monitoring of the model.

## Project structure
The project is implemented on Ubuntu 22.04 using AWS. 

### Gathering data
In order to make predictions about future fuel prices, it is necessary for us to obtain the historical time series data that will be used to train our models. To accomplish this, we will run a Python script that scrapes the data from [fuelo.net](https://es.fuelo.net) for the previous days. The collected data will be combined with existing historical data and stored as a CSV file in an S3 storage.

The following functions have been used:
- The `fuel_scraper_daily` function is used to scrape the data
- The `upload_s3` function is used to upload new data to the s3
- The `merge_datasets_s3` is used to merge the old and new data in the s3

<details>
<summary>View code</summary>

```python
def fuel_scraper_daily(year, month):

    # Let's make a request to check the status
    response = requests.get('https://es.fuelo.net/calendar/month/' + str(year) +  "/" + str(month))
    status_code = response.status_code

    if status_code != 200:
        #print( "The status code is not 200")
        return pd.DataFrame()
    
    else:
        # Extract content
        soup = bs(response.content,'html.parser')

        # Prepare the dataframe
        df=pd.DataFrame(columns=["Day", "Diesel"])
        
        # Scraper
        i = 0
        for td in soup.table.find_all('td'):
            if td.text.find("DSL")>-1:
                pattern = " " + ".*"
                day = re.sub(pattern, '', td.get_text(strip=False) )
                pattern  = ".*" + "DSL:" 
                price = re.sub(pattern, '', td.get_text(strip=False) )
                pattern = "€/l" + ".*"
                price = re.sub(pattern, '', price )
                df.at[i, "Day"] = day
                df.at[i, "Diesel"] = price
                i = i+1

        # Add Date Column
        df['Date'] = pd.to_datetime(dict(year=year, month=month, day=df.Day))

        # Transform diesel column to float
        df['Diesel'] = df['Diesel'] .str.replace('[A-Za-z]', '').str.replace(',', '.').astype(float)

        # Remove week columns
        df = df.drop('Day', axis=1)

        # Reorganize columns
        df = df[['Date', 'Diesel']]
        
        return df

# Upload to S3
def upload_s3(bucket, new_data):
    s3 = boto3.client('s3')
    csv_buffer = StringIO()
    new_data.to_csv(csv_buffer, index=False)

    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket, 'new_data.csv').put(Body=csv_buffer.getvalue())

# Merge new data with original data
def merge_datasets_S3():
    bucket = 'gas-prices-project'
    filename_1 = 'data.csv'
    filename_2 = 'new_data.csv'
    
    s3 = boto3.client('s3')
    
    first_obj = s3.get_object(Bucket= bucket, Key= filename_1)
    second_obj = s3.get_object(Bucket= bucket, Key= filename_2)
    
    first_df = pd.read_csv(first_obj['Body'])
    second_df = pd.read_csv(second_obj['Body'])
    
    concat_data = pd.concat([first_df, second_df]) 
    concat_data = concat_data.drop_duplicates(subset=None, keep="first", inplace=False)
    concat_data = concat_data[concat_data['Diesel'] >0]

    return concat_data
  
```

</details>

### Modeling / Experiment tracking
So far, only ARIMA models with multiple parameters have been trained for this project (check next steps in the "To-do list"). MLFlow was used for experiment tracking and model registry. I have saved the tracking information in an Amazon Relational Database Service (RDS) and stored the best model inside an s3 bucket

The following functions have been used:
- The `download_s3` function is used to download the data from the s3 bucket
- The `preprocess_fuelprice` function is used to group the daily data into weekly data
- The `split_train_test` function is used to split the data
- The `train_arima_mlflow` function is used to train ARIMA models

<details>
<summary>View code</summary>

```python
def download_s3(bucket, data):
    client = boto3.client('s3')
    bucket_name = bucket
    object_key = data
    csv_obj = client.get_object(Bucket=bucket_name, Key=object_key)
    body = csv_obj['Body']
    csv_string = body.read().decode('utf-8')

    data = pd.read_csv(StringIO(csv_string))
    return data
    
 def preprocess_fuelprice(data):
    data['Diesel'] = data['Diesel'] .astype(float)
    data['Date'] = pd.to_datetime(data['Date'],format="%Y-%m-%d")
    data.sort_values(by='Date', inplace = True) 
    #data.drop_duplicates(data, inplace = True)
    data= data.groupby([ pd.Grouper(key='Date', freq = 'W-MON', label='left',closed='left')])['Diesel'].mean()

    return data 
  
def split_train_test(data, days_test):
    train = data[:-days_test]
    test = data[-days_test:]

    return train,test
    
def train_arima_mlflow(train,test, run_name):

    # Paramters
    p = range(0,3)
    d = range(0,3)
    q = range(0,3)

    P = range(0,3)
    D = range(0,3)
    Q = range(0,3)

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
    
 ```

</details>

### Model deployment
Once the training process is completed, a Python script evaluates model performance metrics and chooses the best model for deployment.

The following functions have been used:
- The `register_best_model` function deploys the best model in MLFlow for production.
- The `delete_no_registered_models_s3` function deletes from the s3 bucket the not selected models. 
- The `model_prod_s3` function stores in a s3 bucket the chosen model we'll use to make the predictions. 

<details>
<summary>View code</summary>

```python
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
    
 ```

</details>

### Predictions
Now that we have a selected model, we can start predicting fuel prices for the upcoming weeks.

The following functions have been used:
- The `download_s3` function is used to download the data from the S3 bucket
- The `preprocess_fuelprice` function is used to group the daily data into weekly data
- The `load_model_production` function is used to download the model in production from the s3 bucket
- The `apply_model` function is used to make predictions
- The `upload_s3` function is used to store the predictions in an S3 bucket


<details>
<summary>View code</summary>

```python
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
    #data.drop_duplicates(data, inplace = True)
    data= data.groupby([ pd.Grouper(key='Date', freq = 'W-MON', label='left',closed='left')])['Diesel'].mean()

    return data

# Load model in production (mlflow)
def load_model_production(bucket):
    # set up the S3 client
    s3 = boto3.client('s3')

    # specify the name of the bucket and the file path
    logged_model = 's3://gas-prices-project/production/'
    model = mlflow.sklearn.load_model(logged_model)
    
    return model

## Apply best model
def apply_model(model, data):
    pred =  model.predict(start=len(data), end=len(data)+4)
    return pred

# Save predictions into the s3
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

 ```
</details>

### Monitoring 
Evidently.ai is used in this project to monitor the pipeline and detecting data drift using the Kolmogorov-Smirnov (K-S) test. The K-S test is a nonparametric statistical test employed to determine whether two sets of data are derived from the same distribution.  The null hypothesis in this test assumes that the distributions are identical. If this hypothesis is rejected, it suggests that there is a drift in the model. More information about data drift parameters using Evidently [here](https://docs.evidentlyai.com/user-guide/customization/options-for-statistical-tests)

The following functions have been used:
- The `download_s3` function is used to download the data from the S3 bucket
- The `preprocess_fuelprice` function is used to group the daily data into weekly data
- The `check_data_drift` function is used to detect  if there are changes in the distribution of the input data over time.

<details>
<summary>View code</summary>

```python
    
    
 ```
</details>

## Architecture 
This is a diagram illustrating the architecture used in this project.
<a href="https://files.fm/u/y33vfwxjy#/view/architecture.PNG"><img src="https://files.fm/thumb_show.php?i=hte93metc"></a>

## GitHub Actions
- Continuous Integration (WIP). With every push, the Docker images are built and tested.
- Continuous Deployment (WIP). Upon successful completion of the CI workflow, the updated pipeline is deployed with every push.

## Tools and technologies used
- Cloud: AWS 
- Experiment tracking and Model Registry: MLFlow
- Workflow Orchestration: Prefect
- Model deployment: S3
- Model monitoring: EvidentlyAI
- Best practices:
    - GitHub Actions: CI/ CD
    - Docker: to containerize the project
    - Pytest: provide a simple, scalable, and powerful testing framework
    - Pylint: checks for errors in Python code and encourages good coding patterns 

## To do list
- [X] Monitor data drift using Evidently
- [ ]  [ ] CI/CD workflow with GitHub Actions (automatically train, retrain and deploy new models)
- [ ] Containerize project with Docker
- [ ] Include more tests

## Credits


