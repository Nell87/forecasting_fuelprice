# Fuel price project
[![mlops](https://github.com/Nell87/forecasting_fuelprice/actions/workflows/mlops.yaml/badge.svg)](https://github.com/Nell87/forecasting_fuelprice/actions/workflows/mlops.yaml)
[![testing](https://github.com/Nell87/forecasting_fuelprice/actions/workflows/test.yaml/badge.svg)](https://github.com/Nell87/forecasting_fuelprice/actions/workflows/test.yaml)
[![linter-pylint](https://github.com/Nell87/forecasting_fuelprice/actions/workflows/linter.yaml/badge.svg)](https://github.com/Nell87/forecasting_fuelprice/actions/workflows/linter.yaml)

## Objective
The objective of the project is to build an end to end machine learning project to predict fuel price. It involves all the steps from collecting data to training, deploying and monitoring a model. 

## Project structure and architecture
The project is implemented on Ubuntu 22.04 using AWS. 

### Gathering data
Run `001_collect_data.py` from src folder to gather fuel price from the previous days.
- The `fuel_scraper_daily` function will scrape the data
- The `upload_s3` function will upload new data to the s3
- The `merge_datasets_s3` function will merge old and new data in the s3

### Modeling / Experiment tracking
Run `002_modeling.py` from src folder to train SARIMA models with different parameters. The parameters and models registry will be created using MLFlow. I'll save the tracking in an Amazon Relational Database Service (RDS) and keep the best model inside a s3 bucket

### Model deployment
Run '003_deployment.py' to put into production the best model resulting from the experiments. 

### Monitoring
Evidently.ai is used to monitor the pipeline. 

### Schema 
This is a schema about the architecture used in this project
![Schema](https://files.fm/thumb_show.php?i=bmynmww54)

## Tools and technologies used
- Cloud: AWS
- Experiment tracking and Model Registry: MLFlow
- Workflow Orchestration: Prefect
- Model deployment:
- Model monitoring: 
- Best practices: 

## To do next
The project has some improvements to do 
