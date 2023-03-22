# Fuel price project

## Objective
The objective of the project is to predict fuel price in the next seven days. This project is an end to end machine learning project that involves all the steps from building to deploying and monitoring the model. 

## Project structure and architecture
The project is implemented on Ubuntu 22.04 using AWS. 

### Gathering data
Run `001_collect_data.py` from src folder to gather fuel price from the previous days.
- The function `fuel_scraper_daily` will scrape the data
- The function `upload_s3` will upload new data to the s3
- The function `merge_datasets_s3` will merge old and new data in the s3

### Modeling
Run `002_modeling.py` from src folder to train SARIMA models with different parameters. The parameters and models registry will be saved in MLFlow

### Schema 
This is a schema about the architecture used in this project

## Tools and technologies used
- Cloud: AWS
- Experiment tracking and Model Registry: MLFlow
- Workflow Orchestration: Prefect
- Model deployment:
- Model monitoring: 
- Best practices: 

## To do next
The project has some improvements
