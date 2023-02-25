#Load Libraries
import requests 
from bs4 import BeautifulSoup as bs
import pandas as pd
import numpy as np
import re
from decimal import Decimal
from datetime import datetime, timedelta
from io import StringIO
import boto3

# ------------------------ FUNCTIONS ------------------------ #
# Function to scrape weekly prices (Year has four digits and month one/two digits)
def oil_scraper(year, month):

    # Let's make a request to check the status
    response = requests.get('https://es.fuelo.net/calendar/week/' + str(year) +  "/" + str(month) + "?lang=en'"    )
    status_code = (response.status_code)    

    if status_code != 200:
        return "The status code is not 200"

    # Extract content
    soup = bs(response.content,'html.parser')
    calendar_week = soup.find_all('div', {'class': 'calendar week'})
    calendar_week_elements = soup.find_all('div', class_='cell border')

    # Prepare the dataframe
    df=pd.DataFrame(columns=["Week", "Unleaded 95", "Diesel", "LPG"])

    # Scraper
    column_names = list(df.columns.values.tolist())
    columns = len(df.columns)
    rows = int(len(calendar_week_elements)/columns)

    i=0
    for row in range(rows):
        for column in range(columns):
            if column_names[column] == "Week":
                df.at[row, column_names[column]] = re.sub(re.compile(r'^[^0-9]*'), '', calendar_week_elements[i].text).strip()
                i=i+1
            else:
                df.at[row, column_names[column]] = re.sub(re.compile(r'^[^0-9]*'), '', calendar_week_elements[i].text)[:-5]
                i=i+1
        
    # Add end day
    for row in range(rows):
        df.at[row,'end_day'] = datetime.strptime(df.at[row, "Week"][-8:],'%d.%m.%y')
    
    # Add start day
    for row in range(rows):
        df.at[row,'start_day'] = datetime.strptime(df.at[row, "Week"][-8:],'%d.%m.%y') -  timedelta(days=6)

    # Remove week columns
    df = df.iloc[: , 1:]

    # Reorganize columns
    df = df[['start_day', 'end_day', 'Unleaded 95', 'Diesel', 'LPG']]


    return df

def upload_s3(bucket, new_data):
    s3 = boto3.client('s3')
    csv_buffer = StringIO()
    new_data.to_csv(csv_buffer, index=False)

    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket, 'new_data.csv').put(Body=csv_buffer.getvalue())

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
    return concat_data
    

# ------------------------ WORKFLOW ------------------------ #
# Scraper
dataset = oil_scraper(2023, 1)

# Upload S3
upload_s3("gas-prices-project", dataset)

# Merge
concat_data = merge_datasets_S3()

# Upload S3
upload_s3("gas-prices-project", concat_data)

# Rename
s3 = boto3.resource('s3')
s3.Object('gas-prices-project','data.csv').delete()
s3.Object('gas-prices-project','data.csv').copy_from(CopySource='gas-prices-project/new_data.csv')
s3.Object('gas-prices-project','new_data.csv').delete()

