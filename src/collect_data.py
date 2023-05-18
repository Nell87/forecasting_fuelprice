# ------------------------ Libraries & Credentials ------------------------ #
import re
from datetime import datetime
from io import StringIO
import os
import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
import boto3
from prefect import flow,task

# ------------------------ FUNCTIONS ------------------------ #

# Check if the code is run locally to set up the environment configuration
@task
def where_am_i():
    hostname=os.popen('hostname').read()
    desktop = "DESKTOP"

    if desktop in hostname:
        os.environ["AWS_PROFILE"] = "mlops" # fill in with your AWS profile.
        os.environ['AWS_DEFAULT_REGION'] = "eu-west-1"

# Function to scrape weekly prices (Year has four digits and month one/two digits)
@task
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

@task
# Upload to S3
def upload_s3(bucket, new_data):
    s3 = boto3.client('s3')
    csv_buffer = StringIO()
    new_data.to_csv(csv_buffer, index=False)

    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket, 'new_data.csv').put(Body=csv_buffer.getvalue())

@task
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

# Scraper historical data
@task
def fuel_scraper_historical_data(first_year, last_year):
# Prepare the dataframe
    df=pd.DataFrame(columns=["Date", "Diesel"])
    years= range(first_year,last_year+1)

    for year in years:
        for month in range(1,13):
            temp_dataset = fuel_scraper_daily(year, month)
            df = pd.concat([df, temp_dataset])

    df['Diesel'] = df['Diesel'] .str.replace('[A-Za-z]', '')
    df = df[df['Diesel'] != 0]
             
    return df

# Main function
@flow
def pipeline():
    # Configuration
    where_am_i()

    # Scraper current month
    currentYear = datetime.now().year
    currentMonth = datetime.now().month
    dataset = fuel_scraper_daily(currentYear, currentMonth)

    # Scraper previous month
    if currentMonth > 1:
        currentMonth = currentMonth -1
    else:
        currentMonth = 12
        currentYear = currentYear - 1

    dataset_prev = fuel_scraper_daily(currentYear, currentMonth)
    dataset = pd.concat([dataset_prev, dataset])

    if dataset.empty == False:
        # Upload S3
        upload_s3("gas-prices-project", dataset)

        # Merge
        concat_data = merge_datasets_S3()

        # Upload S3
        upload_s3("gas-prices-project", concat_data)

        # Rename
        s3 = boto3.resource('s3')
        s3.Object('gas-prices-project','data.csv').delete()
        s3.Object('gas-prices-project','data.csv').copy_from(
                  CopySource='gas-prices-project/new_data.csv')
        s3.Object('gas-prices-project','new_data.csv').delete()

# ------------------------ WORKFLOW ------------------------ #
if __name__ == "__main__":
    pipeline()
