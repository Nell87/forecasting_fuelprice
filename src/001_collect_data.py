#Load Libraries
import requests 
from bs4 import BeautifulSoup as bs
import pandas as pd
import numpy as np
import re
from decimal import Decimal
from datetime import datetime, timedelta
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

def upload_s3():
    s3 = boto3.client("s3")
    bucket = "gas-prices-project"

    s3.put_object(Bucket=bucket, Key=test, Body = "Testing script")

    print("Done!")

    # ------------------------ WORKFLOW ------------------------ #
test = oil_scraper(2023, 1)
upload_s3()