import csv

import requests
import pandas as pd
from bs4 import BeautifulSoup

# Read the csv file
data_df = pd.read_csv('OtherData/Data_train.csv')


# Get the distance between two cities
def get_distance(url):
    # URL to scrape
    # url = "https://www.distance.to/" + source + "/" + destination
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) '
                             'Chrome/50.0.2661.102 Safari/537.36'}
    # Get the page
    page = requests.get(url, headers=headers)

    # Parse the page
    soup = BeautifulSoup(page.content, 'html.parser')
    results = soup.find_all('span', class_='value km')

    # Return the distance
    return results[0].text


# Subset the data to Destination, Origin, Route
subs_data_df = data_df[['Source', 'Destination', 'Route']]

city_codes = []
destination = []

for index, row in subs_data_df.iterrows():
    # split the route into a list
    city_codes.append(str(row['Route']).split(';'))


