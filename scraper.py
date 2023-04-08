import csv
import requests
import pandas as pd
from bs4 import BeautifulSoup
import time

# Read the csv file
data_df = pd.read_csv('OtherData/Data_train.csv')

count = 1

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

    # Wait for 1 second
    time.sleep(1)

    # Print the counter to keep track of progress
    print(count, ' - ', results[0].text)

    # Return the distance
    return results[0].text


# Subset the data to Destination, Origin, Route
subs_data_df = data_df[['Source', 'Destination', 'Route']]

# Create a list of lists of city codes and list of distance
city_codes = []
distance = []


for index, row in subs_data_df.iterrows():
    # split the route into a list
    city_codes.append(str(row['Route']).split(';'))

# Get the distance between the cities
for i in range(len(city_codes)):
    match len(city_codes[i]):
        case 2:
            url = "https://www.distance.to/" + city_codes[i][0] + "/" + city_codes[i][1]
            distance.append(get_distance(url))
        case 3:
            url = "https://www.distance.to/" + city_codes[i][0] + "/" + city_codes[i][1] + "/" + city_codes[i][2]
            distance.append(get_distance(url))
        case 4:
            url = "https://www.distance.to/" + city_codes[i][0] + "/" + city_codes[i][1] + "/" + \
                  city_codes[i][2] + "/" + city_codes[i][3]
            distance.append(get_distance(url))
        case 5:
            url = "https://www.distance.to/" + city_codes[i][0] + "/" + city_codes[i][1] + "/" + \
                    city_codes[i][2] + "/" + city_codes[i][3] + "/" + city_codes[i][4]
            distance.append(get_distance(url))
        case 6:
            url = "https://www.distance.to/" + city_codes[i][0] + "/" + city_codes[i][1] + "/" + \
                    city_codes[i][2] + "/" + city_codes[i][3] + "/" + city_codes[i][4] + "/" + city_codes[i][5]
            distance.append(get_distance(url))

    count += 1


with open('OtherData/Distance_between_cities.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Source', 'Destination', 'Route', 'Distance_in_km'])
    for i in range(len(data_df)):
        writer.writerow([data_df['Source'][i], data_df['Destination'][i], data_df['Route'][i], distance[i]])

