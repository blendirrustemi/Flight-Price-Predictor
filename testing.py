import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import plotly.express as px


# pd.set_option('display.max_columns', None) # ALlows to display all columns

data_train_df = pd.read_csv('Datasets/Data_train_id.csv')
distance_df = pd.read_csv('Datasets/Data_distance.csv')

# Merge the datasets on ID
data_df = pd.merge(data_train_df, distance_df[['id', 'Route', 'Distance_in_km']], on='id')

# print(data_df.head())


# check for missing values and remove missing values
print("Missing values: ", data_df.isnull().sum().sum())
data_df = data_df.dropna()

# check and remove for duplicates
print("Duplicates: ", data_df.duplicated().sum())

# removing outliers using the Interquartile Range (IQR) Method
Q1 = data_df['Price_in_Euro'].quantile(0.25)
Q3 = data_df['Price_in_Euro'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

data_df = data_df[(data_df['Price_in_Euro'] >= lower_bound) & (data_df['Price_in_Euro'] <= upper_bound)]

# convert the Total_Stops column from "non-stop", "1 stop, "2 stops", etc. to 0, 1, 2, etc
data_df['Total_Stops'] = data_df['Total_Stops'].replace({'non-stop': 0, '1 stop': 1, '2 stops': 2,
                                                         '3 stops': 3, '4 stops': 4})

# convert duration to minutes
data_df = data_df[data_df['Duration'] != '5m']  # removed one row where duration is only 5 minutes


# function to convert time like '1h 25m' to minutes
def duration_to_minutes(duration_str):
    time_parts = duration_str.split()
    hours = 0
    minutes = 0

    for part in time_parts:
        if 'h' in part:
            hours = int(part.replace('h', ''))
        elif 'm' in part:
            minutes = int(part.replace('m', ''))

    total_minutes = hours * 60 + minutes
    return total_minutes

# call the function and apply to every row in the Duration column
data_df['Duration'] = data_df['Duration'].apply(duration_to_minutes)


# -------------------------------------------------------


avg_price_airline = data_df.groupby('Airline')['Price_in_Euro'].mean().reset_index()

# Create the bar plot using Plotly Express
fig = px.bar(avg_price_airline, x='Airline', y='Price_in_Euro', title='Average Flight Prices by Airline')
fig.update_layout(xaxis_title='Airline', yaxis_title='Average Price in Euro')
fig.show()
