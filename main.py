import numpy as np
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# pd.set_option('display.max_columns', None) # ALlows to display all columns

data_train_df = pd.read_csv('Datasets/Data_train_id.csv')
distance_df = pd.read_csv('Datasets/Data_distance.csv')

# Merge the datasets on ID
data_df = pd.merge(data_train_df, distance_df[['id', 'Route', 'Distance_in_km']], on='id')


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


# Analyze which airline companies dominate the market ----------------------------
flight_counts = data_df['Airline'].value_counts()
plt.figure(figsize=(16, 9))
sns.barplot(x=flight_counts.index, y=flight_counts.values)
plt.title('Most popular Airline Companies', fontsize=20)
plt.xlabel('Airline Company', fontsize=15)
plt.ylabel('Number of Flights', fontsize=15)
plt.xticks(rotation=15)
plt.tight_layout()


# Find the most popular destination ----------------------------
destination_counts = data_df.groupby(['Airline', 'Destination']).size().reset_index(name='Counts')

# Get the top 5 destinations for each airline
top_destinations = destination_counts.groupby('Airline').apply(lambda x: x.nlargest(5, 'Counts'))

# Create a bar plot for each airline
fig, axs = plt.subplots(nrows=len(destination_counts['Airline'].unique()), figsize=(10, 20), sharex=True)

for i, airline in enumerate(top_destinations['Airline'].unique()):
    airline_data = top_destinations[top_destinations['Airline'] == airline]
    axs[i].barh(airline_data['Destination'], airline_data['Counts'], color='blue')
    axs[i].set_title(airline)

plt.tight_layout()
plt.show()


# Analyze correlation between duration of flight and price ----------------------------
corr_duration_price = data_df['Duration'].corr(data_df['Price_in_Euro'])  # correlation ranges from -1 to 1
print('Correlation between Duration and Price in EUR: {:.2f}'.format(corr_duration_price))
plt.scatter(data_df['Duration'], data_df['Price_in_Euro'], alpha=0.5)
plt.xlabel('Duration of Flight (in minutes)')
plt.ylabel('Price in Euro')
plt.title('Correlation between Flight Duration and Price')
plt.show()


# Analyze correlation between total stops and price ----------------------------
corr_stops_price = data_df['Total_Stops'].corr(data_df['Price_in_Euro'])
print('Correlation between Total Stops and Price in EUR: {:.2f}'.format(corr_stops_price))
sns.violinplot(data=data_df, x='Total_Stops', y='Price_in_Euro', alpha=0.5)
plt.xlabel('Total Stops')
plt.ylabel('Price in Euro')
plt.title('Correlation between Total Stops and Price')
plt.show()


# Analyze route popularity, which Source -> Destination routes are the most popular ----------------------------
route_popularity_df = data_df.groupby(['Source', 'Destination']).size().reset_index(name='Count')
sorted_routes = route_popularity_df.sort_values(by='Count', ascending=False)
plt.figure(figsize=(16, 10))
plt.barh(sorted_routes['Source'] + ' -> ' + sorted_routes['Destination'], sorted_routes['Count'])
plt.xlabel('Number of Flights', fontsize=12)
plt.ylabel('Route', fontsize=12)
plt.title('Route Popularity', fontsize=20)
plt.show()


# Analyze what days of the week are the cheapest/expensive ----------------------------
# Function to create a new column which converts the date to day of the week
def date_to_day_of_week(date_str):
    date_obj = datetime.datetime.strptime(date_str, '%d/%m/%Y')
    return date_obj.strftime('%A')


data_df['Day_of_Week'] = data_df['Date_of_Journey'].apply(date_to_day_of_week)

price_df = data_df[['Day_of_Week', 'Price_in_Euro']]

# Calculate average price for each day of the week
avg_price_df = price_df.groupby('Day_of_Week').mean().reset_index()

# Plot average price for each day of the week
plt.bar(avg_price_df['Day_of_Week'], avg_price_df['Price_in_Euro'])
plt.xlabel('Day of the Week')
plt.ylabel('Average Price (Euros)')
plt.title('Flight Prices by Day of the Week')
plt.show()


# Convert the Time to a correct format, and split the day into 4 quarters ----------------------------
# Convert the 'Dep_Time' column to datetime
data_df['Dep_Time'] = pd.to_datetime(data_df['Dep_Time'], format='%H:%M')

# Create a new column for the time of day
data_df['Dep_Time_Part'] = pd.cut(data_df['Dep_Time'].dt.hour,
                                  bins=[0, 6, 12, 18, 24],
                                  labels=['Late Night', 'Morning', 'Afternoon', 'Evening'],
                                  include_lowest=True)

# Calculate the average price for each time of day
dep_time_prices = data_df.groupby('Dep_Time_Part')['Price_in_Euro'].mean().reset_index()

# Sort the values by time of day
dep_time_prices.sort_values(by='Dep_Time_Part', inplace=True)

# Create the bar plot using matplotlib
plt.figure(figsize=(10, 6))
plt.bar(dep_time_prices['Dep_Time_Part'], dep_time_prices['Price_in_Euro'])
plt.title('Average Flight Prices by Time of Day')
plt.xlabel('Time of Day')
plt.ylabel('Average Price in Euro')
plt.show()


#  Predict flight prices ----------------------------
print("-------------- Prediction --------------")
# X - features
X = data_df[['Duration', 'Total_Stops']]
# y - target variable
y = data_df['Price_in_Euro']

# Splitting Data into training and evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Decision Tree Regressor
dtree = DecisionTreeRegressor(min_samples_split=10)
dtree.fit(X_train, y_train)
y_pred_dtree = dtree.predict(X_test)
rmsle_dtree = 1 - mean_squared_log_error(y_test, y_pred_dtree)**0.5
print("Decision Tree Regressor 1-RMSLE:", rmsle_dtree)

# SVR
svr = SVR(C=2000.0, epsilon=0.1)
svr.fit(X_train, y_train)
y_pred_svr = svr.predict(X_test)
rmsle_svr = 1 - mean_squared_log_error(y_test, y_pred_svr)**0.5
print("SVR 1-RMSLE:", rmsle_svr)

# Random Forest Regressor
forest = RandomForestRegressor(n_estimators=100, min_samples_split=12)
forest.fit(X_train, y_train)
y_pred_forest = forest.predict(X_test)
rmsle_forest = 1 - mean_squared_log_error(y_test, y_pred_forest)**0.5
print("Random Forest Regressor 1-RMSLE:", rmsle_forest)

# Gradient Boosting Regressor
Grad = GradientBoostingRegressor(learning_rate=0.1, n_estimators=100)
Grad.fit(X_train, y_train)
y_pred_Grad = Grad.predict(X_test)
rmsle_Grad = 1 - mean_squared_log_error(y_test, y_pred_Grad)**0.5
print("Gradient Boosting Regressor 1-RMSLE:", rmsle_Grad)

# Random Forest Regressor selected for predictions
forest.fit(X_test, y_test)
y_predict = forest.predict(X_test)

df = pd.DataFrame(y_predict, columns=['Price'])

# Storing the prices in Excel format
df.to_excel('Predictions/Flight_Predictions.xlsx', index=False)


# Compare the Prices with the Predicted ----------------------------
# Create a DataFrame with actual and predicted prices
df = pd.DataFrame({'Actual_Price': y_test, 'Predicted_Price': y_predict})

# Calculate evaluation metrics
rmse = np.sqrt(mean_squared_error(df['Actual_Price'], df['Predicted_Price']))
mae = mean_absolute_error(df['Actual_Price'], df['Predicted_Price'])
r2 = r2_score(df['Actual_Price'], df['Predicted_Price'])

df.to_excel('Predictions/actual_predicted_prices.xlsx', index=False)

# Print evaluation metrics
print('RMSE:', rmse)
print('MAE:', mae)
print('R-squared:', r2)



# Interactive visualization using Plotly ----------------------------
fig = px.scatter(data_df, x='Duration', y='Price_in_Euro', opacity=0.5,
                 labels={'Duration': 'Duration of Flight (in minutes)', 'Price_in_Euro': 'Price in Euro'},
                 title='Correlation between Flight Duration and Price')

fig.show()


# Flight Prices by Month ----------------------------
# Extract the month from the 'Date_of_Journey' column
data_df['Month'] = pd.to_datetime(data_df['Date_of_Journey'], format='%d/%m/%Y').dt.month_name()

# Define the order of months
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July']
# Convert the 'Month' column to an ordered categorical data type
data_df['Month'] = pd.Categorical(data_df['Month'], categories=month_order, ordered=True)

# Group the data by month and calculate the average price
monthly_prices = data_df.groupby('Month')['Price_in_Euro'].mean().reset_index()

# Create the line plot using Plotly Express
fig = px.line(monthly_prices, x='Month', y='Price_in_Euro', markers=True, title='Flight Prices by Month')
fig.update_layout(xaxis={'type': 'category'})
fig.show()


# Average Price per Airline ----------------------------
avg_price_airline = data_df.groupby('Airline')['Price_in_Euro'].mean().reset_index()

# Create the bar plot using Plotly Express
fig = px.bar(avg_price_airline, x='Airline', y='Price_in_Euro', title='Average Flight Prices by Airline')
fig.update_layout(xaxis_title='Airline', yaxis_title='Average Price in Euro')
fig.show()
