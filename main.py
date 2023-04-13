import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# # removing outliers using the Interquartile Range (IQR) Method
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


data_df['Duration'] = data_df['Duration'].apply(duration_to_minutes)

# Analyze which airline companies dominate the market ------
flight_counts = data_df['Airline'].value_counts()
plt.figure(figsize=(16, 9))
sns.barplot(x=flight_counts.index, y=flight_counts.values)
plt.title('Most popular Airline Companies')
plt.xlabel('Airline Company', fontsize=5)
plt.ylabel('Number of Flights')
plt.xticks(rotation=15)
plt.tight_layout()

# Find the most popular destination ------
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


# Analyze correlation between duration of flight and price ------
corr_duration_price = data_df['Duration'].corr(data_df['Price_in_Euro'])  # correlation ranges from -1 to 1
print('Correlation between Duration and Price in EUR: {:.2f}'.format(corr_duration_price))
plt.scatter(data_df['Duration'], data_df['Price_in_Euro'], alpha=0.5)
plt.xlabel('Duration of Flight (in minutes)')
plt.ylabel('Price in Euro')
plt.title('Correlation between Flight Duration and Price')
plt.show()

# Analyze correlation between total stops and price
corr_stops_price = data_df['Total_Stops'].corr(data_df['Price_in_Euro'])
print('Correlation between Total Stops and Price in EUR: {:.2f}'.format(corr_stops_price))
sns.violinplot(data=data_df, x='Total_Stops', y='Price_in_Euro', alpha=0.5)
plt.xlabel('Total Stops')
plt.ylabel('Price in Euro')
plt.title('Correlation between Total Stops and Price')
plt.show()

# Analyze correlation between date/time and price
# Analyze route popularity, which Source -> Destination routes are the most popular
# Examine competition between Airlines, examine how different airlines compete with each other on different routes by comparing prices, flight frequency, and other factors.
# Predict flight prices
