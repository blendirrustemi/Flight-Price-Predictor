import pandas as pd


data_df = pd.read_csv('Datasets/Data_train_id.csv')

print(data_df.head())


# check for missing values and remove missing values
print("Missing values: ", data_df.isnull().sum().sum())
data_df = data_df.dropna()


# removing outliers using the Interquartile Range (IQR) Method
Q1 = data_df['Price_in_Euro'].quantile(0.25)
Q3 = data_df['Price_in_Euro'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

data_df = data_df[(data_df['Price_in_Euro'] >= lower_bound) & (data_df['Price_in_Euro'] <= upper_bound)]


# check and remove for duplicates
print("Duplicates: ", data_df.duplicated().sum())


# convert the Total_Stops column from "non-stop", "1 stop, "2 stops", etc. to 0, 1, 2, etc
# first, check the number of stops
data_df['Total_Stops'] = data_df['Total_Stops'].replace({'non-stop': 0, '1 stop': 1, '2 stops': 2,
                                                         '3 stops': 3, '4 stops': 4})



# convert duration to minutes
data_df = data_df[data_df['Duration'] != '5m'] # removed one row where duration is only 5 minutes

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

