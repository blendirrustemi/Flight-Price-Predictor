import pandas as pd

data_df = pd.read_csv('Datasets/Data_train_id.csv')

print(data_df.head())


# check for missing values
print("Missing values: ", data_df.isnull().sum().sum())


# remove missing values
data_df = data_df.dropna()


# check and remove for duplicates
print("Duplicates: ", data_df.duplicated().sum())


# convert the Total_Stops column from "non-stop", "1 stop, "2 stops", etc. to 0, 1, 2, etc
# first, check the number of stops
print(data_df['Total_Stops'].value_counts())

data_df['Total_Stops'] = data_df['Total_Stops'].replace({'non-stop': 0, '1 stop': 1, '2 stops': 2,
                                                         '3 stops': 3, '4 stops': 4})

