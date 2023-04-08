import pandas as pd

data_df = pd.read_csv('Datasets/Data_train.csv')
dist = pd.read_csv('Datasets/Distance_between_cities.csv')

data_df = data_df[:-1]

data_df.insert(0, 'id', range(1, len(data_df) + 1))

data_df.to_csv('Datasets/Data_train_id.csv', index=False)
