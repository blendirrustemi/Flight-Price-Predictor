import csv
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

data_df = pd.read_csv('OtherData/Data_train.csv')

print(data_df['Route'])
