import pandas as pd
from sklearn.model_selection import train_test_split


#load data split data into train and test
UFC_data_original_copy = pd.read_csv('/users/n/UFC-Predictions/data/data.csv')

#input features and target var, additionally getting rid of draws since that is not apart of this project
UFC_data_original_copy.drop(UFC_data_original_copy[UFC_data_original_copy['Winner'] == 'Draw'].index, inplace = True)

X = UFC_data_original_copy.drop(columns = ['Winner'])
y = UFC_data_original_copy['Winner']

#splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

X_train.to_csv('X_train_data.csv')
X_test.to_csv('X_test_data.csv')
y_train.to_csv('y_train_data.csv')
y_test.to_csv('y_test_data.csv')
