#dataprocessing libraires
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#this simple python script takes in a pandas dataframe as the primary input and cleans the given files
#for this partcular case, this script handles the test data

def filter_era(df):
    df['DateTime'] = pd.to_datetime(df['date'])
    df['year'] = df['DateTime'].dt.year
    clean = df[df['year'] > 2001]
    return clean

def clean_rows(df):
    UFC_filtered_ref = df[df['Referee'].notna()]
    UFC_filtered_stance_r = UFC_filtered_ref[UFC_filtered_ref['R_Stance'].notna()]
    UFC_filtered_final = UFC_filtered_stance_r[UFC_filtered_stance_r['B_Stance'].notna()]
    return UFC_filtered_final

def impute_zeroes(df):
    imputed_zero = df.fillna(0)
    return imputed_zero

def clean_all(df):
    filtered_era = filter_era(df)
    rows = clean_rows(filtered_era)
    imputed = impute_zeroes(rows)
    return imputed

#create a function given the input data
def procesed_labels(df):
    processed_files_all = clean_all(df)
    X = processed_files_all.drop(columns = ['date', 'Winner', 'DateTime'])
    y = processed_files_all['Winner']
    return X, y

#main function, call to run whole script
def main():
    X_train = pd.read_csv('/users/n/UFC-Predictions/data/X_test_data.csv', index_col=0)
    y_train = pd.read_csv('/users/n/UFC-Predictions/data/y_test_data.csv', index_col=0)

    #creating our training dataframe, we will process this data and create our processed X_train, and y_train frames
    train_data = X_train.join(y_train)

    train, test = procesed_labels(train_data)

    train.to_csv('X_processed_test.csv')
    test.to_csv('y_processed_test.csv')

#call our function
main()
