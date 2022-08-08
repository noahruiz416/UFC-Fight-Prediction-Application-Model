#dataprocessing libraires
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#this python script handles basic data cleaning for the UFC project
#for this partcular case, this script handles the test data


#helper function that will allows us to display all values
def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')


#after dropping those columns, we can view the total amount of missing values, it seems there are a large amount of null values
#we can assume that these null values for certain fighters is because of the nature of UFC fights
#it is common for fighters to have one fight in the organization and then to never come again
#because of this we will use mean imputation, to fill in these values, but before that we will filter our dataframe to the modern UFC era
#since we have a large amount of missing data, we can make an assumption and fill null values with 0, the reason this assumption may hold,
#is that since a fighter may be new to the UFC they do not have any data or statisitcs for the UFC so we will imput with 0's for know
#however we will also perform mean imputationt to see if model performance changes
#for null objects we will drop those specific rows
    # Refree
    # R_Stance
    # B_Stance

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
    X_train = pd.read_csv('/users/n/UFC-Predictions/data/X_train_data.csv', index_col=0)
    y_train = pd.read_csv('/users/n/UFC-Predictions/data/y_train_data.csv', index_col=0)

    #creating our training dataframe, we will process this data and create our processed X_train, and y_train frames
    train_data = X_train.join(y_train)

    train, test = procesed_labels(train_data)

    train.to_csv('X_processed_train.csv')
    test.to_csv('y_processed_train.csv')

#call our function
main()
