import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np


def drop_duplicates(df, subset_name):
    df.drop_duplicates(subset=[subset_name], inplace=True)
    return df


def encode(df, column_to_encode):
    le = LabelEncoder()
    df[column_to_encode] = le.fit_transform(df[column_to_encode])
    return df


def outlier_handling(df, column_with_outliers, lower_bound=0.25, upper_bound=0.75):
    q1 = df[column_with_outliers].quantile(lower_bound)
    q3 = df[column_with_outliers].quantile(upper_bound)
    iqr = q3 - q1
    df = df[(df[column_with_outliers] > (q1 - 1.5 * iqr)) & (df[column_with_outliers] < (q3 + 1.5 * iqr))]
    return df


def date_formatting(df, column_with_date):
    df[column_with_date] = pd.to_datetime(df[column_with_date],
                                          format='%m/%d/%Y')
    return df


def remove_missing_values(df):
    missing_values = df.isnull().sum()
    df = df.dropna()
    print("Removed {} missing values".format(missing_values.sum()))
    return df


def data_cleaning_pipeline(df, duplication_subset=None, column_to_encode=None, column_with_outliers=None, column_with_date=None,
                           sep=','):
    if isinstance(df, str):
        print("\nDataframe path detected\n")
        df = pd.read_csv(df, sep=sep)
    elif isinstance(df, pd.DataFrame):
        print("\nDataframe detected\n")
    else:
        return print("\nNot a Dataframe\n")
    print("\nData cleaning in progress...\n")
    df = drop_duplicates(df, duplication_subset)
    df = encode(df, column_to_encode)
    df = outlier_handling(df, column_with_outliers)
    df = date_formatting(df, column_with_date)
    df = remove_missing_values(df)
    print("\nData cleaning finished\n")
    return df


""""
#testing data

data = {'Name': ['John', 'Jane', 'Bob', 'John', 'Alice'], 
        'Age': [30, 25, 40, 30, np.NaN], 
        'Gender': ['Male', 'Female', 'Male', 'Male', 'Female'], 
        'Income': [50000, 60000, 70000, 45000, 80000], 
        'Birthdate': ['01/01/1990', '02/14/1996', '03/15/1981', 
                      '01/01/1990', '06/30/1986'], 
        'Married': [True, False, True, False, True], 
        'Children': [2, 0, 1, 0, 3]} 
        
df = pd.DataFrame(data) 

print('Before Preprocessing:\n',df) 
      
clean_df = data_cleaning_pipeline(df, 
                                  'Name',  
                                  'Gender',  
                                  'Income', 
                                  'Birthdate') 
  
print('\nAfter preprocessing') 
clean_df.head()
"""