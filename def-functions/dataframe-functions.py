########################################
#### Dataframe Functions ###############
########################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def analyze_column(df, column_name):
    column_data = df[column_name]

    analysis = {
        "column_name": column_name,
        "data_type": column_data.dtype,
        "count": column_data.count(),
        "missing_values": column_data.isnull().sum(),
        "unique_values": column_data.nunique(),
        "top_values": column_data.value_counts().head(5),
        "summary_statistics": column_data.describe()
    }

    if np.issubdtype(column_data.dtype, np.number):
        analysis["mean"] = column_data.mean()
        analysis["median"] = column_data.median()
        analysis["std"] = column_data.std()
        analysis["min"] = column_data.min()
        analysis["max"] = column_data.max()
        analysis["quartiles"] = column_data.quantile([0.25, 0.5, 0.75])

    return analysis


def handle_missing_values(df, strategy='dropna', fillna=None, imputena=None):
    if strategy == 'dropna':  # The strategy for handling missing values. Options are 'dropna', 'fill', 'impute'.
        return df.dropna()
    elif strategy == 'fill':
        return df.fillna(fillna)
    elif strategy == 'impute':      # Implement imputation logic
        pass
    else:
        raise ValueError("Invalid strategy for handling missing values.")


def handle_outliers(df, columns, method='zscore', threshold=3):
    if method == 'zscore':
        for col in columns:
            z_scores = stats.zscore(df[col])
            df = df[abs(z_scores) < threshold]
    elif method == 'iqr':  # Options are 'zscore', 'iqr'.
        # Implement IQR-based outlier detection
        pass
    else:
        raise ValueError("Invalid method for outlier detection.")
    return df



def encode_categorical_features(df, columns):
    return pd.get_dummies(df, columns=columns)


def describe_numerical_columns(df, columns):
    return df[columns].describe()

