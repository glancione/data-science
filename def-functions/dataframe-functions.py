########################################
#### Dataframe Functions ###############
########################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import nltk
from nltk.corpus import stopwords

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


def preprocess_suggestion(column):
    suggestion_ = ""
    data_type_ = str(column.dtype)
    if data_type_.startswith('int') or data_type_.startswith('float'):
        if column.isnull().sum() > 0:
            suggestion_ += "Fill missing values with: mean, median, or mode (depending on distribution).\n"
        q1 = column.quantile(0.25)
        q3 = column.quantile(0.75)
        iqr = q3 - q1
        outliers = column[(column < (q1 - 1.5 * iqr)) | (column > (q3 + 1.5 * iqr))]

        if len(outliers) > 0:
            suggestion_ += "Consider handling outliers: capping, flooring, or removal.\n"

        if (column.skew() > 1) or (column.skew() < -1):
            suggestion_ += "Distribution is skewed. Consider log or square root transformation.\n"

        if (column.max() - column.min()) > 100:
            suggestion_ += "Large scale difference. Consider normalization or standardization.\n"

    elif data_type_ == 'object':
        if column.isnull().sum() > 0:
            suggestion_ += "Fill missing values with: mode or 'unknown' category if categorical.\n"
            suggestion_ += "Fill missing values with: empty string or 'missing' value (if text).\n"

        if len(column.unique()) / len(column) > 0.1:
            suggestion_ += "High cardinality. Consider feature hashing, target encoding, or frequency encoding.\n"

        if not pd.api.types.is_datetime64_dtype(column):
            suggestion_ += "Check for consistent formatting in text data.\n"

        if column.str.contains(r'[^a-zA-Z\s]').any():
            suggestion_ += "Consider removing noise (numbers, special characters).\n"

        if column.apply(lambda x: len(set(x.split()) & set(stopwords.words('english'))) > 0).any():
            suggestion_ += "Consider removing stop words.\n"

        if column.str.len().mean() > 100:
            suggestion_ += "Text length is relatively long. Consider text summarization or truncation.\n"

        if column.apply(lambda x: len(set(x.split())) / len(x.split())).mean() < 0.5:
            suggestion_ += "Text contains repeated words. Consider deduplication or stemming.\n"

    elif data_type_ == 'datetime64[ns]':
        if column.isnull().sum() > 0:
            suggestion_ += "Fill missing values with appropriate date or interpolation.\n"

        if not pd.api.types.is_datetime64_dtype(column):
            suggestion_ += "Check for consistent date format.\n"

        suggestion_ += "Consider extracting features like year, month, day, hour, minute, etc.\n"
    else:
        suggestion_ += "Data type not recognized. Please investigate further.\n"

    if not suggestion_:
        suggestion_ = "Column seems clean, no immediate preprocessing needed."

    return suggestion_

