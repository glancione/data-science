########################################
#### Dataframe Functions ###############
########################################

import pandas as pd
import numpy as np
from scipy import stats
from nltk.corpus import stopwords
from sklearn.impute import SimpleImputer
import os


def subsample_set(n, df):
    n_subsampling = min(n, df.shape[0])
    print("Subsampling Data ENABLED!")
    df = df[0:n_subsampling]
    return df
    

def get_train_test_val_sets(train_path, test_path, val_path, n_subsampling=None):
    print("Loading Train data...")
    X_train = pd.read_csv(os.path.join(train_path, "X_train.csv"), sep=";")
    y_train = pd.read_csv(os.path.join(train_path, "y_train.csv"), sep=";")

    if n_subsampling:
        X_train = subsample_set(n_subsampling, X_train)
        y_train = subsample_set(n_subsampling, y_train)

    print("Correctly loaded Train data!")
    print("Train data Sample: \n {}".format(X_train.sample(5)))
    print("-" * 20)

    print("Loading Test data...")
    X_test = pd.read_csv(os.path.join(test_path, "X_test.csv"), sep=";")
    y_test = pd.read_csv(os.path.join(test_path, "y_test.csv"), sep=";")

    if n_subsampling:
        X_test = subsample_set(n_subsampling, X_test)
        y_test = subsample_set(n_subsampling, y_test)

    print("Correctly loaded Test data!")
    print("Test data Sample: \n {}".format(X_test.sample(5)))
    print("-" * 20)

    print("Loading Val data...")
    X_val = pd.read_csv(os.path.join(val_path, "X_val.csv"), sep=";")
    y_val = pd.read_csv(os.path.join(val_path, "y_val.csv"), sep=";")

    if n_subsampling:
        X_val = subsample_set(n_subsampling, X_val)
        y_val = subsample_set(n_subsampling, y_val)

    print("Correctly loaded Val data!")
    print("Val data Sample: \n {}".format(X_val.sample(5)))
    print("-" * 20)

    print("-" * 20)
    print("Train set has %s rows", str(X_train.shape[0]))
    print("Test set has %s rows", str(X_test.shape[0]))
    print("Validation set has %s rows", str(X_val.shape[0]))
    print("-" * 20)

    return X_train, y_train, X_test, y_test, X_val, y_val


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


def handle_missing_values(df, strategy_='drop', fill_val=0, impute_strategy='most_frequent'):
    if strategy_ == 'drop':
        return df.dropna()
    elif strategy_ == 'fill':
        return df.fillna(fill_val)
    elif strategy_ == 'impute':
        imp = SimpleImputer(strategy=impute_strategy)
        return imp.fit_transform(df)
    else:
        raise ValueError("Invalid strategy for handling missing values.")


def impute_missing_values(df, column, strategy='mean'):
    if strategy == 'mean':
        return df[column].fillna(df[column].mean(), inplace=True)
    elif strategy == 'median':
        return df[column].fillna(df[column].median(), inplace=True)
    elif strategy == 'mode':
        return df[column].fillna(df[column].mode()[0], inplace=True)
    else:
        raise ValueError("Invalid strategy for imputation.")


def calculate_correlation(df, column1, column2):
    return df[column1].corr(df[column2])


def handle_outliers(df, columns, method='zscore', threshold=3):
    if method == 'zscore':
        for col in columns:
            z_scores = stats.zscore(df[col])
            df = df[abs(z_scores) < threshold]
    elif method == 'iqr':  # Options are 'zscore', 'iqr'.
        # TO DO Implement IQR-based outlier detection
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
            percent_missing = column.isnull().sum() * 100 / len(column)
            suggestion_ += str(percent_missing) + " % of values are outliers.\n"
            suggestion_ += "Fill missing values with: mean, median, or mode (depending on distribution).\n"
        q1 = column.quantile(0.25)
        q3 = column.quantile(0.75)
        iqr = q3 - q1
        outliers = column[(column < (q1 - 1.5 * iqr)) | (column > (q3 + 1.5 * iqr))]

        if len(outliers) > 0:
            suggestion_ += "Found " + str(len(outliers)) + " outliers.\n"
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


def unroll_vector_column(df, column_name, drop_old_column=False):
    print('-'*20)
    print('Unrolling {} column'.format(column_name))
    numpy_data = np.array(df[column_name])
    df_out = pd.DataFrame(data=numpy_data)
    df_out.columns = [f'vector{i}' for i in range(0,len(df_out.T))] 
    df= pd.concat([df, df_out], axis = 1)
    print('Column {} correctly unrolled'.format(column_name))
    print('Unrolled vector sample:\n {}'.format(df_out.sample(5)))
    print('-'*20)
    if drop_old_column:
        print('Drop Old Column ENABLED')
        try:
            df = df.drop(column_name, axis=1)
            print('{} correctly dropped!'.format(column_name))
        except Exception as e:
            print(e)
        print('-'*20)
    return df
