########################################
#### Dataframe Functions ###############
########################################

import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from scipy import stats
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)


def calculate_correlation(df: pd.DataFrame, column1: str, column2: str) -> float:
    return df[column1].corr(df[column2])


def encode_categorical_features(df: pd.DataFrame, columns: List) -> pd.DataFrame:
    return pd.get_dummies(df, columns=columns)


def describe_numerical_columns(df: pd.DataFrame, columns: List) -> pd.DataFrame:
    return df[columns].describe()


def subsample_set(n, df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(n, int):
        logger.info("-" * 20)
        logger.info("Subsampling Data ENABLED!")
        logger.info("New dataset number of row(s) = {}".format(n))
        n_subsampling = min(n, df.shape[0])
        df = df[0:n_subsampling]
    else:
        logger.info("Subsampling Data DISABLED!")
    return df


def log_stats_df(df: pd.DataFrame, df_label=None) -> None:
    logger.info("-" * 20)

    if df_label:
        logger.info("Log Stats dataframe {}".format(df_label))

    logger.info("Check NA: \n{}".format(df.isna().sum()))

    logger.info(
        "Check duplicates: \n{} rows are duplicated".format(
            df.shape[0] - df.drop_duplicates(subset=None, keep="first", ignore_index=False).shape[0]
        )
    )

    logger.info("dataframe has %s row(s)", str(df.shape[0]))

    logger.info("-" * 20)


def get_train_test_val_sets(
    train_path: str, test_path: str, val_path: str, num_samples=None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    """
    Load Train, Test and Validation data
    """

    # Train Data
    try:
        logger.info("-" * 20)
        logger.info("Loading Train data...")

        X_train = pd.read_csv(os.path.join(train_path, "X_train.csv"), sep=";")
        y_train = pd.read_csv(os.path.join(train_path, "y_train.csv"), sep=";")

        X_train = subsample_set(n=num_samples, df=X_train)
        y_train = subsample_set(n=num_samples, df=y_train)

        logger.info("Correctly loaded Train data!")
        logger.info("Train set has %s rows", str(X_train.shape[0]))
        logger.info("Train data Sample: \n {}".format(X_train.sample(5)))
    except Exception as e:
        logger.info("Error loading Train data: %s", e)
        logger.info("Train data not found!")

        X_train = None
        y_train = None

    # Test Data
    try:
        logger.info("-" * 20)
        logger.info("Loading Test data...")

        X_test = pd.read_csv(os.path.join(test_path, "X_test.csv"), sep=";")
        y_test = pd.read_csv(os.path.join(test_path, "y_test.csv"), sep=";")

        X_test = subsample_set(n=num_samples, df=X_test)
        y_test = subsample_set(n=num_samples, df=y_test)

        logger.info("Correctly loaded Test data!")
        logger.info("Test set has %s rows", str(X_test.shape[0]))
        logger.info("Test data Sample: \n {}".format(X_test.sample(5)))
    except Exception as e:
        logger.info("Error loading Test data: %s", e)
        logger.info("Test data not found!")

        X_test = None
        y_test = None

    # Val Data
    try:
        logger.info("-" * 20)
        logger.info("Loading Val data...")

        X_val = pd.read_csv(os.path.join(val_path, "X_val.csv"), sep=";")
        y_val = pd.read_csv(os.path.join(val_path, "y_val.csv"), sep=";")

        X_val = subsample_set(n=num_samples, df=X_val)
        y_val = subsample_set(n=num_samples, df=y_val)

        logger.info("Correctly loaded Val data!")
        logger.info("Validation set has %s rows", str(X_val.shape[0]))
        logger.info("Val data Sample: \n {}".format(X_val.sample(5)))
    except Exception as e:
        logger.info("Error loading Val data: %s", e)
        logger.info("Val data not found!")

        X_val = None
        y_val = None

    logger.info("-" * 20)

    return X_train, y_train, X_test, y_test, X_val, y_val


def analyze_column(df: pd.DataFrame, column_name: str) -> Dict:
    column_data = df[column_name]

    analysis = {
        "column_name": column_name,
        "data_type": column_data.dtype,
        "count": column_data.count(),
        "missing_values": column_data.isnull().sum(),
        "unique_values": column_data.nunique(),
        "top_values": column_data.value_counts().head(5),
        "summary_statistics": column_data.describe(),
    }

    if np.issubdtype(column_data.dtype, np.number):
        analysis["mean"] = column_data.mean()
        analysis["median"] = column_data.median()
        analysis["std"] = column_data.std()
        analysis["min"] = column_data.min()
        analysis["max"] = column_data.max()
        analysis["quartiles"] = column_data.quantile([0.25, 0.5, 0.75])

    return analysis


def impute_missing_values(df: pd.DataFrame, column: str, strategy="mean") -> pd.DataFrame:
    if strategy == "mean":
        return df[column].fillna(df[column].mean(), inplace=True)
    elif strategy == "median":
        return df[column].fillna(df[column].median(), inplace=True)
    elif strategy == "mode":
        return df[column].fillna(df[column].mode()[0], inplace=True)
    else:
        raise ValueError("Invalid strategy for imputation.")


def preprocess_suggestion(column: pd.Series) -> str:
    suggestion_ = ""
    data_type_ = str(column.dtype)
    if data_type_.startswith("int") or data_type_.startswith("float"):
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

    elif data_type_ == "object":
        if column.isnull().sum() > 0:
            suggestion_ += "Fill missing values with: mode or 'unknown' category if categorical.\n"
            suggestion_ += "Fill missing values with: empty string or 'missing' value (if text).\n"

        if len(column.unique()) / len(column) > 0.1:
            suggestion_ += "High cardinality. Consider feature hashing, target encoding, or frequency encoding.\n"

        if not pd.api.types.is_datetime64_dtype(column):
            suggestion_ += "Check for consistent formatting in text data.\n"

        if column.str.contains(r"[^a-zA-Z\s]").any():
            suggestion_ += "Consider removing noise (numbers, special characters).\n"

        if column.str.len().mean() > 100:
            suggestion_ += "Text length is relatively long. Consider text summarization or truncation.\n"

        if column.apply(lambda x: len(set(x.split())) / len(x.split())).mean() < 0.5:
            suggestion_ += "Text contains repeated words. Consider deduplication or stemming.\n"

    elif data_type_ == "datetime64[ns]":
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


def handle_missing_values(df, strategy_="drop", fill_val=0, impute_strategy="most_frequent"):
    if strategy_ == "drop":
        return df.dropna()
    elif strategy_ == "fill":
        return df.fillna(fill_val)
    elif strategy_ == "impute":
        imp = SimpleImputer(strategy=impute_strategy)
        return imp.fit_transform(df)
    else:
        raise ValueError("Invalid strategy for handling missing values.")


def handle_outliers(df, columns, method="zscore", threshold=3):
    if method == "zscore":
        for col in columns:
            z_scores = stats.zscore(df[col])
            df = df[abs(z_scores) < threshold]
    elif method == "iqr":  # Options are 'zscore', 'iqr'.
        # TO DO Implement IQR-based outlier detection
        pass
    else:
        raise ValueError("Invalid method for outlier detection.")
    return df


def unroll_vector_column(df, column_name, drop_orig_column=False):
    # this function is useful for unrolling vectors obtained with w2v algorithms
    # i.e. [x1,x2,x3,x4,...,x20] -> x1 | x2 | x3 | x4 | ... | x20
    logger.info("-" * 20)
    logger.info("Unrolling {} column".format(column_name))
    numpy_data = np.array(df[column_name])
    df_out = pd.DataFrame(data=numpy_data)
    df_out.columns = [f"vector{i}" for i in range(0, len(df_out.T))]
    df = pd.concat([df, df_out], axis=1)
    logger.info("Column {} correctly unrolled".format(column_name))
    logger.info("Unrolled vector sample:\n {}".format(df_out.sample(5)))
    logger.info("-" * 20)
    if drop_orig_column:
        logger.info("Drop original column {} ENABLED".format(column_name))
        try:
            df = df.drop(column_name, axis=1)
            logger.info("{} correctly dropped!".format(column_name))
        except Exception as e:
            logger.info(e)
        logger.info("-" * 20)
    return df
