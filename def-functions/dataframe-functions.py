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

