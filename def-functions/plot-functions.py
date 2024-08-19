########################################
#### Plot Functions ####################
########################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def plot_histogram(df, column, bins=20):
    plt.hist(df[column], bins=bins)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {column}')
    plt.show()



def plot_boxplot(df, column):
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot of {column}')
    plt.show()



def plot_correlation_matrix(df, annot=True):
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=annot)
    plt.title('Correlation Matrix')
    plt.show()

