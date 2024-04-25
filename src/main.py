import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

df = pd.read_csv('./res/1- mental-illnesses-prevalence.csv')

print(df.head())
print(df.describe())

def dataframe_info_as_table(df):
    variables = []
    dtypes = []
    count = []
    unique = []
    missing = []

    for item in df.columns:
        variables.append(item)
        dtypes.append(df[item].dtype)
        count.append(len(df[item]))
        unique.append(len(df[item].unique()))
        missing.append(df[item].isna().sum())

    output = pd.DataFrame({
        'variable': variables,
        'dtype': dtypes,
        'count': count,
        'unique': unique,
        'missing value': missing
    })
    return output


print("The describe table of df : Mental illness")
print(dataframe_info_as_table(df))

df = df.rename(columns={
    'Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized': 'Schizophrenia disorders',
    'Depressive disorders (share of population) - Sex: Both - Age: Age-standardized': 'Depressive disorders',
    'Anxiety disorders (share of population) - Sex: Both - Age: Age-standardized': 'Anxiety disorders',
    'Bipolar disorders (share of population) - Sex: Both - Age: Age-standardized': 'Bipolar disorders',
    'Eating disorders (share of population) - Sex: Both - Age: Age-standardized': 'Eating disorders'})

# Get a dataframe of data that we are concerned about
df_variables = df[["Schizophrenia disorders", "Depressive disorders", "Anxiety disorders", "Bipolar disorders",
                   "Eating disorders"]]


print("Skewness", df_variables.skew())
print("Kurtosis", df_variables.kurtosis())


# create scatterplots
def plot_scatter_plots(df, column_pairs):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11, 5))
    axes = axes.flatten()
    titles = ['Schizophrenia - Eating', 'Depressive - Eating', 'Anxiety - Eating', 'Bipolar - Eating']
    for ax, (col_x, col_y), title in zip(axes, column_pairs, titles):
        ax.set_title(title)
        sns.scatterplot(x=col_x, y=col_y, data=df, ax=ax)
    plt.show()


column_pairs = [
    ("Schizophrenia disorders", "Eating disorders"),
    ("Depressive disorders", "Eating disorders"),
    ("Anxiety disorders", "Eating disorders"),
    ("Bipolar disorders", "Eating disorders")
]


plot_scatter_plots(df_variables, column_pairs)

# plot correlation matrix
def plot_correlation_matrix(df):
    corr = df.corr()
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, cmap=cmap, vmax=.3, center=0, annot=True, fmt=".2f", linewidths=.5)
    plt.show()


plot_correlation_matrix(df_variables)

