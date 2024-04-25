import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans

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


print("Skewness", df_variables.skew)
print("Kurtosis", df_variables.kurtosis)
