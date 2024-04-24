import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans

df= pd.read_csv('./res/1- mental-illnesses-prevalence.csv')
print(df.head())