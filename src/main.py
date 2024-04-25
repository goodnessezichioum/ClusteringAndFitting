import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
from scipy.stats import t

df = pd.read_csv('./res/1- mental-illnesses-prevalence.csv')

print(df.head())
print(df.describe())


def dataframe_info_as_table(df):
    """
    Create a summary table of information about a dataframe including the variable names,
    data types, total count of values, count of unique values, and number of missing values
    for each column.

    Parameters:
    df (DataFrame): The dataframe to summarize.

    Returns:
    DataFrame: A new dataframe containing the summary information.
    """
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
def plot_sub(df, column_pairs):
    """
    Plot scatterplots for specified pairs of columns in a 2x2 subplot layout.

    Parameters:
    df (DataFrame): The dataframe containing the columns to plot.
    column_pairs (list of tuple): Pairs of column names to be plotted against each other.
    """
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


# plot_sub(df_variables, column_pairs)

# plot correlation matrix
def plot_correlation_matrix(df, title):
    """
    Generate and display a heatmap of the correlation matrix for the specified dataframe.

    Parameters:
    df (DataFrame): The dataframe whose correlations are to be plotted.
    title (str): The title of the plot.
    """
    corr = df.corr()
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, cmap=cmap, vmax=.3, center=0, annot=True, fmt=".2f", linewidths=.5)
    plt.title(title)
    plt.show()


plot_correlation_matrix(df_variables, "Correlation Matrix")

# Standardizing the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_variables)

# setting up the clusterer
kmeans = KMeans(init="random", n_clusters=3, n_init=10, max_iter=300, random_state=42)
cluster_km = kmeans.fit(df_scaled)

kmeans_kw = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}


# Checking best cluster numbers with the elbow method
def plot_elbow_method(df, iner, title):
    """
    Plot the elbow method graph for determining the optimal number of clusters in k-means clustering.

    Parameters:
    df (array): The scaled data used in k-means clustering.
    iner (list): List of inertia values corresponding to different numbers of clusters.
    title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 5), dpi=200)
    plt.plot(range(1, 10), iner, color='purple')
    plt.xticks(range(1, 10))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Iner")
    plt.title(title)
    plt.axvline(x=3, color='b', label='axvline - full height', linestyle="dashed")
    plt.show()
    return


iner = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, **kmeans_kw)
    kmeans.fit(df_scaled)
    iner.append(kmeans.inertia_)

plot_elbow_method(df_scaled, iner, "Elbow plot")

kmeans = KMeans(n_clusters=3, init='random', n_init=10, max_iter=300, random_state=42)


def perform_clustering(data):
    """
    Perform k-means clustering on the provided data using a pre-initialized k-means model.

    Parameters:
    data (array): The dataset to cluster.

    Returns:
    array: Array of cluster labels for each data point.
    """
    kmeans.fit(data)
    return kmeans.labels_


labels_Km = perform_clustering(df_scaled)


def plot_scatter(df, x, y, labels, centers, scaler):
    """Plot scatter graph with clustering and labeled cluster centers."""
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=df[x], y=df[y], hue=labels, palette='viridis', s=100, alpha=0.6)

    # Transform cluster centers back to original scale
    centers_original_scale = scaler.inverse_transform(centers)

    # Extract x and y coordinates of the centers
    x_centers = centers_original_scale[:, df.columns.get_loc(x)]
    y_centers = centers_original_scale[:, df.columns.get_loc(y)]

    # Plot the centers and create a legend entry for them
    for i, (cx, cy) in enumerate(zip(x_centers, y_centers)):
        plt.scatter(cx, cy, s=300, c='black', marker='X')  # Large black crosses for centers
        plt.text(cx, cy, f'Center {i}', color='black', fontsize=12, weight='bold')
        plt.scatter([], [], c='black', marker='X', label=f'Center {i}')

    plt.title(f'Scatter Plot of {x} vs {y} with Clustering and Centers')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend(title='With clustering')
    plt.show()


# scatterplots with clustering
plot_scatter(df_variables, 'Schizophrenia disorders', 'Depressive disorders', labels_Km,
             kmeans.cluster_centers_, scaler)
plot_scatter(df_variables, 'Depressive disorders', 'Anxiety disorders', labels_Km,
             kmeans.cluster_centers_, scaler)
plot_scatter(df_variables, 'Anxiety disorders', 'Bipolar disorders', labels_Km,
             kmeans.cluster_centers_, scaler)
plot_scatter(df_variables, 'Bipolar disorders', 'Eating disorders', labels_Km,
             kmeans.cluster_centers_, scaler)
plot_scatter(df_variables, 'Eating disorders', 'Schizophrenia disorders', labels_Km,
             kmeans.cluster_centers_, scaler)

# Prediction

# Assume the last 5 entries are "new" data
new_data_scaled = scaler.transform(df_variables.tail(5))

# Making predictions on the new data
predictions = kmeans.predict(new_data_scaled)
print("Predictions: ", predictions)


# Fitting

def poly_model(x, *params):
    """Polynomial model for curve fitting."""
    return sum(p * (x ** i) for i, p in enumerate(params))


def fit_polynomial(x, y, degree=4):
    """Fit a polynomial to the data and return the fit parameters and standard errors."""
    initial_guess = [1] * (degree + 1)
    popt, pcov = curve_fit(poly_model, x, y, p0=initial_guess)
    perr = np.sqrt(np.diag(pcov))  # Standard errors of the parameters
    return popt, perr


def calculate_confidence_interval3(X, popt, perr, confidence=0.95):
    """Calculate the confidence interval for the polynomial fit."""
    alpha = 1 - confidence
    t_val = t.ppf(1 - alpha / 2, len(X) - len(popt))
    x_fit = np.linspace(X.min(), X.max(), 200)
    y_fit = poly_model(x_fit, *popt)

    ci = t_val * perr * np.sqrt(1 / len(X) + (x_fit - np.mean(X)) ** 2 / np.sum((X - np.mean(X)) ** 2))
    return x_fit, y_fit, ci


def calculate_confidence_interval(X, popt, perr, confidence=0.95):
    """Calculate the confidence interval for the polynomial fit."""
    alpha = 1 - confidence
    t_val = t.ppf(1 - alpha / 2, len(X) - len(popt))
    x_fit = np.linspace(X.min(), X.max(), 200)
    y_fit = poly_model(x_fit, *popt)

    mean_sq_deviation = np.mean((X - np.mean(X)) ** 2)

    # Calculate the confidence interval using the mean squared deviation
    ci = t_val * perr[-1] * np.sqrt(1 / len(X) + (x_fit - np.mean(X)) ** 2 / mean_sq_deviation)
    return x_fit, y_fit, ci


def plot_fit_with_confidence(X, y, x_fit, y_fit, ci, degree=4, confidence=0.95):
    """Plot the data, the polynomial fit, and the confidence interval."""
    plt.figure(figsize=(10, 8))
    plt.scatter(X, y, label='Data', color='blue', alpha=0.5)
    plt.plot(x_fit, y_fit, color='red', label=f'Polynomial Degree {degree} Fit')
    plt.fill_between(x_fit, y_fit - ci, y_fit + ci, color='red', alpha=0.2,
                     label=f'{confidence * 100}% Confidence Interval')
    plt.xlabel('Anxiety disorders')
    plt.ylabel('Depressive disorders')
    plt.title('Polynomial Curve Fit with Confidence Interval')
    plt.legend()
    plt.show()


# Usage:
# Extract the columns for fitting
x = df_variables['Anxiety disorders'].values
y = df_variables['Depressive disorders'].values

# Check if there are any NaN values and remove them
nan_mask = ~np.isnan(x) & ~np.isnan(y)
x = x[nan_mask]
y = y[nan_mask]

degree = 4

# Fit the polynomial and get the parameters and errors
popt, perr = fit_polynomial(x, y, degree)

# Calculate the confidence interval
x_fit, y_fit, ci = calculate_confidence_interval(x, popt, perr)

# Plot the results
plot_fit_with_confidence(x, y, x_fit, y_fit, ci, degree)


def make_predictions(new_x, fitted_params):
    """Make predictions using the fitted polynomial model."""
    predicted_y = poly_model(new_x, *fitted_params)
    return predicted_y


# Predict the next 5 'Depressive disorders' from 'Anxiety disorders'
# Assume the last 5 entries are "new" data
new_x_values = np.array(df_variables["Anxiety disorders"].tail(5))  # Replace with actual new values

# Making predictions on the new data for 'Depressive disorders'
predictions = make_predictions(new_x_values, popt)
print("Fitting Predictions for Depressive Disorder:", predictions)
