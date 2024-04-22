import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd


characteristics_df = pd.read_csv('data/county_characteristics.csv', encoding = "ISO-8859-1")
social_capital_df = pd.read_csv('data/social_capital_county.csv')
characteristics_df.rename(columns={'cty': 'county'}, inplace=True)
full_df = pd.merge(social_capital_df, characteristics_df, on='county', how='outer')
# remove nan values
no_nan_df = full_df.dropna()
processed_df = no_nan_df
# remove columns which are not numerical
removed_columns = []
for col in processed_df.columns:
    if processed_df[col].dtypes == 'object':
        removed_columns.append(col)
for col in removed_columns:
    processed_df = processed_df.drop(col, axis=1)
# also remove county identifier, it's not relevant to a county's outcomes
processed_df = processed_df.drop('county', axis=1)


# predict migration inflow rates, remove this variable from the matrix
y = processed_df['primcarevis_10'].to_numpy()
y = (y - np.mean(y)) / np.std(y)
processed_df = processed_df.drop('primcarevis_10', axis=1)
X = processed_df.to_numpy()
# standardize matrix
X = X.T
for row in X:
    row_mean = np.mean(row)
    row_std = np.std(row)
    for i in range(len(row)):
        row[i] = (row[i] - row_mean) / row_std
# transpose back to original form
X = X.T


# linear regression with l2 regularization
reg = LinearRegression().fit(X, y)
print(reg.score(X, y))





















#
