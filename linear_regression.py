import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split


characteristics_df = pd.read_csv('data/county_characteristics.csv', encoding = "ISO-8859-1")
social_capital_df = pd.read_csv('data/social_capital_county.csv')
characteristics_df.rename(columns={'cty': 'county'}, inplace=True)
full_df = pd.merge(social_capital_df, characteristics_df, on='county', how='outer')
processed_df = full_df
# remove columns which are not numerical
removed_columns = []
for col in processed_df.columns:
    if processed_df[col].dtypes == 'object':
        removed_columns.append(col)
for col in removed_columns:
    processed_df = processed_df.drop(col, axis=1)
# also remove county identifier, it's not relevant to a county's outcomes
processed_df = processed_df.drop('county', axis=1)
processed_df = processed_df.drop('mig_inflow', axis=1)
processed_df = processed_df.drop('state_id', axis=1)
processed_df = processed_df.drop('pop2018', axis=1)


# predict migration inflow rates, remove this variable from the matrix
y = processed_df['mig_outflow'].to_numpy()
# replace nan values with column average
inds = np.where(np.isnan(y))
y[inds] = np.nanmean(y)
# standardize y
y = (y - np.mean(y)) / np.std(y)
# create X matrix by dropping the dependent variable from the dataframe
processed_df = processed_df.drop('mig_outflow', axis=1)
X = processed_df.to_numpy()
# replace nan values with column averages
col_means = np.nanmean(X, axis=0)
inds = np.where(np.isnan(X))
X[inds] = np.take(col_means, inds[1])
# standardize matrix
X = X.T
for row in X:
    row_mean = np.mean(row)
    row_std = np.std(row)
    for i in range(len(row)):
        row[i] = (row[i] - row_mean) / row_std
# transpose back to original form
X = X.T

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# linear regression with l2 regularization
reg = LinearRegression().fit(X_train, y_train)
print(reg.score(X_test, y_test))

# linear regression coefficients
coefficients = pd.DataFrame({"Feature":processed_df.columns,"Coefficients":np.transpose(reg.coef_)})
coefficients = coefficients.sort_values('Coefficients', ascending=False)
print(coefficients.head(10))
















#
