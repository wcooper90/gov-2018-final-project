# plotting libraries
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
import numpy as np

csfont = {'fontname':'cmr10'}
plt.rcParams.update({'font.size': 22})


import scipy
from sklearn import decomposition
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score

characteristics_df = pd.read_csv('data/county_characteristics.csv', encoding = "ISO-8859-1")
social_capital_df = pd.read_csv('data/social_capital_county.csv')
location_df = pd.read_csv('data/uscounties.csv', usecols=['county_fips', 'lat', 'lng'])

characteristics_df.rename(columns={'cty': 'county'}, inplace=True)
location_df.rename(columns={'county_fips': 'county'}, inplace=True)
full_df = pd.merge(social_capital_df, characteristics_df, on='county', how='outer')
full_df = pd.merge(full_df, location_df, on='county', how='outer')
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
processed_df = processed_df.drop('lat', axis=1)
processed_df = processed_df.drop('lng', axis=1)


# predict migration inflow rates, remove this variable from the matrix
y = processed_df['mig_inflow'].to_numpy()
y = (y - np.mean(y)) / np.std(y)
processed_df = processed_df.drop('mig_inflow', axis=1)
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# linear regression with l2 regularization
reg = LinearRegression().fit(X_train, y_train)
print(reg.score(X_test, y_test))

# linear regression coefficients
reg_coefficients = pd.DataFrame({"Feature":processed_df.columns,"Coefficients":np.transpose(reg.coef_)})
print(reg_coefficients.head())


# linear regression with l2 regularization
# Loop to compute the different values of cross-validation scores
cross_val_scores_ridge = []
alpha = []
for i in range(0, 6):
    ridge = Ridge(alpha = i * 10)
    ridge.fit(X_train, y_train)
    scores = cross_val_score(ridge, X, y, cv = 10)
    avg_cross_val_score = np.mean(scores)*100
    cross_val_scores_ridge.append(avg_cross_val_score)
    alpha.append(i * 10)

# Loop to print the different values of cross-validation scores
for i in range(0, len(alpha)):
    print(str(alpha[i])+' : '+str(cross_val_scores_ridge[i]))


#
# ridge = Ridge().fit(X_train, y_train)
# print(ridge.score(X_test, y_test))
#
# # linear regression coefficients
# ridge_coefficients = pd.DataFrame({"Feature":processed_df.columns,"Coefficients":np.transpose(ridge.coef_)})
# print(ridge_coefficients.head())
#


















#
