# plotting libraries
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
import numpy as np
from pprint import pprint
from tqdm import tqdm


csfont = {'fontname':'cmr10'}
plt.rcParams.update({'font.size': 22})


import scipy
from sklearn import decomposition
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score


def create_df():
    characteristics_df = pd.read_csv('data/county_characteristics.csv', encoding = "ISO-8859-1")
    social_capital_df = pd.read_csv('data/social_capital_county.csv')
    location_df = pd.read_csv('data/uscounties.csv', usecols=['county_fips', 'lat', 'lng'])
    characteristics_df.rename(columns={'cty': 'county'}, inplace=True)
    location_df.rename(columns={'county_fips': 'county'}, inplace=True)
    full_df = pd.merge(social_capital_df, characteristics_df, on='county', how='outer')
    full_df = pd.merge(full_df, location_df, on='county', how='outer')
    processed_df = full_df
    # remove columns which are not numerical
    removed_columns = []
    for col in processed_df.columns:
        if processed_df[col].dtypes == 'object':
            removed_columns.append(col)
    for col in removed_columns:
        processed_df = processed_df.drop(col, axis=1)
    # also remove county, lat and lng identifiers, it's not relevant to a county's outcomes
    processed_df = processed_df.drop('county', axis=1)
    processed_df = processed_df.drop('lat', axis=1)
    processed_df = processed_df.drop('lng', axis=1)
    processed_df = processed_df.drop('state_id', axis=1)
    return processed_df


processed_df = create_df()
# dependent_variables = list(processed_df.columns)
dependent_variables = ['mig_inflow', 'mig_outflow', 'child_ec_county', 'cs_fam_wkidsinglemom', 'exercise_any_q1']
linear_regression_scores = {}
ridge_regression_scores_lo_l2 = {}
ridge_regression_scores_hi_l2 = {}
for var in tqdm(dependent_variables):
    processed_df = create_df()
    # predict migration inflow rates, remove this variable from the matrix
    y = processed_df[var].to_numpy()
    # replace nan values with column average
    inds = np.where(np.isnan(y))
    y[inds] = np.nanmean(y)
    # standardize y
    y = (y - np.mean(y)) / np.std(y)
    # create X matrix by dropping the dependent variable from the dataframe
    processed_df = processed_df.drop(var, axis=1)
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
    # print("*"*80)
    # print("Dependent variable: {var}".format(var=var))
    # print(reg.score(X_test, y_test))
    linear_regression_scores[var] = reg.score(X_test, y_test)
    # linear regression coefficients
    reg_coefficients = pd.DataFrame({"Feature":processed_df.columns,"Coefficients":np.transpose(reg.coef_)})
    # print(reg_coefficients.head())


    ridge = Ridge(alpha=1)
    ridge.fit(X_train, y_train)
    ridge_regression_scores_lo_l2[var] = ridge.score(X_test, y_test)

    ridge = Ridge(alpha=1000)
    ridge.fit(X_train, y_train)
    ridge_regression_scores_hi_l2[var] = ridge.score(X_test, y_test)



pprint(ridge_regression_scores_hi_l2)
print("*"*80)
pprint(ridge_regression_scores_lo_l2)
print("*"*80)
pprint(linear_regression_scores)



vars = ['mig_inflow', 'mig_outflow', 'child_ec_county', 'cs_fam_wkidsinglemom', 'exercise_any_q1']

linear_regression_scores_y = [linear_regression_scores[var] for var in vars]
ridge_regression_scores_hi_l2_y = [ridge_regression_scores_hi_l2[var] for var in vars]
ridge_regression_scores_lo_l2_y = [ridge_regression_scores_lo_l2[var] for var in vars]

# creating the bar plot
fig = plt.figure(figsize = (10, 5))
plt.style.use('ggplot')
width = 0.3
x = np.arange(len(vars))

# plot data in grouped manner of bar type
plt.bar(x-0.3, linear_regression_scores_y, width, label='linear regression')
plt.bar(x, ridge_regression_scores_lo_l2_y, width, label='ridge, l2=1')
plt.bar(x+0.3, ridge_regression_scores_hi_l2_y, width, label='ridge, l2=1000')


xtick_labels = [xtick.upper() for xtick in vars]
plt.xticks(x, xtick_labels, rotation=32, ha='right', fontsize=10, **csfont)
plt.yticks(fontsize=10, **csfont)
plt.title("Regression Scores", fontsize=18, **csfont)
plt.legend(loc='best', prop = {"family": 'cmr10' })
plt.xlabel("Dependent Variable Name", fontsize=16, **csfont)
plt.ylabel("Score", fontsize=16, **csfont)

# save figure
plt.savefig('./regressions_with_nan.png', dpi=300, bbox_inches="tight")












#
