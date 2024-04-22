from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib import font_manager
csfont = {'fontname':'cmr10'}
plt.rcParams.update({'font.size': 22})


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
    # also remove county name, state id, lat and lng identifiers, should not be considered in a county's outcomes
    processed_df = processed_df.drop('county', axis=1)
    processed_df = processed_df.drop('lat', axis=1)
    processed_df = processed_df.drop('lng', axis=1)
    processed_df = processed_df.drop('state_id', axis=1)
    return processed_df


processed_df = create_df()
# dependent_variables = list(processed_df.columns)
dependent_variables = ['mig_inflow', 'mig_outflow', 'child_ec_county', 'cs_fam_wkidsinglemom', 'exercise_any_q1']
nn_scores = {}
nn_lo_dropout_scores = {}
nn_hi_dropout_scores = {}
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
    # standardize matrix, transpose first
    X = X.T
    for row in X:
        row_mean = np.mean(row)
        row_std = np.std(row)
        for i in range(len(row)):
            row[i] = (row[i] - row_mean) / row_std
    # transpose back to original form
    X = X.T
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # design the neural network layout, no dropout
    num_input_vals = len(processed_df.columns)
    model = Sequential()
    model.add(Dense(100, input_shape=(num_input_vals,), activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    model.fit(X_train, y_train, epochs=50, batch_size=10)
    y_pred = model.predict(X_test)

    running_sum_u = 0
    for i in range(len(y_test)):
        running_sum_u += (y_test[i] - y_pred[i]) ** 2
    running_sum_v = 0
    for i in range(len(y_test)):
        running_sum_v += (y_test[i] - y_test.mean()) ** 2
    R_squared = 1 - running_sum_u / running_sum_v
    nn_scores[var] = R_squared[0]


    # design the neural network layout, with low dropout
    num_input_vals = len(processed_df.columns)
    model = Sequential()
    model.add(Dense(100, input_shape=(num_input_vals,), activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    model.fit(X_train, y_train, epochs=50, batch_size=10)
    y_pred = model.predict(X_test)

    running_sum_u = 0
    for i in range(len(y_test)):
        running_sum_u += (y_test[i] - y_pred[i]) ** 2
    running_sum_v = 0
    for i in range(len(y_test)):
        running_sum_v += (y_test[i] - y_test.mean()) ** 2

    R_squared = 1 - running_sum_u / running_sum_v
    nn_lo_dropout_scores[var] = R_squared[0]


    # design the neural network layout, with high dropout
    num_input_vals = len(processed_df.columns)
    model = Sequential()
    model.add(Dense(100, input_shape=(num_input_vals,), activation='relu'))
    model.add(Dropout(0.8))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    model.fit(X_train, y_train, epochs=50, batch_size=10)
    y_pred = model.predict(X_test)

    running_sum_u = 0
    for i in range(len(y_test)):
        running_sum_u += (y_test[i] - y_pred[i]) ** 2
    running_sum_v = 0
    for i in range(len(y_test)):
        running_sum_v += (y_test[i] - y_test.mean()) ** 2

    R_squared = 1 - running_sum_u / running_sum_v
    nn_hi_dropout_scores[var] = R_squared[0]



vars = ['mig_inflow', 'mig_outflow', 'child_ec_county', 'cs_fam_wkidsinglemom', 'exercise_any_q1']

nn_scores_y = [nn_scores[var] for var in vars]
nn_lo_dropout_scores_y = [nn_lo_dropout_scores[var] for var in vars]
nn_hi_dropout_scores_y = [nn_hi_dropout_scores[var] for var in vars]

# creating the bar plot
fig = plt.figure(figsize = (10, 5))
plt.style.use('ggplot')
width = 0.3
x = np.arange(len(vars))

# plot data in grouped manner of bar type
plt.bar(x-0.3, nn_scores_y, width, label='dropout=0.0')
plt.bar(x, nn_lo_dropout_scores_y, width, label='dropout=0.3')
plt.bar(x+0.3, nn_hi_dropout_scores_y, width, label='dropout=0.8')

xtick_labels = [xtick.upper() for xtick in vars]
plt.xticks(x, xtick_labels, rotation=32, ha='right', fontsize=10, **csfont)
plt.yticks(fontsize=10, **csfont)
plt.title("NN Prediction Scores", fontsize=18, **csfont)
plt.legend(loc='best', prop = {"family": 'cmr10' })
plt.xlabel("Dependent Variable Name", fontsize=16, **csfont)
plt.ylabel("Score", fontsize=16, **csfont)

# save figure
plt.savefig('./nn_dropout_with_nan.png', dpi=300, bbox_inches="tight")












#
