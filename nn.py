from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


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

# predict migration inflow rates, remove this variable from the matrix
y = processed_df['exercise_any_q1'].to_numpy()
# replace nan values with column average
inds = np.where(np.isnan(y))
y[inds] = np.nanmean(y)
# standardize y
y = (y - np.mean(y)) / np.std(y)
# create X matrix by dropping the dependent variable from the dataframe
processed_df = processed_df.drop('exercise_any_q1', axis=1)
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


# design the neural network layout
num_input_vals = len(processed_df.columns)
model = Sequential()
model.add(Dense(100, input_shape=(num_input_vals,), activation='relu'))
model.add(Dense(160, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(80, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])


model.fit(X_train, y_train, epochs=80, batch_size=10)
y_pred = model.predict(X_test)

running_sum_u = 0
for i in range(len(y_test)):
    running_sum_u += (y_test[i] - y_pred[i]) ** 2
running_sum_v = 0
for i in range(len(y_test)):
    running_sum_v += (y_test[i] - y_test.mean()) ** 2

R_squared = 1 - running_sum_u / running_sum_v
print("R squared score: {r2}".format(r2=R_squared))

# _, accuracy = model.evaluate(X, y)
# print('Accuracy: %.2f' % (accuracy*100))















#
