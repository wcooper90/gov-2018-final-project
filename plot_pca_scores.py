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

# create matrix
matrix = processed_df.to_numpy()
# replace nan values with column averages
col_means = np.nanmean(matrix, axis=0)
inds = np.where(np.isnan(matrix))
matrix[inds] = np.take(col_means, inds[1])

# standardize matrix
matrix = matrix.T
for row in matrix:
    row_mean = np.mean(row)
    row_std = np.std(row)
    for i in range(len(row)):
        row[i] = (row[i] - row_mean) / row_std
# transpose back to original form
matrix = matrix.T


# apply PCA
pca = decomposition.PCA(n_components=4)
X = pca.fit_transform(matrix)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
loading_matrix = pd.DataFrame(loadings, columns=['PC1', 'PC2', 'PC3', 'PC4'], index=processed_df.columns)



full_df['PC1_score'] = 0
full_df['PC2_score'] = 0


PC1_scores = matrix.dot(pca.components_[0])
for i, score in enumerate(PC1_scores):
    full_df.loc[i, 'PC1_score'] = score


PC2_scores = matrix.dot(pca.components_[1])
for i, score in enumerate(PC2_scores):
    full_df.loc[i, 'PC2_score'] = score



# set up plotting data
x = full_df['lng'].tolist()
y = full_df['lat'].tolist()
PC1 = full_df['PC1_score'].tolist()
PC2 = full_df['PC2_score'].tolist()


plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = [14, 8]


fig = plt.figure()
for i in range(len(x)):
    color = 'green'
    if PC1[i] < 0:
        color = 'orange'
    plt.scatter(x[i], y[i], color=color)

plt.title("Positive and Negative First Dimension Scores by County Location", fontsize=18, **csfont)
plt.xlabel('Longitude', fontsize=12, **csfont)
plt.ylabel('Latitude', fontsize=12, **csfont)
plt.savefig('./pca_1_with_nan.png', dpi=300, bbox_inches="tight")
plt.close()



# fig = plt.figure()
# for i in range(len(x)):
#     color = 'green'
#     if PC2[i] < 0:
#         color = 'orange'
#     plt.scatter(x[i], y[i], color=color)
#
# plt.title("Positive and Negative Second Dimension Scores by County Location")
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.savefig('./pca_2_with_nan.png')
# plt.close()
#


#
