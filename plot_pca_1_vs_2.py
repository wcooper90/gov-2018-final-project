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
# also remove county, lat and lng identifiers, it's not relevant to a county's outcomes
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
# correlations of each variable to the first four PCs

PC1_scores = matrix.dot(pca.components_[0])
PC2_scores = matrix.dot(pca.components_[1])




plt.style.use('ggplot')
plt.scatter(PC1_scores, PC2_scores, 5)
plt.xlabel("PC1 Score", fontsize=16, **csfont)
plt.ylabel("PC2 Score", fontsize=16, **csfont)
plt.title("First Component vs. Second Component Scores by County", fontsize=18, **csfont)
plt.savefig('./pca_1_with_nan_lowest_correlations.png', dpi=300, bbox_inches="tight")
plt.close()
