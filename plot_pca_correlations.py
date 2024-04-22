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

loading_matrix = loading_matrix.sort_values('PC1')
variables_PC1 = loading_matrix.index.tolist()
correlations_PC1 = loading_matrix['PC1'].tolist()

loading_matrix = loading_matrix.sort_values('PC2')
variables_PC2 = loading_matrix.index.tolist()
correlations_PC2 = loading_matrix['PC2'].tolist()


X1 = variables_PC1[:10]
X2 = variables_PC1[-10:][::-1]
Y1 = correlations_PC1[:10]
Y2 = correlations_PC1[-10:][::-1]
X_axis = np.arange(len(X1))
plt.style.use('ggplot')
plt.bar(X_axis, Y1, 0.4)
plt.xticks(X_axis, X1, rotation=30)
plt.xlabel("Variable Name")
plt.ylabel("First Dimension Correlation")
plt.title("Variables with Lowest Correlation in First PCA Dimension")
plt.savefig('./pca_1_with_nan_lowest_correlations.png', dpi=300, bbox_inches="tight")
plt.close()

plt.bar(X_axis, Y2, 0.4)
plt.xticks(X_axis, X2, rotation=30)
plt.xlabel("Variable Name")
plt.ylabel("First Dimension Correlation")
plt.title("Variables with Highest Correlation in First PCA Dimension")
plt.savefig('./pca_1_with_nan_highest_correlations.png', dpi=300, bbox_inches="tight")
plt.close()


X1 = variables_PC2[:10]
X2 = variables_PC2[-10:][::-1]
Y1 = correlations_PC2[:10]
Y2 = correlations_PC2[-10:][::-1]
X_axis = np.arange(len(X1))
plt.style.use('ggplot')
plt.bar(X_axis, Y1, 0.4)
plt.xticks(X_axis, X1, rotation=30)
plt.xlabel("Variable Name")
plt.ylabel("First Dimension Correlation")
plt.title("Variables with Lowest Correlation in Second PCA Dimension")
plt.savefig('./pca_2_with_nan_lowest_correlations.png', dpi=300, bbox_inches="tight")
plt.close()

plt.bar(X_axis, Y2, 0.4)
plt.xticks(X_axis, X2, rotation=30)
plt.xlabel("Variable Name")
plt.ylabel("First Dimension Correlation")
plt.title("Variables with Highest Correlation in Second PCA Dimension")
plt.savefig('./pca_2_with_nan_highest_correlations.png', dpi=300, bbox_inches="tight")
plt.close()
