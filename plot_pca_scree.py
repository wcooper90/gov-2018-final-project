import pandas as pd
import scipy
import numpy as np
from sklearn import decomposition
from sklearn import datasets
from sklearn.preprocessing import scale

import matplotlib.pyplot as plt
from matplotlib import font_manager
csfont = {'fontname':'cmr10'}
plt.rcParams.update({'font.size': 22})



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
pca = decomposition.PCA(n_components=10)
X = pca.fit_transform(matrix)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
loading_matrix = pd.DataFrame(loadings, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10'], index=processed_df.columns)
# correlations of each variable to the first four PCs
print(loading_matrix.head(10))
print(pca.explained_variance_)



plt.style.use('ggplot')

components = np.arange(1, 11)
variance_explained = pca.explained_variance_
plt.plot(components, variance_explained, 'o-', linewidth=2, color='blue')
plt.title('PCA Scree Plot', fontsize=18, **csfont)
plt.xlabel('Principal Component', fontsize=16, **csfont)
plt.ylabel('Variance Explained (%)', fontsize=16, **csfont)
plt.yticks(fontsize=10, **csfont)
plt.xticks(fontsize=10, **csfont)



plt.savefig('./scree_plot.png', dpi=300, bbox_inches="tight")










#
