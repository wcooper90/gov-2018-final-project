import pandas as pd
import scipy
import numpy as np
from sklearn import decomposition
from sklearn import datasets
from sklearn.preprocessing import scale

characteristics_df = pd.read_csv('data/county_characteristics.csv', encoding = "ISO-8859-1")
social_capital_df = pd.read_csv('data/social_capital_county.csv')
characteristics_df.rename(columns={'cty': 'county'}, inplace=True)
full_df = pd.merge(social_capital_df, characteristics_df, on='county', how='outer')
# PCA on dataframe with no nan values
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
# create matrix
matrix = processed_df.to_numpy().T
# standardize matrix
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
print(loading_matrix.head(10))
print(pca.explained_variance_)

print("*"*80)
print("PC1 highest correlations")
PC1_correlations = loading_matrix['PC1'].tolist()
top_10_percentile = np.percentile(PC1_correlations, 90)
bottom_10_percentile = np.percentile(PC1_correlations, 10)
counter = 0
for i, score in enumerate(PC1_correlations):
    if score > top_10_percentile:
        print(i, list(processed_df.columns)[i])
        counter += 1
    if counter > 15:
        break
print("*"*80)
print("PC1 lowest correlations")
counter = 0
for i, score in enumerate(PC1_correlations):
    if score < bottom_10_percentile:
        print(i, list(processed_df.columns)[i])
        counter += 1
    if counter > 15:
        break

print("*"*80)
print("PC2 highest correlations")
counter = 0
PC2_correlations = loading_matrix['PC2'].tolist()
top_10_percentile = np.percentile(PC2_correlations, 90)
bottom_10_percentile = np.percentile(PC2_correlations, 10)
for i, score in enumerate(PC2_correlations):
    if score > top_10_percentile:
        print(i, list(processed_df.columns)[i])
        counter += 1
    if counter > 15:
        break
print("*"*80)
print("PC2 lowest correlations")
counter = 0
for i, score in enumerate(PC2_correlations):
    if score < bottom_10_percentile:
        print(i, list(processed_df.columns)[i])
        counter += 1
    if counter > 15:
        break




# highest and lowest county scores
print("*"*80)
print("PC1 highest scores")
PC1_scores = matrix.dot(pca.components_[0])
top_10_percentile = np.percentile(PC1_scores, 90)
bottom_10_percentile = np.percentile(PC1_scores, 10)
counter = 0
for i, score in enumerate(PC1_scores):
    if score > top_10_percentile:
        print(i, no_nan_df[i:i + 1]['county_name_x'].iloc[0])
        counter += 1
    if counter > 15:
        break
print("*"*80)
print("PC1 lowest scores")
counter = 0
for i, score in enumerate(PC1_scores):
    if score < bottom_10_percentile:
        print(i, no_nan_df[i:i + 1]['county_name_x'].iloc[0])
        counter += 1
    if counter > 15:
        break


PC2_scores = matrix.dot(pca.components_[1])
top_10_percentile = np.percentile(PC2_scores, 90)
bottom_10_percentile = np.percentile(PC2_scores, 10)
print("*"*80)
print("PC2 highest scores")
counter = 0
for i, score in enumerate(PC2_scores):
    if score > top_10_percentile:
        print(i, no_nan_df[i:i + 1]['county_name_x'].iloc[0])
        counter += 1
    if counter > 15:
        break
print("*"*80)
print("PC2 lowest scores")
counter = 0
for i, score in enumerate(PC2_scores):
    if score < bottom_10_percentile:
        print(i, no_nan_df[i:i + 1]['county_name_x'].iloc[0])
        counter += 1
    if counter > 15:
        break





#
