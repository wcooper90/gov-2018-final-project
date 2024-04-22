import pandas as pd
import scipy
import numpy as np
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

bruh = processed_df['mort_30day_hosp_z'].tolist()

print(bruh)

#
