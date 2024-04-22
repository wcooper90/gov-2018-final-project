import pandas as pd



df = pd.read_csv('data/uscounties.csv', encoding = "ISO-8859-1")
characteristics_df = pd.read_csv('data/county_characteristics.csv', encoding = "ISO-8859-1")


print(len(set(df['county_fips'].tolist()) | set(characteristics_df['cty'].tolist())))
print(len(characteristics_df['cty'].tolist()))
print(len(df['county_fips'].tolist()))
