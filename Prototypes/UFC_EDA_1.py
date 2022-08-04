#loading in an validating dataset

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('/users/n/UFC-Predictions/data/data.csv')

df.info()

#due to the large amount of data that we have we must analyze based on types
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
categorical = ['object']
bool = ['bool']

numeric_vals_df = df.select_dtypes(include=numerics)
cat_vals_df = df.select_dtypes(include=categorical)
boolean_vals = df.select_dtypes(include=bool)

#first we will analyze the categorical data

cat_vals_df.info()

ax = cat_vals_df['weight_class'].value_counts().sort_values().plot.barh()
ax.set_xlabel("Count of Weight Class",fontsize=10)
ax.set_title("Weight Class Frequency", fontsize=10)

#top 10 most used locations of fights
ax = cat_vals_df['location'].value_counts(ascending=False).sort_values(ascending=False).head(15).plot.barh()
ax.set_xlabel("Frequency of Fight Locations",fontsize=10)
ax.set_title("Location Frequency", fontsize=10)

#extracting date data
df['DateTime'] = pd.to_datetime(df['date'])
df['month'] = df['DateTime'].dt.month
df['year'] = df['DateTime'].dt.year

#showing which years occur frequently
ax = df['year'].value_counts(ascending=False).sort_values(ascending=False).head(20).plot.barh()
ax.set_xlabel("Year Frequency",fontsize=10)
ax.set_title("Year Frequency Counts", fontsize=10)

#shows the number of fights in the UFC from 1993 - 2021
#as you can see fights have increased substantially since the start of the ufc in 1993, with huge amounts of growth from 2005 - 2015
filter = df['year'] < 2022
df[['Winner', 'year']].where(filter).groupby('year').count().plot()

ax = df['Referee'].value_counts(ascending=False).sort_values(ascending=False).head(20).plot.barh()
ax.set_xlabel("Refree Frequency",fontsize=10)
ax.set_title("Count of Number of UFC Fights an official refereed", fontsize=10)

#stances for red and blue fighters
df['B_Stance'].value_counts()
df['R_Stance'].value_counts()

#visualizing fighter stances
#blue stances
ax = df['B_Stance'].value_counts(ascending=False).sort_values(ascending=False).head(20).plot.barh()
ax.set_xlabel("Blue Stance Frequency",fontsize=10)
ax.set_title("Number of times Stance Occured", fontsize=10)

#Red stances
ax = df['R_Stance'].value_counts(ascending=False).sort_values(ascending=False).head(20).plot.barh()
ax.set_xlabel("Red Stance Frequency",fontsize=10)
ax.set_title("Number of times Stance Occured", fontsize=10)

#showing how fighter stances have changed over time
df[['year', 'R_Stance']].groupby('year').count()

#How the red stance has changed from 1993 - 2021
#switch hitters and southpaws are becoming more prevalent showing evolution of the sport
test = df[['year', 'R_Stance']]
test = pd.get_dummies(test)
test_less_2022 = test < 2022
test_filtered = test.where(test_less_2022)
group_by_red_corne_stance = test_filtered.groupby('year').sum()
ax = group_by_red_corne_stance.plot()
ax.set_title('Red Stance Changes from 1993-2021')

#how blue stance has changed from 1993 - 2021
#switch hitters and southpaws are becoming more prevalent showing evolution of the sport
test = df[['year', 'B_Stance']]
test = pd.get_dummies(test)
test_less_2022 = test < 2022
test_filtered = test.where(test_less_2022)
group_by_red_corne_stance = test_filtered.groupby('year').sum()
ax = group_by_red_corne_stance.plot()
ax.set_title('Blue Stance Changes from 1993-2021')

#analyzing numerical data
#using PCA to lower dimensionality of dataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

scaler = StandardScaler()
pca = PCA(n_components=2)

num_vals_dropped = numeric_vals_df.dropna()

scaler.fit(numeric_vals_df)
scaler.transform(numeric_vals_df)

#pca with pca library
from pca import pca
# Initialize
model = pca()
# Fit transform
out = model.fit_transform(numeric_vals_df)

print(model.results['topfeat'].head(20))

#finding the important variables
#top 5 most import vars are for the first 7 loadings as follows:
    #1. R_total_rounds_fought
    #2. B_total_rounds_fought
    #3. R_avg_opp_CTRL_time(seconds)
    #4. R_total_time_fought(seconds)
    #5. B_avg_opp_CTRL_time(seconds)
    #6. B_total_time_fought(seconds)

#these features are exceptionally well at explaining the variance within our dataset and are the highest loadings in the top 6 Principal Components
numeric_vals_df.iloc[:, 115]

numeric_vals_df.iloc[:, 49]

numeric_vals_df.iloc[:, 113]

numeric_vals_df.iloc[:, 114]

numeric_vals_df.iloc[:, 47]

numeric_vals_df.iloc[:, 48]


#pca seems to be able to detect the differences between blue and red fighters quite easily
#fight time and control time seem to be the strongest loadings
model.biplot(cmap=None, label=None, legend=False)

#trying to use PCA with imputation methods
from sklearn.impute import SimpleImputer
from pca import pca

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(numeric_vals_df)
numeric_vals_df = imp.transform(numeric_vals_df)

model = pca()
# Fit transform
out = model.fit_transform(numeric_vals_df)

print(out['topfeat'])
model.biplot(cmap=None, label=None, legend=False)

model.plot()


#Analyzing based on fighter stats we have the blue and the red figher, each row of data shows a different fight
df_blue_stats = df.iloc[:, 8:77]
df_red_stats = df.iloc[:, 75: 144]



#plotting first half of the blue fighter dataset, no drop na
df_blue_stats.iloc[:, 0:35].hist(figsize=(40,40))
#plotting second half of the blue fighter dataset, no drop na
df_blue_stats.iloc[:, 35:69].hist(figsize=(40,40))


df_red_stats.iloc[:, 0:35].hist(figsize=(40,40))

df_red_stats.iloc[:, 35:69].hist(figsize=(40,40))



#Analyzing based on fighter stats we have the blue and the red figher with dropped null values and standardization
df_blue_stats_dropped = df_blue_stats.dropna()
df_red_stats_dropped = df_red_stats.dropna()

#standarding our values
df_blue_stats_std = (df_blue_stats_dropped - df_blue_stats_dropped.mean()) / (df_blue_stats_dropped.std())
df_red_stats_std = (df_red_stats_dropped - df_red_stats_dropped.mean()) / (df_red_stats_dropped.std())

#plotting first half of the blue fighter dataset, no drop na
df_red_stats_std.iloc[:, 0:35].hist(figsize=(40,40))

#plotting second half of the blue fighter dataset, no drop na
df_blue_stats_std.iloc[:, 35:69].hist(figsize=(40,40))

#breifly analyzing class balance
#majority of fights are won by the red group, with blue and draw following
df['Winner'].value_counts(normalize=True).plot.barh()

# 60% of winners are in the red corner, 32% in blue corner and the rest are draws
df['Winner'].value_counts(normalize = True)

#finally before finishing this EDA we will explore missing values, will be used to decide imputation method

import pylab
def plot_missing_values(df):
    """ For each column with missing values plot proportion that is missing."""
    data = [(col, df[col].isnull().sum() / len(df))
            for col in df.columns if df[col].isnull().sum() > 0]
    col_names = ['column', 'percent_missing']
    missing_df = pd.DataFrame(data, columns=col_names).sort_values('percent_missing')
    pylab.rcParams['figure.figsize'] = (15, 8)
    missing_df.plot(kind='barh', x='column', y='percent_missing');
    plt.title('Percent of missing values in colummns');

plot_missing_values(df_blue_stats)

plot_missing_values(df_red_stats)
