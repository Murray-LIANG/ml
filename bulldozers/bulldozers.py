import math
import os

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

import utils

pd.set_option('display.width', 0)
pd.set_option('display.max_columns', 0)

DATA_DIR = '/home/liangr/git/ml/bulldozers/data'
TRAIN_CSV = os.path.join(DATA_DIR, 'Train.csv')
TRAIN_FEATHER = os.path.join(DATA_DIR, 'tmp', 'bulldozers.train.tmp')

''' Uncomment below two lines to reload the original data from csv.'''
# df_raw = pd.read_csv(TRAIN_CSV, low_memory=False, parse_dates=['saledate'])
# df_raw.to_feather(TRAIN_FEATHER)

df_raw = pd.read_feather(TRAIN_FEATHER)

origin_saleprice = df_raw.SalePrice.copy()
print('Skewness: {:f}'.format(origin_saleprice.skew()))
print('Kurtosis: {:f}'.format(origin_saleprice.kurt()))

df_raw.SalePrice = np.log(df_raw.SalePrice)
sns.distplot(df_raw.SalePrice)
print('Skewness: {:f}'.format(df_raw.SalePrice.skew()))
print('Kurtosis: {:f}'.format(df_raw.SalePrice.kurt()))

print(df_raw.head(1))
''' There are some string type columns, which are not suitable for decision
    tree.'''
utils.str_to_category(df_raw)
df_raw.UsageBand.cat.set_categories(['High', 'Medium', 'Low'],
                                    ordered=True, inplace=True)

''' Parse the datetime into several columns, like date, dayofweek, .etc'''
utils.parse_date(df_raw, 'saledate', drop_origin=True)

''' Lots of missing data.'''
print(df_raw.isnull().sum().sort_index() / len(df_raw))

''' Fill the missing value with mean, and use codes to represent categories.'''
df, y = utils.process(df_raw, 'SalePrice')

print(df.head(1))

''' Use all data to train will lead to overfitting.'''
# m = RandomForestRegressor(n_jobs=-1)
# m.fit(df, y)
# # `m.score` will return r² value (1 is good, 0 is bad)
# print(m.score(df, y))

''' Split the data to train set and validate set.'''
validate_size = 12000  # kaggle test set size
train_size = len(df) - validate_size

X_train, X_valid = utils.split(df, train_size)
y_train, y_valid = utils.split(y, train_size)
print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)

''' Take sample to train will save a lot of time.'''
df, y = utils.process(df_raw, 'SalePrice', sample_size=30000)

''' Don't change the validate set.'''
X_train, _ = utils.split(df, 20000)
y_train, _ = utils.split(y, 20000)
print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)

m = RandomForestRegressor(n_estimators=1, max_depth=3,
                          bootstrap=False, n_jobs=-1)
m.fit(X_train, y_train)
print(utils.scores(m, X_train, y_train, X_valid, y_valid))
