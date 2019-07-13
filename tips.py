#
# columns_to_shift = X.columns[~(X.columns.isin(['date']))]
# X_lag1 = pd.concat([X['date'], X[columns_to_shift].shift(1)], axis=1)
# X_lag1.set_index('date', inplace=True)

# test = X_lag1['2011-3':]
# test = test.drop(test.columns[~test.isnull().any()], axis=1)
# test

# import numpy as np

# X_lag1_Y1[~X_lag1_Y1.applymap(np.isreal).all(1)]

# test = pd.DataFrame(X_lag1_Y1.applymap(lambda x: isinstance(x, (int, float))).all(0))
# test[~test[0]]
