import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# pd.set_option('display.height', 10000)
pd.set_option('display.max_rows', 10000)
pd.set_option('display.width', 10000)
pd.set_option('display.max_columns', 10000)


def clean_data(df, y_id):

    # 将房价2010的数据用S2707403的数据补齐
    col_new = 'S2707411'
    df.loc['2010', col_new] = [12.83, 14.87, 11.97, 9.17]

    # 保留空值少于70%的列
    to_drop = df.columns[df.isnull().mean() > .7]
    print('以下列将被去除，因为它们的空值超过70%: {}'.format(to_drop))
    df = df.drop(to_drop, axis=1)

    df_y = df.loc[:, [y_id]]
    df.drop(y_id, axis=1, inplace=True)
    # 去除第一行为空值的列，因为X已经是根据起始/结束日期选出的数据
    to_drop = df.columns[df.iloc[0].isnull()]
    print('以下列将被去除，因为它们的第一行为空值: {}'.format(to_drop))
    print('第一行数据: {}'.format(df.head(1)))
    df = df.drop(to_drop, axis=1)

    df[y_id] = df_y
    print('清理后的数据前3行: {}'.format(df.head(3)))
    return df


def shift_down(df, y_id, lag_num):
    # 在数据后增加lag_num行
    for _ in range(lag_num):
        empty_row = pd.DataFrame([],
                                 index=pd.to_datetime(df[-1:].index) + pd.offsets.QuarterEnd(),
                                 columns=df.columns)
        df = pd.concat([df, empty_row])

    # 保留未移动过的Y
    df_y = df.loc[:, [y_id]]

    # 将X下移lag_num
    df_x = df.drop(y_id, axis=1)
    df_x = df_x.shift(lag_num)

    print('向下移动{}行的X（头3行）: {}'.format(lag_num, df_x.head(3)))
    print('未移动的Y（头3行）: {}'.format(df_y.head(3)))
    return df_x, df_y


def corr(df_x, df_y, y_id):
    # 只保留行中没有空值的行
    df_x_y = pd.concat([df_x, df_y], axis=1)
    df_x_y = df_x_y.loc[~df_x_y.isnull().sum(axis=1).astype(bool)]

    df_y_calc = df_x_y[y_id]
    df_x_calc = df_x_y.drop(y_id, axis=1)
    print('计算相关性的X: {}'.format(df_x_calc.head(3)))

    corr = pd.DataFrame(df_x_calc.corrwith(df_y_calc))
    print('相关性系数: {}'.format(corr))
    return corr


def pick_feature_to_lag(df_x_dict, df_y, y_id, corr_filter):
    # df_x_dict = {'lag1': df_x_lag1,
    #              'lag2': df_x_lag2,
    #              'lag3': df_x_lag3,
    #              'lag4': df_x_lag4}
    def helper(lag_name, df_x):
        c = corr(df_x, df_y, y_id)
        c.rename(columns={0: lag_name}, inplace=True)
        return c
    # 计算所有的x（lag1, lag2, lag3, lag4）与y的相关性
    corr_all = pd.concat([helper(lag_name, df_x)
                          for lag_name, df_x in df_x_dict.items()], axis=1)

    # 去掉相关性绝对值小于corr_filter（比如：0.6）的指标
    corr_all[corr_all.abs() < corr_filter] = np.nan
    # 计算出相关性最大的那个lag
    corr_all['lag'] = corr_all.abs().idxmax(axis=1)

    res = {}
    for lag_name, df_x in df_x_dict.items():
        # 只保留相关性最大的那个lag中的指标
        df_x = df_x.loc[:, corr_all.loc[corr_all['lag'] == lag_name].index]
        # 将相关性为负数的指标乘以-1
        df_x.loc[:, corr_all[lag_name] < 0] *= -1
        res[lag_name] = df_x

    return res


def pca(df_x, df_y, y_id, lag_num):
    # 只保留行中没有空值的行
    df_x_y = pd.concat([df_x, df_y], axis=1)
    df_x_y = df_x_y.loc[~df_x_y.isnull().sum(axis=1).astype(bool)]

    df_y_calc = df_x_y[y_id]
    df_x_calc = df_x_y.drop(y_id, axis=1)

    pca = PCA(n_components=1)

    pca_lag = None
    while True:
        df_x_std = StandardScaler().fit_transform(df_x_calc)
        pca.fit_transform(df_x_std)
        pca_lag = pd.DataFrame(pca.components_, columns=df_x_calc.columns)

        pca_lag_t = pca_lag.T
        cols_to_drop = [e for e in pca_lag_t[pca_lag_t[0] <= 0].index]
        if not cols_to_drop:
            break
        print('以下列将被删除，因为他们的pca系数小于零: {}'.format(cols_to_drop))
        df_x_calc.drop(cols_to_drop, axis=1, inplace=True)
        df_x.drop(cols_to_drop, axis=1, inplace=True)

    print('Y: {}, lag: {}, pca贡献度: {}'.format(y_id, lag_num, pca.explained_variance_ratio_))
    print('pca系数: {}'.format(pca_lag))

    pca_lag_norm = pca_lag / abs(pca_lag.sum()).sum()
    print('归一化后的pca系数: {}'.format(pca_lag_norm))

    df_x_std = pd.DataFrame(StandardScaler().fit_transform(df_x))
    df_x_std.columns = df_x.columns
    df_x_std.index = df_x.index

    df_y_lag = df_x_std.dot(pca_lag_norm.T)
    df_y_lag.index = df_x_std.index

    result = df_y.copy()
    result.rename(columns={y_id: 'Y_' + y_id}, inplace=True)
    result['Y_' + y_id + '_std'] = StandardScaler().fit_transform(pd.DataFrame(df_y))
    y_name = 'Y_lag{}_{}'.format(lag_num, y_id)
    result[y_name] = df_y_lag

    print('Y: {}, lag: {}, result: \n{}'.format(y_id, lag_num, result.head(3)))
    return result


def corr_pca(df, start_month, end_month, y_id, corr_filter):
    df = df[start_month:end_month]
    df = clean_data(df, y_id)

    df_x_lags = {'lag{}'.format(lag_num): shift_down(df, y_id, lag_num)[0] for lag_num in range(1, 5)}
    _, df_y = shift_down(df, y_id, 4)
    df_x_lags = pick_feature_to_lag(df_x_lags, df_y, y_id, corr_filter)

    result = pd.concat([pca(df_x_lags['lag{}'.format(lag_num)], df_y, y_id, lag_num) for lag_num in range(1, 5)],
                       axis=1)
    return result.loc[:, ~result.columns.duplicated()]


def calc_reg_lag(df_y, y_id, lag_list):
    # 如果lag_list=[3,4]，则计算Y，Y_lag3，Y_lag4的线性回归
    lag_drop = [e for e in range(1, 5) if e not in lag_list]
    columns = [col for col in df_y.columns if not any(col.startswith('Y_lag' + str(lag)) for lag in lag_drop)]
    # 比如：如果lag_list=[3,4]，去掉Y_lag1和Y_lag2的列
    df_y_calc = df_y.loc[:, columns]
    print('去掉不在lag_list中的列后的Y: {}'.format(df_y_calc.head(3)))

    # 去掉那些有空值的行
    df_y_calc = df_y.loc[~df_y.isnull().sum(axis=1).astype(bool)]
    # 比如：把Y_lag3和Y_lag4列取出来
    columns = [col for col in df_y.columns if any(col.startswith('Y_lag' + str(lag)) for lag in lag_list)]
    df_y_lags = df_y_calc.loc[:, columns]

    y_full_id = 'Y_' + y_id

    linear_regressor = LinearRegression()
    print('用于线性回归的X: {}'.format(df_y_lags.head(3)))
    print('用于线性回归的Y: {}'.format(df_y_calc.loc[:, y_full_id].head(3)))
    linear_regressor.fit(df_y_lags, df_y_calc.loc[:, y_full_id])

    print('Y: {}, lag_list: {}, coef: {}'.format(y_id, lag_list, linear_regressor.coef_))
    print('Y: {}, lag_list: {}, intercept: {}'.format(y_id, lag_list, linear_regressor.intercept_))

    # 比如：保留Y_lag3和Y_lag4两列
    # columns = [col for col in df_y.columns if any(col.startswith('Y_lag' + str(lag)) for lag in lag_list)]
    df_y_keep = df_y.loc[:, columns]
    df_y_keep = df_y_keep.loc[~df_y_keep.isnull().sum(axis=1).astype(bool)]
    print('用来预测的Y: {}'.format(df_y_keep))
    df_y_reg = df_y.copy()
    y_reg_name = 'Y_{}_reg_{}'.format(y_id, ''.join(map(str, lag_list)))
    df_y_reg[y_reg_name] = pd.DataFrame(linear_regressor.predict(df_y_keep), index=df_y_keep.index)
    print('线性回归之后的Y: {}'.format(df_y_reg))

    # 计算mse和r^2
    df_validate = df_y_reg.loc[~df_y_reg.isnull().sum(axis=1).astype(bool)]
    print('Y: {}, lag_list: {}, mse: {}'.format(
        y_id, lag_list, mean_squared_error(df_validate[y_full_id], df_validate[y_reg_name])))
    print('Y: {}, lag_list: {}, r^2: {}'.format(
        y_id, lag_list, r2_score(df_validate[y_full_id], df_validate[y_reg_name])))

    return df_y_reg


def calc_reg(df_y, y_id):
    result = pd.concat([calc_reg_lag(df_y, y_id, lag_list)
                        for lag_list in ([1, 2, 3, 4], [2, 3, 4], [3, 4], [4])], axis=1)
    result = result.loc[:, ~result.columns.duplicated()]
    print(result.head(3))
    col_name = 'Y_{}_merged'.format(y_id)
    result.loc[:, col_name] = np.nan

    for col in ['Y_{}_reg_{}'.format(y_id, lag) for lag in ('4', '34', '234', '1234')]:
        result.loc[~result[col].isnull(), col_name] = result[col]

    return result


def draw_plot(df, y_id):
    x = df.index

    f1 = plt.figure(1)
    # Plot y1 vs x in blue on the left vertical axis.
    # plt.xlabel("date")
    # plt.ylabel("", color="b")
    # plt.tick_params(axis="y", labelcolor="b")
    # plt.plot(x, y, "b-", linewidth=2)
    y_std = df['Y_{}_std'.format(y_id)]
    plt.plot(x, y_std, '-', linewidth=2, label='y_std')

    y_id_fmt = 'Y_lag{}_{}'
    y1 = df[y_id_fmt.format(1, y_id)]
    y2 = df[y_id_fmt.format(2, y_id)]
    y3 = df[y_id_fmt.format(3, y_id)]
    y4 = df[y_id_fmt.format(4, y_id)]
    plt.plot(x, y1, '-', linewidth=2, label='y1')
    plt.plot(x, y2, '-', linewidth=2, label='y2')
    plt.plot(x, y3, '-', linewidth=2, label='y3')
    plt.plot(x, y4, '-', linewidth=2, label='y4')
    plt.legend(loc='upper right')
    ax = plt.gca()
    ax.set_xticks(x)
    plt.tick_params(axis='x', labelrotation=90)


    f2 = plt.figure(2)
    y = df['Y_' + y_id]
    plt.plot(x, y, '-', linewidth=2, label='y')

    y_merged = df['Y_{}_merged'.format(y_id)]
    plt.plot(x, y_merged, '-', linewidth=2, label='y_merged')
    plt.legend(loc='upper right')
    ax = plt.gca()
    ax.set_xticks(x)
    plt.tick_params(axis='x', labelrotation=90)

    plt.show()


DATA_PATH = '/home/murray/git/ipynb/alldata.xlsx'

X_orig = pd.read_excel(DATA_PATH, index_col=[0])
X_orig.drop('指标ID', axis=1, inplace=True)
X_orig.drop('S0033812', axis=1, inplace=True)  # S0033812有多个空值

Y_IDS = [
    # 'M5567876',
    # 'M0001548_M5567876',
    # # 'M1000216',
    # 'S2707411',
    'M0010049',
]

for y_id in Y_IDS:
    X = X_orig.copy()
    if '_' in y_id:
        y1_id, y2_id = y_id.split('_')
        X[y_id] = X[y1_id] / X[y2_id]
        X.drop(y1_id, axis=1, inplace=True)
        X.drop(y2_id, axis=1, inplace=True)
    df_y = corr_pca(X, '2010-03', '2018-12', y_id, 0.8)
    print('Y: {}, lag1234合并结果: {}'.format(y_id, df_y.head(3)))

    df_y_reg = calc_reg(df_y, y_id)
    df_y = pd.concat([df_y, df_y_reg], axis=1)
    df_y = df_y.loc[:, ~df_y.columns.duplicated()]
    print('Y: {}, 线性回归后的Y: {}'.format(y_id, df_y))
    y_name = 'Y_' + y_id
    df_y.loc[:, ~df_y.columns.duplicated()].to_csv('./{}.csv'.format(y_name))
    draw_plot(df_y, y_id)
