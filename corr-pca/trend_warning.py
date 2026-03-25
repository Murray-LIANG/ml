import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import r2_score


def fit_trend(df, y_id, r2_filter):
    df_y = pd.DataFrame(df[y_id])
    df_y = df_y[~df_y[y_id].isnull()]

    _, trend = sm.tsa.filters.hpfilter(df_y[y_id], lamb=1600)
    df_y['trend'] = trend

    # 后面加四行空行
    for _ in range(4):
        empty_row = pd.DataFrame(
            [],
            index=pd.to_datetime(df_y[-1:].index) + pd.offsets.QuarterEnd(),
            columns=df_y.columns)
        df_y = pd.concat([df_y, empty_row])

    df_y['dateseries'] = np.arange(1, df_y.shape[0] + 1, 1)
    df_calc = df_y[~df_y[y_id].isnull()]

    x = df_calc.dateseries
    y = df_calc.trend

    i = 5
    while True:
        coefs = np.polyfit(x, y, i)  # 用i次多项式拟合
        yvals = np.polyval(coefs, x)

        print(yvals)

        r2 = r2_score(y, yvals)
        print('r^2: {}'.format(r2))
        if r2 >= r2_filter:
            break
        i += 1

    df_y['fit_line'] = np.polyval(coefs, df_y.dateseries)

    df_y['trend_merge'] = df_y['trend']
    df_y.loc[df_y['trend'].isnull(), 'trend_merge'] = df_y['fit_line']
    return df_y


def add_warning_lines(df_y, y_id, coefs, only_lower=False):
    std_value = np.std(df_y[y_id])
    print('std deviation: {}'.format(std_value))
    if not only_lower:
        for i, coef in enumerate(coefs):
            df_y['upper_{}'.format(i + 1)] = df_y['trend_merge'] + coef * std_value
    for i, coef in enumerate(coefs):
        df_y['lower_{}'.format(i + 1)] = df_y['trend_merge'] - coef * std_value
    return df_y


if __name__ == '__main__':
    DATA_PATH = '/home/murray/git/ipynb/alldata.xlsx'

    X_orig = pd.read_excel(DATA_PATH, index_col=[0])
    y_id = 'M5567876'

    df_res = fit_trend(X_orig.copy(), 'M5567876', 0.99)
    df_res = add_warning_lines(df_res, y_id, (0.8, 1.6, 2.4))

    x = df_res.index
    y = df_res[y_id]
    plt.plot(x, y, '-', label='original values')
    plt.plot(x, df_res['trend'], 'b', label='real trend')
    plt.plot(x, df_res['fit_line'], 'g', label='polyfit values')
    plt.plot(x, df_res['trend_merge'], 'y', label='trend with predict')

    plt.plot(x, df_res['lower_1'], 'r', label='lower 1')
    plt.plot(x, df_res['lower_2'], 'r', label='lower 2')
    plt.plot(x, df_res['lower_3'], 'r', label='lower 3')

    plt.plot(x, df_res['upper_1'], 'r', label='upper 1')
    plt.plot(x, df_res['upper_2'], 'r', label='upper 2')
    plt.plot(x, df_res['upper_3'], 'r', label='upper 3')
    plt.show()
