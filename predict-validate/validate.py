import pandas as pd


def rule_mean(df: pd.DataFrame) -> pd.DataFrame:
    null_cols = list(df.columns[df.isnull().any()])
    if null_cols:
        print('WARNING!!! 以下指标存在空值： ', null_cols)
    df = df.groupby('quarter').mean()
    return df
    # df = df.mean(axis=0)
    # return pd.DataFrame(df, columns=['wind_converted'])


def rule_last(df: pd.DataFrame) -> pd.DataFrame:
    df = df.groupby('quarter').tail(1)

    tmp = df.drop('quarter', axis=1)
    print(tmp)
    print(tmp.index.is_quarter_end)
    not_quarter_end = list(tmp.columns[~tmp.index.is_quarter_end | tmp.isnull().any()])
    if not_quarter_end:
        print('WARNING!!! 以下指标季末存在空值： ', not_quarter_end)
    return df.set_index('quarter')


RULES_P1 = {
    rule_mean: [
        'X1', 'X2',
    ],
    rule_last: [
        'X2', 'X3', 'X4',
    ],
}


def rule_add(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(df.sum(axis=1), columns=['_'.join(df.columns)])
    # return df.sum(axis=1)


# 需要基于RULES_P1的结果数据计算
RULES_P2 = {
    rule_add: [
        ['X1', 'X2'],
    ],
}


def rule_yoy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reindex(pd.period_range(df.index.min(), df.index.max(), freq='Q').strftime('%YQ%q'))
    null_cols = list(df.columns[df.isnull().any()])
    if null_cols:
        print('WARNING!!! 以下指标存在季度空值： ', null_cols)
    return df.pct_change(4).add_suffix('_yoy')


# 需要基于RULES_P2的结果数据计算
RULES_P3 = {
    rule_yoy: [
        'X1',
    ],
}


def read_wind_data() -> pd.DataFrame:
    df = pd.read_csv('./wind_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.pivot(index='date', values='value',
                  columns='id').sort_index()
    df['quarter'] = pd.PeriodIndex(df.index, freq='Q').strftime('%YQ%q')
    return df


def convert_data(df: pd.DataFrame) -> pd.DataFrame:
    # Phase 1
    df_p1 = pd.concat([method(df[ids + ['quarter']])
                       for method, ids in RULES_P1.items()],
                      axis=1)

    # Phase 2
    # df_p2 = [pd.DataFrame(method(df_p1[ids]), columns=['_'.join(ids)])
    df_p2 = [method(df_p1[ids])
             for method, ids_lst in RULES_P2.items()
             for ids in ids_lst]
    df_p2 = pd.concat([df_p1] + df_p2, axis=1)

    # Phase 3
    df_p3 = [method(df_p2[ids])
             for method, ids in RULES_P3.items()]
    df_p3 = pd.concat([df_p2] + df_p3, axis=1, sort=True)

    return df_p3


if __name__ == '__main__':
    df = read_wind_data()
    print(df)
    df = convert_data(df)
    print(df)
