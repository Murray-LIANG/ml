import os

import pandas as pd

pd.set_option('display.width', 0)
pd.set_option('display.max_columns', 0)

CSV_PATH = '/home/liangr/git/ml/zhj/data_2016.csv'
FEATHER_PATH = CSV_PATH + '.feather'


def read_data():
    if os.path.exists(FEATHER_PATH):
        df = pd.read_feather(FEATHER_PATH)
    else:
        df = pd.read_csv(CSV_PATH)
        df.to_feather(FEATHER_PATH)
    return df


def convert_date(df, date_column_name):
    df[date_column_name + '_STR'] = df[date_column_name]
    df[date_column_name] = pd.to_datetime(df[date_column_name])


def add_forecast(df):
    df.sort_values(by=['CODE', 'ORGAN_ID', 'CREATE_DATE'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df['last_forecast'] = df['FORECAST_PROFIT'].shift(1)
    df['increase'] = ((df['CODE'].shift(1) == df['CODE'].shift(0))
                      & (df['ORGAN_ID'].shift(1) == df['ORGAN_ID'].shift(0))
                      & (df['FORECAST_PROFIT'].shift(1)
                         < df['FORECAST_PROFIT'].shift(0)))

    df['same'] = ((df['CODE'].shift(1) == df['CODE'].shift(0))
                  & (df['ORGAN_ID'].shift(1) == df['ORGAN_ID'].shift(0))
                  & (df['FORECAST_PROFIT'].shift(1)
                     == df['FORECAST_PROFIT'].shift(0)))

    df['decrease'] = ((df['CODE'].shift(1) == df['CODE'].shift(0))
                      & (df['ORGAN_ID'].shift(1) == df['ORGAN_ID'].shift(0))
                      & (df['FORECAST_PROFIT'].shift(1)
                         > df['FORECAST_PROFIT'].shift(0)))


def main():
    df = read_data()
    convert_date(df, 'CREATE_DATE')
    print(df)
    add_forecast(df)
    print(df)
    print(df[df['CODE'] == 600318])


if __name__ == '__main__':
    main()
