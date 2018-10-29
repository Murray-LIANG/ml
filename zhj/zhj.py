import os
from datetime import timedelta

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


def split_by_stock(df):
    codes = df['CODE'].unique()
    return {code: df[df['CODE'] == code] for code in codes}


def process_stock(df, stock_code, end_date):
    print('PROCESSING STOCK: {}, END DATE: {}'.format(stock_code, end_date))
    end_date = pd.to_datetime(end_date)
    start_date = end_date - timedelta(days=30)

    df.sort_values(by=['ORGAN_ID', 'CREATE_DATE'], inplace=True)
    print(df)

    result = {}
    for organ_id in df['ORGAN_ID'].unique():
        rows_w_end_date = df[(df['ORGAN_ID'] == organ_id)
                             & (df['CREATE_DATE'] <= end_date)]

        rows_w_start_date = rows_w_end_date[
            rows_w_end_date['CREATE_DATE'] >= start_date]
        if rows_w_start_date.empty:
            print('!!! NO VALID DATA IN THIS DATE RANGE !!!')
            continue

        print(rows_w_start_date)

        indexes = list(rows_w_end_date.index.values)
        start_index = rows_w_start_date.index.values[0]
        i = (0 if indexes.index(start_index) == 0
             else indexes.index(start_index) - 1)

        signs = []
        for j in range(i + 1, len(indexes)):
            name = 'FORECAST_PROFIT'
            sign = '='
            if df.loc[indexes[j]][name] > df.loc[indexes[j - 1]][name]:
                sign = '+'
            elif df.loc[indexes[j]][name] < df.loc[indexes[j - 1]][name]:
                sign = '-'
            signs.append(sign)
        result[(stock_code, organ_id)] = signs

    return result


def main():
    df = read_data()
    convert_date(df, 'CREATE_DATE')
    df_dict = split_by_stock(df)
    print(len(df_dict))
    for code, df in df_dict.items():
        if code == 600318:
            print(process_stock(df, code, '9/22/2016'))


if __name__ == '__main__':
    main()
