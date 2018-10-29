import os

import pandas as pd

pd.set_option('display.width', 0)
pd.set_option('display.max_columns', 0)

CSV_PATH = '/home/liangr/git/ml/zhj/results_a.csv'
FEATHER_PATH = CSV_PATH + '.feather'


def read_data():
    if os.path.exists(FEATHER_PATH):
        df = pd.read_feather(FEATHER_PATH)
    else:
        df = pd.read_csv(CSV_PATH)
        df.to_feather(FEATHER_PATH)
    return df


def main():
    df = read_data()
    print(df['predindex'].corr(df['hs300']))


if __name__ == '__main__':
    main()
