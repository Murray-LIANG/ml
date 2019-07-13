import numpy as np
import pandas as pd

pd.set_option('display.width', 0)
pd.set_option('display.max_columns', 0)

CSV_PATH = '/home/murray/git/ml/zhj/stock_pepb.csv'

df = pd.read_csv(CSV_PATH)

date = '2019-01-11'
df_pe = df[df.date == date][['stk', 'PE']]

dict_new_pe = df_pe.set_index('stk').to_dict('dict')['PE']


def new_pe(row):
    return dict_new_pe.get(row.stk, 0)


df['PE_new'] = df.apply(new_pe, axis=1)

df['is_less'] = np.where(df.PE < df.PE_new, 1, 0)

df_percent = df[['stk', 'is_less']].groupby('stk').aggregate(['sum', 'count'])

df_percent.columns = ['count_smaller_than_new_pe', 'count_total']

df_percent['percent'] = df_percent['count_smaller_than_new_pe'] / df_percent['count_total']

print(df_percent)
