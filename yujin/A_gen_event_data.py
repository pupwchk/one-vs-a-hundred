'''이벤트 데이터를 만들어보자'''
import pandas as pd
import yfinance as yf
import requests
import numpy as np

import warnings
warnings.filterwarnings(action='ignore')

# S&P100 전체 종목 이름, 심볼, 섹터 가져오기
res = requests.get(url = 'https://en.wikipedia.org/wiki/S%26P_100',
                    headers={'user-agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) \
                    AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36'})

df_symbol = pd.read_html(res.text)[2]
df_symbol.to_csv('../data/stock_data.csv', index=False) # 저장까지

df_stock = pd.read_csv('../data/df_cond1_tidy.csv')
cond = df_stock['Symbol'].isin(df_symbol.Symbol.tolist())
df_cond1 = df_stock[cond]

df_merged = df_cond1.merge(df_symbol, how='left', on='Symbol')
df_event = df_merged[['Date', 'Symbol', 'Name', 'Sector', 'price_change']]

df_event['up-down'] = df_event['price_change'].apply(lambda x: np.sign(x))

df_event.to_csv('../data/event_data.csv', index=False)