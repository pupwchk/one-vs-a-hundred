'''event_data에 정답이 있음.'''


import pandas as pd
from E_data_form_making import convert_df_to_agent_format

sheet = pd.read_csv('../data/answer/prediction_results.csv')
origin_data = pd.read_csv('../data/articles/csv/news_with_market_cap_20250929_180045.csv') # 이런 상대경로들도 os.listdir로 바꾸긴 해야할듯?


print(origin_data)
print(origin_data.columns)