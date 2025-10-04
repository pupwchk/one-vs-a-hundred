"""
1. 이벤트 데이터 불러오기(우리가 조사해야 하는 조건에 대한 검색)
2. 해당하는 일자(등락이 심했던 날의 전날)의 뉴스 제목 불러오기

--> 반환값은 json, csv 형식으로
"""

from itertools import count
import pandas as pd
from gnews import GNews
from datetime import datetime, timedelta
from tqdm import tqdm


# 필요한 파일 불러오기
df_event = pd.read_csv('../data/event_data.csv')
df_stock = pd.read_csv('../data/stock_data.csv')


# Date 컬럼 → datetime 변환
df_event["Date"] = pd.to_datetime(df_event["Date"])

# GNews 객체 생성
google_news = GNews(language='en', country='US')

# 첫 번째 이벤트 날짜 가져오기
first_date = df_event["Date"].iloc[0].date()

# start_date / end_date 속성 설정
google_news.start_date = first_date# - timedelta(days=1)
google_news.end_date = first_date

# 뉴스 검색
results = pd.DataFrame(google_news.get_news(df_event['Symbol'][0]))
print(results)
print(first_date)