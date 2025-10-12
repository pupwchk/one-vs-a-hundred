'''크롤링이 다 이루어지지 않은 파일들, 링크들을 출력해야 함.'''

import pandas as pd
df = pd.read_csv('../data/articles/csv/news_with_article_body_retry.csv')
print(df['article_body'].isna().sum())

print(df[df['article_body'].isna()][['publisher_title', 'actual_url']])