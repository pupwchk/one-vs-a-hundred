import pandas as pd
df = pd.read_csv('../data/articles/csv/news_with_article_body_retry.csv')
print(df['article_body'].isna().sum())
