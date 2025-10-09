import pandas as pd
from newspaper import Article
from pandas.io.formats.style import Subset
from tqdm import tqdm


# CSV 파일 읽기
df_articles = pd.read_csv('../data/articles/csv/news_with_market_cap_with_actual_url.csv')

def get_body(url):
    try:
        article = Article(url, language='en')
        article.download()
        article.parse()
        return article.text
    except:
        return None

# article_body 열 추가
df_articles['article_body'] = None

for idx in tqdm(range(len(df_articles)), desc="기사 수집 중"):
    url = df_articles.loc[idx, 'actual_url']
    body = get_body(url)
    df_articles.loc[idx, 'article_body'] = body


# 저장
df_articles.to_csv("../data/articles/csv/news_with_article_body.csv", index=False)