"""
실패한 기사 본문 수집 재시도 스크립트

1단계: newspaper 라이브러리로 최대 3번까지 재시도
2단계: 여전히 실패한 경우 Selenium으로 본문 수집
"""

import pandas as pd
import time
from newspaper import Article
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

# ========== 설정 ==========
INPUT_CSV = '../data/articles/csv/news_with_article_body.csv'
OUTPUT_CSV = '../data/articles/csv/news_with_article_body_retry.csv'
MAX_NEWSPAPER_RETRIES = 3
SELENIUM_TIMEOUT = 15

# ========== 데이터 로드 및 실패 데이터 필터링 ==========
print("="*60)
print("실패한 기사 본문 수집 재시도 시작")
print("="*60)

# retry 파일이 이미 존재하면 그 파일을 불러오기
import os
if os.path.exists(OUTPUT_CSV):
    print(f"\n기존 retry 파일 발견: {OUTPUT_CSV}")
    print("기존 retry 파일을 불러와서 재시도를 이어갑니다.\n")
    df_article = pd.read_csv(OUTPUT_CSV)
else:
    print(f"\nretry 파일이 없습니다. 원본 파일을 불러옵니다: {INPUT_CSV}\n")
    df_article = pd.read_csv(INPUT_CSV)

print(f"전체 기사 수: {len(df_article):,}")
print(f"본문 수집 실패 수: {df_article['article_body'].isna().sum():,}")

# 실패한 기사만 필터링
failure_mask = df_article['article_body'].isna()
df_failure = df_article[failure_mask].copy()

print(f"\n실패 통계:")
print(f"  - 고유 언론사 수: {df_failure['publisher_title'].nunique()}")
print(f"  - 고유 URL 수: {df_failure['actual_url'].nunique()}")

if len(df_failure) == 0:
    print("\n모든 기사 본문이 이미 수집되었습니다.")
    exit(0)

# ========== 1단계: newspaper로 재시도 ==========
def get_body_with_newspaper(url, max_retries=MAX_NEWSPAPER_RETRIES):
    """newspaper 라이브러리로 기사 본문 수집 (재시도 포함)"""
    for attempt in range(1, max_retries + 1):
        try:
            article = Article(url, language='en')
            article.download()
            article.parse()

            if article.text and len(article.text.strip()) > 0:
                return article.text, True, None
            else:
                if attempt == max_retries:
                    return None, False, "빈 본문"
        except Exception as e:
            if attempt == max_retries:
                return None, False, f"newspaper 에러: {str(e)[:50]}"
            time.sleep(0.5)

    return None, False, "알 수 없는 에러"

print(f"\n{'='*60}")
print(f"1단계: newspaper로 재시도 (최대 {MAX_NEWSPAPER_RETRIES}회)")
print(f"{'='*60}\n")

newspaper_success = 0
newspaper_fail = 0
failed_indices = []

for idx in tqdm(df_failure.index, desc="newspaper 재시도"):
    url = df_article.loc[idx, 'actual_url']

    if pd.isna(url):
        newspaper_fail += 1
        failed_indices.append(idx)
        continue

    body, success, error_msg = get_body_with_newspaper(url)

    if success:
        df_article.loc[idx, 'article_body'] = body
        newspaper_success += 1
    else:
        newspaper_fail += 1
        failed_indices.append(idx)

print(f"\n1단계 완료:")
print(f"  - 성공: {newspaper_success}개")
print(f"  - 실패: {newspaper_fail}개")

# ========== 2단계: Selenium으로 재시도 ==========
if len(failed_indices) > 0:
    print(f"\n{'='*60}")
    print(f"2단계: Selenium으로 재시도")
    print(f"{'='*60}\n")

    def get_body_with_selenium(url):
        """Selenium으로 기사 본문 수집"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.page_load_strategy = 'eager'

        driver = webdriver.Chrome(options=chrome_options)
        driver.set_page_load_timeout(SELENIUM_TIMEOUT)

        try:
            driver.get(url)
            time.sleep(3)

            # 여러 일반적인 기사 본문 선택자 시도
            selectors = [
                'article',
                '[class*="article-body"]',
                '[class*="article-content"]',
                '[class*="post-content"]',
                '[class*="entry-content"]',
                '[id*="article-body"]',
                '[id*="article-content"]',
                'main',
                '.content',
                '#content'
            ]

            body_text = None
            for selector in selectors:
                try:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        text = '\n\n'.join([elem.text for elem in elements if elem.text])
                        if len(text.strip()) > 100:  # 최소 100자 이상
                            body_text = text
                            break
                except:
                    continue

            # 선택자로 찾지 못한 경우 body 전체 텍스트 가져오기
            if not body_text:
                try:
                    body = driver.find_element(By.TAG_NAME, 'body')
                    body_text = body.text
                except:
                    pass

            if body_text and len(body_text.strip()) > 50:
                return body_text, True, None
            else:
                return None, False, "본문 추출 실패"

        except TimeoutException:
            return None, False, "타임아웃"
        except WebDriverException as e:
            return None, False, f"WebDriver 에러: {str(e)[:50]}"
        except Exception as e:
            return None, False, f"예상치 못한 에러: {str(e)[:50]}"
        finally:
            driver.quit()

    selenium_success = 0
    selenium_fail = 0
    final_failed_indices = []

    for idx in tqdm(failed_indices, desc="Selenium 재시도"):
        url = df_article.loc[idx, 'actual_url']

        if pd.isna(url):
            selenium_fail += 1
            final_failed_indices.append(idx)
            continue

        body, success, error_msg = get_body_with_selenium(url)

        if success:
            df_article.loc[idx, 'article_body'] = body
            selenium_success += 1
        else:
            selenium_fail += 1
            final_failed_indices.append(idx)

        time.sleep(0.3)  # 서버 부하 방지

    print(f"\n2단계 완료:")
    print(f"  - 성공: {selenium_success}개")
    print(f"  - 실패: {selenium_fail}개")
else:
    selenium_success = 0
    selenium_fail = 0
    final_failed_indices = []

# ========== 결과 저장 ==========
print(f"\n{'='*60}")
print(f"결과 저장 중...")
print(f"{'='*60}\n")

df_article.to_csv(OUTPUT_CSV, index=False)
print(f"업데이트된 데이터 저장: {OUTPUT_CSV}")

# ========== 최종 통계 ==========
print(f"\n{'='*60}")
print(f"최종 통계")
print(f"{'='*60}")
print(f"총 기사 수: {len(df_article):,}")
print(f"본문 수집 성공: {df_article['article_body'].notna().sum():,}")
print(f"본문 수집 실패: {df_article['article_body'].isna().sum():,}")
print(f"\n재시도 결과:")
print(f"  - 1단계(newspaper) 성공: {newspaper_success}개")
print(f"  - 2단계(Selenium) 성공: {selenium_success}개")
print(f"  - 최종 실패: {len(final_failed_indices)}개")

if len(final_failed_indices) > 0:
    print(f"\n최종 실패 URL (최대 10개):")
    for i, idx in enumerate(final_failed_indices[:10]):
        url = df_article.loc[idx, 'actual_url']
        publisher = df_article.loc[idx, 'publisher_title']
        print(f"  [{i+1}] {publisher} - {url}")
    if len(final_failed_indices) > 10:
        print(f"  ... 외 {len(final_failed_indices) - 10}개")

print(f"\n{'='*60}")
print(f"재시도 프로세스 완료!")
print(f"{'='*60}")
