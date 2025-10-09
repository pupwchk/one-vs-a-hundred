import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# CSV 불러오기
df = pd.read_csv('../data/articles/csv/news_with_market_cap_20250929_180045.csv')
urls = df.url.tolist()

def format_time(seconds: int) -> str:
    """초 → HH:MM:SS 포맷 변환"""
    h, m, s = seconds // 3600, (seconds % 3600) // 60, seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def fetch_url(index, url, max_retries=3, wait_time=10):
    """
    하나의 URL을 처리해서 최종 리디렉션 URL 반환
    - 리디렉션 실패 시 재시도
    - WebDriverWait 사용
    - headless 모드 실행
    """
    for attempt in range(1, max_retries + 1):
        chrome_options = Options()
        chrome_options.add_argument("--headless")       # 창 안뜨게
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")

        driver = webdriver.Chrome(options=chrome_options)

        try:
            driver.get(url)
            time.sleep(7)

            final_url = driver.current_url

            if final_url == url:
                print(f"[{index}] 시도 {attempt}/{max_retries} → 리디렉션 실패 (같은 URL)")
                if attempt == max_retries:
                    return (index, None, False)
                continue  # 재시도

            print(f"[{index}] 시도 {attempt}/{max_retries} → 성공: {final_url}")
            return (index, final_url, True)

        except Exception as e:
            print(f"[{index}] 시도 {attempt}/{max_retries} → 에러: {e}")
            if attempt == max_retries:
                return (index, None, False)
        finally:
            driver.quit()

    return (index, None, False)


# 실행 시작 시간 기록
start_time = time.time()

# 결과 저장할 리스트
results = [None] * len(urls)

# 진행 상태 추적
total = len(urls)
completed = 0
success_count = 0
fail_count = 0

with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(fetch_url, i, url) for i, url in enumerate(urls)]

    for future in as_completed(futures):
        idx, actual_url, success = future.result()
        results[idx] = actual_url
        completed += 1
        if success:
            success_count += 1
        else:
            fail_count += 1

        # 시간 계산
        elapsed = int(time.time() - start_time)
        avg_time = elapsed / completed if completed > 0 else 0
        remaining = int((total - completed) * avg_time)

        print(f"[진행 상황] {completed}/{total} 완료 | 성공 {success_count} | 실패 {fail_count} "
              f"| 경과 {format_time(elapsed)} | ETA {format_time(remaining)}")

# df에 열 추가
df["actual_url"] = results

# CSV 저장
output_path = '../data/articles/csv/news_with_market_cap_with_actual_url.csv'
df.to_csv(output_path, index=False)

# 전체 경과 시간
elapsed_total = int(time.time() - start_time)

print(f"\n=== 전체 작업 완료 ===")
print(f"총 {total}개 중 성공 {success_count}, 실패 {fail_count}")
print(f"총 경과 시간: {format_time(elapsed_total)}")
print(f"결과 저장: {output_path}")
