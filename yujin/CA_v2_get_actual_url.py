# 이것이 리얼 #
import pandas as pd
import time
import threading
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException, WebDriverException

# CSV 불러오기
df = pd.read_csv('../data/articles/csv/news_with_market_cap_20250929_180045.csv')
urls = df.url.tolist()

# Thread-local storage for WebDriver instances
thread_local = threading.local()

def get_driver():
    """각 스레드마다 하나의 WebDriver 인스턴스 재사용"""
    if not hasattr(thread_local, 'driver'):
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-logging")
        chrome_options.add_argument("--disable-images")
        chrome_options.add_argument("--blink-settings=imagesEnabled=false")
        chrome_options.page_load_strategy = 'eager'
        
        service = Service(log_path='NUL' if os.name == 'nt' else '/dev/null')
        thread_local.driver = webdriver.Chrome(service=service, options=chrome_options)
        thread_local.driver.set_page_load_timeout(25)  # 더 긴 타임아웃
    
    return thread_local.driver

def format_time(seconds: int) -> str:
    """초 → HH:MM:SS 포맷 변환"""
    h, m, s = seconds // 3600, (seconds % 3600) // 60, seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def fetch_url(index, url, max_retries=3, wait_time=5, retry_phase=1):
    """
    하나의 URL을 처리해서 최종 리디렉션 URL 반환
    
    Args:
        index: URL 인덱스
        url: 처리할 URL
        max_retries: 최대 재시도 횟수
        wait_time: 페이지 로드 후 대기 시간 (초)
        retry_phase: 재시도 단계 (1차/2차 표시용)
    
    Returns:
        (index, actual_url or None, success, failure_type)
    """
    driver = get_driver()
    failure_type = None
    
    for attempt in range(1, max_retries + 1):
        try:
            driver.get(url)
            time.sleep(wait_time)
            
            final_url = driver.current_url

            if final_url == url:
                failure_type = "리디렉션 실패"
                print(f"[{index}] {retry_phase}차-시도{attempt}/{max_retries} (대기{wait_time}초) → {failure_type}")
                if attempt == max_retries:
                    return (index, None, False, failure_type)
                time.sleep(1)
                continue

            print(f"[{index}] {retry_phase}차-시도{attempt}/{max_retries} (대기{wait_time}초) → 성공: {final_url[:80]}...")
            return (index, final_url, True, None)

        except TimeoutException:
            failure_type = "타임아웃"
            print(f"[{index}] {retry_phase}차-시도{attempt}/{max_retries} → {failure_type}")
            if attempt == max_retries:
                return (index, None, False, failure_type)
            
        except WebDriverException as e:
            failure_type = "WebDriver 에러"
            print(f"[{index}] {retry_phase}차-시도{attempt}/{max_retries} → {failure_type}: {str(e)[:50]}")
            
            # WebDriver 재생성이 필요한 경우
            if "invalid session id" in str(e).lower() or "session deleted" in str(e).lower():
                if hasattr(thread_local, 'driver'):
                    try:
                        thread_local.driver.quit()
                    except:
                        pass
                    delattr(thread_local, 'driver')
                driver = get_driver()
            
            if attempt == max_retries:
                return (index, None, False, failure_type)
                
        except Exception as e:
            failure_type = "예상치 못한 에러"
            print(f"[{index}] {retry_phase}차-시도{attempt}/{max_retries} → {failure_type}: {str(e)[:50]}")
            if attempt == max_retries:
                return (index, None, False, failure_type)
        
        time.sleep(1)
    
    return (index, None, False, failure_type or "알 수 없는 에러")

def cleanup_drivers():
    """모든 스레드의 WebDriver 정리"""
    if hasattr(thread_local, 'driver'):
        try:
            thread_local.driver.quit()
        except:
            pass

# ========== 1차 시도 (5초 대기, 3회 재시도) ==========
print(f"총 {len(urls)}개 URL 처리 시작...\n")
start_time = time.time()

results = [None] * len(urls)
failure_info = {}  # {index: failure_type}
total = len(urls)
completed = 0
success_count = 0
fail_count = 0

MAX_WORKERS = 15
print(f"병렬 스레드 수: {MAX_WORKERS}")
print(f"{'='*60}")
print(f"1차 시도 - 대기 시간: 5초, 재시도: 3회")
print(f"{'='*60}\n")

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(fetch_url, i, url, max_retries=3, wait_time=5, retry_phase=1) 
               for i, url in enumerate(urls)]

    for future in as_completed(futures):
        idx, actual_url, success, fail_type = future.result()
        results[idx] = actual_url
        completed += 1
        
        if success:
            success_count += 1
        else:
            fail_count += 1
            failure_info[idx] = fail_type

        elapsed = int(time.time() - start_time)
        avg_time = elapsed / completed if completed > 0 else 0
        remaining = int((total - completed) * avg_time)

        if completed % 10 == 0 or completed == total:
            print(f"\n[1차 진행] {completed}/{total} ({100*completed/total:.1f}%) | "
                  f"✓ {success_count} | ✗ {fail_count} | "
                  f"경과 {format_time(elapsed)} | ETA {format_time(remaining)}")

print(f"\n{'='*60}")
print(f"1차 시도 완료")
print(f"{'='*60}")
print(f"성공: {success_count}개, 실패: {fail_count}개")

# 실패 유형별 통계
if failure_info:
    from collections import Counter
    fail_types = Counter(failure_info.values())
    print(f"\n실패 유형 분석:")
    for fail_type, count in fail_types.most_common():
        print(f"  - {fail_type}: {count}개")

# ========== 2차 시도 (실패한 모든 링크, 15초 대기, 5회 재시도) ==========
failed_indices = list(failure_info.keys())

if failed_indices:
    print(f"\n{'='*60}")
    print(f"2차 시도 - 대기 시간: 15초, 재시도: 5회")
    print(f"{'='*60}")
    print(f"총 {len(failed_indices)}개 실패 링크 재처리 시작...\n")
    
    retry_start = time.time()
    retry_completed = 0
    retry_success = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        retry_futures = [
            executor.submit(fetch_url, idx, urls[idx], max_retries=5, wait_time=15, retry_phase=2) 
            for idx in failed_indices
        ]
        
        for future in as_completed(retry_futures):
            idx, actual_url, success, fail_type = future.result()
            retry_completed += 1
            
            if success:
                results[idx] = actual_url
                retry_success += 1
                success_count += 1
                fail_count -= 1
                del failure_info[idx]  # 성공했으므로 실패 정보 제거
            else:
                failure_info[idx] = fail_type  # 실패 유형 업데이트
            
            elapsed = int(time.time() - retry_start)
            avg_time = elapsed / retry_completed if retry_completed > 0 else 0
            remaining = int((len(failed_indices) - retry_completed) * avg_time)
            
            if retry_completed % 5 == 0 or retry_completed == len(failed_indices):
                print(f"\n[2차 진행] {retry_completed}/{len(failed_indices)} "
                      f"({100*retry_completed/len(failed_indices):.1f}%) | "
                      f"✓ {retry_success} | ✗ {len(failed_indices)-retry_success} | "
                      f"경과 {format_time(elapsed)} | ETA {format_time(remaining)}")
    
    print(f"\n{'='*60}")
    print(f"2차 시도 완료")
    print(f"{'='*60}")
    print(f"추가 성공: {retry_success}개, 최종 실패: {fail_count}개")

# 정리
cleanup_drivers()

# 결과 저장
df["actual_url"] = results
output_path = '../data/articles/csv/news_with_market_cap_with_actual_url.csv'
df.to_csv(output_path, index=False)

# ========== 최종 실패 링크 저장 ==========
if failure_info:
    print(f"\n{'='*60}")
    print(f"⚠️  최종 실패 링크: {len(failure_info)}개")
    print(f"{'='*60}")
    
    # 실패 유형별 통계
    from collections import Counter
    final_fail_types = Counter(failure_info.values())
    print(f"\n최종 실패 유형:")
    for fail_type, count in final_fail_types.most_common():
        print(f"  - {fail_type}: {count}개")
    
    print(f"\n최종 실패 링크 목록 (최대 10개):")
    for i, idx in enumerate(list(failure_info.keys())[:10]):
        print(f"  [{idx}] {failure_info[idx]} - {urls[idx]}")
    if len(failure_info) > 10:
        print(f"  ... 외 {len(failure_info) - 10}개")
    
    # 실패 링크 데이터프레임 생성
    failed_df = df.iloc[list(failure_info.keys())].copy()
    failed_df['failure_type'] = [failure_info[idx] for idx in failure_info.keys()]
    
    failed_output = '../data/articles/csv/failed_redirects.csv'
    failed_df.to_csv(failed_output, index=False)
    print(f"\n최종 실패 링크 저장: {failed_output}")
    print(f"  (failure_type 컬럼 포함)")

# ========== 전체 통계 ==========
elapsed_total = int(time.time() - start_time)
print(f"\n{'='*60}")
print(f"✅ 전체 작업 완료")
print(f"{'='*60}")
print(f"총 처리: {total}개")
print(f"최종 성공: {success_count}개 ({100*success_count/total:.1f}%)")
print(f"최종 실패: {fail_count}개 ({100*fail_count/total:.1f}%)")
print(f"총 경과 시간: {format_time(elapsed_total)}")
print(f"평균 처리 시간: {elapsed_total/total:.2f}초/URL")
print(f"\n결과 파일:")
print(f"  - 전체 결과: {output_path}")
if failure_info:
    print(f"  - 실패 링크: {failed_output}")
print(f"{'='*60}")