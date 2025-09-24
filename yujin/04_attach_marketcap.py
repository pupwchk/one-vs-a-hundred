"""이벤트 발생 전 날의 시가 총액 계산 및 병합"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from tqdm import tqdm
import json
import os

print("시가총액 계산 스크립트 시작")
print("="*50)

# 1. 데이터 로드
input_file = '../data/articles/csv/filtered_news_20250924_043743.csv'
print(f"데이터 로드: {input_file}")

df = pd.read_csv(input_file)
print(f"데이터 로드 성공: {len(df):,}개 뉴스 기사")
print(f"컬럼: {list(df.columns)}")
print(f"고유 심볼 수: {df['symbol'].nunique()}")
print(f"날짜 범위: {df['search_date'].min()} ~ {df['search_date'].max()}")

print("\n" + "="*50)

# 2. 시가총액 계산
print("시가총액 계산 시작...")

market_caps = []
error_log = []
successful_count = 0
failed_count = 0

# 각 행에 대해 시가총액 계산
for idx, row in tqdm(df.iterrows(), total=len(df), desc="시가총액 계산 중"):
    symbol = row['symbol']
    search_date = row['search_date']  # 'YYYY-MM-DD' 형식
    
    try:
        # 날짜 문자열을 datetime 객체로 변환
        target_date = datetime.strptime(search_date, '%Y-%m-%d').date()
        
        # yfinance ticker 객체 생성
        ticker = yf.Ticker(symbol)
        
        # 발행주식수 가져오기
        info = ticker.info
        shares_outstanding = info.get('sharesOutstanding')
        
        if not shares_outstanding:
            shares_outstanding = info.get('impliedSharesOutstanding')
        
        if not shares_outstanding:
            print(f"경고: {symbol}: 발행주식수 정보 없음")
            market_caps.append(None)
            failed_count += 1
            error_log.append({
                'index': idx,
                'symbol': symbol,
                'search_date': search_date,
                'error': 'no_shares_outstanding_data'
            })
            continue
        
        # 해당 날짜 주변의 주가 데이터 가져오기
        start_date = target_date - timedelta(days=7)
        end_date = target_date + timedelta(days=1)
        
        hist = ticker.history(start=start_date, end=end_date)
        
        if hist.empty:
            print(f"경고: {symbol} ({search_date}): 주가 데이터 없음")
            market_caps.append(None)
            failed_count += 1
            error_log.append({
                'index': idx,
                'symbol': symbol,
                'search_date': search_date,
                'error': 'no_price_data'
            })
            continue
        
        # 목표 날짜에 가장 가까운 거래일의 종가 찾기
        target_close = None
        
        # 정확한 날짜의 데이터가 있는지 확인
        if target_date.strftime('%Y-%m-%d') in [d.strftime('%Y-%m-%d') for d in hist.index.date]:
            target_close = hist.loc[hist.index.date == target_date, 'Close'].iloc[0]
        else:
            # 목표 날짜 이전의 가장 가까운 거래일 찾기
            available_dates = [d.date() for d in hist.index]
            available_dates = [d for d in available_dates if d <= target_date]
            
            if available_dates:
                closest_date = max(available_dates)
                target_close = hist.loc[hist.index.date == closest_date, 'Close'].iloc[0]
            else:
                print(f"경고: {symbol} ({search_date}): 적절한 거래일 데이터 없음")
                market_caps.append(None)
                failed_count += 1
                error_log.append({
                    'index': idx,
                    'symbol': symbol,
                    'search_date': search_date,
                    'error': 'no_suitable_trading_day'
                })
                continue
        
        if target_close is None:
            print(f"경고: {symbol} ({search_date}): 종가 데이터 없음")
            market_caps.append(None)
            failed_count += 1
            error_log.append({
                'index': idx,
                'symbol': symbol,
                'search_date': search_date,
                'error': 'no_close_price'
            })
            continue
        
        # 시가총액 계산 (발행주식수 × 종가)
        market_cap = shares_outstanding * target_close
        market_caps.append(market_cap)
        successful_count += 1
        
        print(f"성공: {symbol} ({search_date}): 시가총액 ${market_cap:,.0f} (주식수: {shares_outstanding:,}, 종가: ${target_close:.2f})")
        
    except Exception as e:
        print(f"에러: {symbol} ({search_date}): 에러 발생 - {str(e)}")
        market_caps.append(None)
        failed_count += 1
        error_log.append({
            'index': idx,
            'symbol': symbol,
            'search_date': search_date,
            'error': str(e)
        })

# 새로운 컬럼 추가
df['market_cap'] = market_caps

print(f"\n시가총액 계산 완료!")
print(f"성공: {successful_count}개")
print(f"실패: {failed_count}개")

# 에러 로그 저장
if error_log:
    error_log_path = '../data/articles/market_cap_errors.json'
    with open(error_log_path, 'w', encoding='utf-8') as f:
        json.dump(error_log, f, ensure_ascii=False, indent=2)
    print(f"에러 로그 저장: {error_log_path}")

print("\n" + "="*50)

# 3. 결과 저장
print("결과 파일 저장...")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# CSV 저장
csv_output = f'../data/articles/csv/news_with_market_cap_{timestamp}.csv'
df.to_csv(csv_output, index=False, encoding='utf-8')
print(f"CSV 저장 완료: {csv_output}")

# JSON 저장
json_output = f'../data/articles/json/news_with_market_cap_{timestamp}.json'
df.to_json(json_output, orient='records', force_ascii=False, indent=2)
print(f"JSON 저장 완료: {json_output}")

# 통계 요약
print(f"\n최종 통계:")
print(f"총 뉴스 기사 수: {len(df):,}")
print(f"시가총액 데이터 있음: {df['market_cap'].notna().sum():,}")
print(f"시가총액 데이터 없음: {df['market_cap'].isna().sum():,}")

if df['market_cap'].notna().any():
    print(f"평균 시가총액: ${df['market_cap'].mean():,.0f}")
    print(f"중간값 시가총액: ${df['market_cap'].median():,.0f}")
    print(f"최대 시가총액: ${df['market_cap'].max():,.0f}")
    print(f"최소 시가총액: ${df['market_cap'].min():,.0f}")

print("\n" + "="*50)
print("모든 작업 완료!")
print(f"결과 파일:")
print(f"  - CSV: {csv_output}")
print(f"  - JSON: {json_output}")