"""
데이터 불러와서, LLM Agent에 전달할 수 있는 형태로 변환
해야할 일: 종목명을 dict에 추가해야함. --> done
"""

import pandas as pd
from pprint import pprint

def convert_df_to_agent_format(csv_path='../data/articles/csv/news_with_market_cap_20250929_180045.csv', symbol='ADBE', search_date='2024-12-11'):
    """
    CSV 파일을 읽어서 특정 symbol과 search_date의 데이터를 LLM Agent에 전달할 수 있는 딕셔너리 형태로 변환
    
    Args:
        csv_path (str): CSV 파일 경로
        symbol (str): 필터링할 종목 심볼 (예: 'AAPL', 'MSFT')
        search_date (str): 필터링할 검색 날짜 (예: '2025-09-29')
        
    Returns:
        dict: LLM Agent에 전달할 데이터 딕셔너리
            - search_date: 검색 날짜
            - titles: title 열의 모든 값을 ' / '로 연결한 문자열
            - descriptions: description 열의 모든 값을 ' / '로 연결한 문자열
            - sector: 해당 종목의 섹터 정보
    """
    # CSV 파일 읽기
    df = pd.read_csv(csv_path)
    
    # symbol과 search_date로 필터링
    df_filtered = df[(df['symbol'] == symbol) & (df['search_date'] == search_date)]
    
    # 데이터가 없는 경우 처리
    if len(df_filtered) == 0:
        return {
            'search_date': search_date,
            'titles': "",
            'descriptions': "",
            'sector': ""
        }
    
    # NaN 값을 빈 문자열로 처리
    df_filtered = df_filtered.fillna('')
    
    # 딕셔너리 생성
    agent_data = {
        'symbol': symbol,
        'search_date': search_date,
        'titles': ' / '.join(df_filtered['title'].astype(str).tolist()),
        'descriptions': ' / '.join(df_filtered['description'].astype(str).tolist()),
        'sector': df_filtered['Sector'].iloc[0] if 'Sector' in df_filtered.columns else ""
    }
    
    return agent_data


# 사용 예시
# if __name__ == "__main__":
#     # 파일 경로 설정
#     csv_path = '../data/articles/csv/news_with_market_cap_20250929_180045.csv'
    
#     # 예시
#     symbol = 'ADBE'
#     search_date = '2024-12-11'
#     result = convert_df_to_agent_format(csv_path, symbol, search_date)
    
#     pprint(result)