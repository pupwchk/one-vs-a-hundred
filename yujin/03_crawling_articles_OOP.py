import pandas as pd
import json
import os
from gnews import GNews
from datetime import datetime, timedelta
from tqdm import tqdm
from pathlib import Path
import logging
from typing import List, Dict, Optional, Tuple
import dateutil.parser
import pytz
from dateutil.tz import gettz
import time


class NewsCollector:
    """
    주식 이벤트 데이터를 기반으로 전날 뉴스 기사만 수집하고 즉시 저장하는 클래스
    """
    
    def __init__(self, 
                 event_data_path: str = '../data/event_data.csv',
                 stock_data_path: str = '../data/stock_data.csv',
                 output_base_path: str = '../data/articles',
                 language: str = 'en',
                 country: str = 'US'):
        """
        NewsCollector 초기화
        
        Args:
            event_data_path: 이벤트 데이터 CSV 파일 경로
            stock_data_path: 주식 데이터 CSV 파일 경로
            output_base_path: 출력 파일들의 기본 경로
            language: 뉴스 언어
            country: 뉴스 국가
        """
        self.event_data_path = event_data_path
        self.stock_data_path = stock_data_path
        self.output_base_path = output_base_path
        self.language = language
        self.country = country
        
        # 출력 디렉토리 설정
        self.csv_output_dir = Path(output_base_path) / 'csv'
        self.json_output_dir = Path(output_base_path) / 'json'
        
        # 데이터 저장용 변수
        self.df_event = None
        self.df_stock = None
        self.error_log = []
        self.filter_stats = {
            'total_collected': 0,
            'date_filtered': 0,
            'final_count': 0,
            'processed_events': 0
        }
        
        # 파일 경로들
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_file_path = self.csv_output_dir / f"filtered_news_{self.timestamp}.csv"
        self.jsonl_file_path = self.json_output_dir / f"filtered_news_{self.timestamp}.jsonl"
        self.csv_initialized = False
        
        # GNews 객체
        self.google_news = GNews(language=language, country=country)
        
        # 초기화
        self._setup()
    
    def _setup(self):
        """초기 설정 및 데이터 로드"""
        # 로깅 설정 (먼저 설정해야 다른 메서드에서 사용 가능)
        self._setup_logging()
        
        # 출력 디렉토리 생성
        self._create_directories()
        
        # 데이터 로드
        self._load_data()
    
    def _create_directories(self):
        """출력 디렉토리 생성"""
        self.csv_output_dir.mkdir(parents=True, exist_ok=True)
        self.json_output_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_data(self):
        """데이터 파일 로드"""
        try:
            self.df_event = pd.read_csv(self.event_data_path)
            self.df_stock = pd.read_csv(self.stock_data_path)
            
            # Date 컬럼을 datetime으로 변환
            self.df_event["Date"] = pd.to_datetime(self.df_event["Date"])
            
            self.logger.info(f"이벤트 데이터 로드 완료: {len(self.df_event)}개 항목")
            self.logger.info(f"주식 데이터 로드 완료: {len(self.df_stock)}개 항목")
            
        except Exception as e:
            self.logger.error(f"데이터 로드 실패: {str(e)}")
            raise
    
    def _log_error(self, date: datetime, symbol: str, error_message: str, error_type: str = 'general'):
        """에러 로그 기록"""
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'date': date.isoformat() if isinstance(date, datetime) else str(date),
            'symbol': symbol,
            'error_type': error_type,
            'error_message': str(error_message)
        }
        self.error_log.append(error_entry)
        self.logger.error(f"에러 발생 - 날짜: {date}, 심볼: {symbol}, 타입: {error_type}, 에러: {error_message}")
    
    def _save_error_log(self):
        """에러 로그를 JSON 파일로 저장"""
        if self.error_log:
            error_log_path = Path(self.output_base_path) / 'errorlog.json'
            try:
                with open(error_log_path, 'w', encoding='utf-8') as f:
                    json.dump(self.error_log, f, ensure_ascii=False, indent=2)
                self.logger.info(f"에러 로그 저장 완료: {error_log_path}")
            except Exception as e:
                self.logger.error(f"에러 로그 저장 실패: {str(e)}")
    
    def _parse_published_date(self, published_date_str: str) -> Optional[datetime]:
        """
        published date 문자열을 datetime 객체로 파싱
        
        Args:
            published_date_str: 발행일 문자열 (예: "Wed, 09 Apr 2025 07:00:00 GMT")
            
        Returns:
            파싱된 datetime 객체 또는 None (파싱 실패 시)
        """
        try:
            if not published_date_str or published_date_str.strip() == '':
                return None
            
            # dateutil.parser로 다양한 형태의 날짜 문자열 파싱
            parsed_date = dateutil.parser.parse(published_date_str)
            return parsed_date
            
        except Exception as e:
            self.logger.debug(f"날짜 파싱 실패: {published_date_str} - {str(e)}")
            return None
    
    def _is_target_date(self, published_date: datetime, target_date: datetime) -> bool:
        """
        발행일이 목표 날짜와 같은지 확인 (날짜만 비교, 시간 무시)
        
        Args:
            published_date: 기사 발행일
            target_date: 목표 날짜 (이벤트 전날)
            
        Returns:
            같은 날짜인지 여부
        """
        try:
            # 두 날짜를 날짜만으로 비교 (시간 정보 제거)
            pub_date = published_date.date()
            target_date = target_date.date()
            
            return pub_date == target_date
            
        except Exception as e:
            self.logger.debug(f"날짜 비교 실패: {str(e)}")
            return False
    
    def _filter_news_by_date(self, news_list: List[Dict], target_date: datetime) -> List[Dict]:
        """
        뉴스 목록에서 목표 날짜에 해당하는 기사만 필터링
        
        Args:
            news_list: 뉴스 기사 목록
            target_date: 목표 날짜 (이벤트 전날)
            
        Returns:
            필터링된 뉴스 목록
        """
        filtered_news = []
        
        for news_item in news_list:
            published_date_str = news_item.get('published_date', '')
            
            # 날짜 파싱
            published_date = self._parse_published_date(published_date_str)
            
            if published_date is None:
                # 날짜 파싱 실패한 경우 제외
                continue
            
            # 목표 날짜와 비교
            if self._is_target_date(published_date, target_date):
                # 추가 정보를 기사에 포함
                news_item['parsed_published_date'] = published_date.isoformat()
                news_item['is_target_date'] = True
                filtered_news.append(news_item)
            else:
                # 필터링된 기사 정보 로그 (디버깅용)
                self.logger.debug(f"날짜 불일치로 제외: {published_date.date()} != {target_date.date()}")
        
        return filtered_news
    
    def _save_to_csv_immediately(self, news_data: List[Dict]):
        """뉴스 데이터를 CSV 파일에 즉시 저장"""
        if not news_data:
            return
        
        try:
            df_news = pd.DataFrame(news_data)
            
            # 첫 번째 저장시에만 header 포함
            if not self.csv_initialized:
                df_news.to_csv(self.csv_file_path, index=False, encoding='utf-8', mode='w')
                self.csv_initialized = True
                self.logger.info(f"CSV 파일 생성: {self.csv_file_path}")
            else:
                df_news.to_csv(self.csv_file_path, index=False, encoding='utf-8', mode='a', header=False)
            
            self.logger.debug(f"CSV에 {len(news_data)}개 뉴스 추가 저장")
            
        except Exception as e:
            self.logger.error(f"CSV 즉시 저장 실패: {str(e)}")
    
    def _save_to_jsonl_immediately(self, news_data: List[Dict]):
        """뉴스 데이터를 JSONL 파일에 즉시 저장 (JSON Lines 형식)"""
        if not news_data:
            return
        
        try:
            with open(self.jsonl_file_path, 'a', encoding='utf-8') as f:
                for news_item in news_data:
                    json_line = json.dumps(news_item, ensure_ascii=False)
                    f.write(json_line + '\n')
            
            self.logger.debug(f"JSONL에 {len(news_data)}개 뉴스 추가 저장")
            
        except Exception as e:
            self.logger.error(f"JSONL 즉시 저장 실패: {str(e)}")
    
    def collect_news_for_date(self, date: datetime, symbol: str) -> List[Dict]:
        """
        특정 날짜와 심볼에 대한 뉴스 수집, 필터링 및 즉시 저장
        
        Args:
            date: 이벤트 발생 날짜
            symbol: 주식 심볼
            
        Returns:
            전날 뉴스 기사 리스트 (즉시 저장됨)
        """
        try:
            # 이벤트 전날의 뉴스를 검색
            search_date = date - timedelta(days=1)
            
            # 날짜 범위 설정 (GNews 경고 방지를 위해 end_date를 하루 뒤로 설정)
            self.google_news.start_date = search_date.date()
            self.google_news.end_date = search_date.date() + timedelta(days=1)
            
            # 뉴스 검색
            news_results = self.google_news.get_news(symbol)
            
            # 결과 정리
            processed_news = []
            for news in news_results:
                # publisher 처리
                publisher_info = news.get('publisher', {})
                if isinstance(publisher_info, dict):
                    publisher_title = publisher_info.get('title', '')
                    publisher_href = publisher_info.get('href', '')
                else:
                    publisher_title = str(publisher_info)
                    publisher_href = ''
                
                news_item = {
                    'event_date': date.isoformat(),
                    'search_date': search_date.date().isoformat(),
                    'symbol': symbol,
                    'title': news.get('title', ''),
                    'description': news.get('description', ''),
                    'published_date': news.get('published date', ''),
                    'url': news.get('url', ''),
                    'publisher_title': publisher_title,
                    'publisher_href': publisher_href
                }
                processed_news.append(news_item)
            
            # 통계 업데이트
            self.filter_stats['total_collected'] += len(processed_news)
            
            # 전날 날짜에 해당하는 뉴스만 필터링
            filtered_news = self._filter_news_by_date(processed_news, search_date)
            
            # 필터링 통계 업데이트
            filtered_count = len(processed_news) - len(filtered_news)
            self.filter_stats['date_filtered'] += filtered_count
            self.filter_stats['final_count'] += len(filtered_news)
            
            # 즉시 저장
            if filtered_news:
                self._save_to_csv_immediately(filtered_news)
                self._save_to_jsonl_immediately(filtered_news)
            
            self.logger.info(f"{symbol} ({search_date.date()}): 수집 {len(processed_news)}개 → 필터링 후 {len(filtered_news)}개 → 즉시 저장 완료")
            
            return filtered_news
            
        except Exception as e:
            self._log_error(date, symbol, str(e), 'news_collection')
            return []
    
    def collect_all_news(self):
        """모든 이벤트에 대한 뉴스 수집 및 즉시 저장"""
        self.logger.info("뉴스 수집 시작...")
        self.logger.info(f"CSV 파일: {self.csv_file_path}")
        self.logger.info(f"JSONL 파일: {self.jsonl_file_path}")
        
        # 초기화
        self.filter_stats = {
            'total_collected': 0,
            'date_filtered': 0,
            'final_count': 0,
            'processed_events': 0
        }
        
        # 진행 상황 표시를 위한 tqdm
        for index, row in tqdm(self.df_event.iterrows(), 
                              total=len(self.df_event), 
                              desc="뉴스 수집, 필터링 및 저장"):
            
            date = row['Date']
            symbol = row['Symbol']
            
            self.logger.info(f"처리 중: {symbol} - 이벤트 날짜: {date.date()}, 검색 날짜: {(date - timedelta(days=1)).date()}")
            
            # 뉴스 수집, 필터링 및 즉시 저장
            filtered_news_data = self.collect_news_for_date(date, symbol)
            
            # 처리된 이벤트 수 증가
            self.filter_stats['processed_events'] += 1
            
            # API 호출 제한을 위한 잠시 대기 (필요시)
            time.sleep(0.3)
        
        self.logger.info("뉴스 수집, 필터링 및 저장 완료")
        self.logger.info(f"처리된 이벤트 수: {self.filter_stats['processed_events']}")
        self.logger.info(f"총 수집: {self.filter_stats['total_collected']}개")
        self.logger.info(f"날짜 불일치로 제외: {self.filter_stats['date_filtered']}개")
        self.logger.info(f"최종 저장: {self.filter_stats['final_count']}개")
    
    def _convert_jsonl_to_json(self):
        """JSONL 파일을 표준 JSON 배열 형식으로 변환"""
        json_file_path = self.json_output_dir / f"filtered_news_{self.timestamp}.json"
        
        try:
            if not self.jsonl_file_path.exists():
                self.logger.warning("JSONL 파일이 존재하지 않습니다.")
                return None
            
            all_records = []
            with open(self.jsonl_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        all_records.append(json.loads(line.strip()))
            
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(all_records, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"JSON 변환 완료: {json_file_path}")
            return str(json_file_path)
            
        except Exception as e:
            self.logger.error(f"JSONL을 JSON으로 변환 실패: {str(e)}")
            return None
    
    def save_filter_stats(self):
        """필터링 통계를 JSON 파일로 저장"""
        stats_path = Path(self.output_base_path) / 'filter_stats.json'
        
        detailed_stats = {
            'timestamp': datetime.now().isoformat(),
            'total_events_processed': self.filter_stats['processed_events'],
            'filter_statistics': self.filter_stats,
            'filtering_rate': {
                'kept_percentage': (self.filter_stats['final_count'] / max(self.filter_stats['total_collected'], 1)) * 100,
                'filtered_percentage': (self.filter_stats['date_filtered'] / max(self.filter_stats['total_collected'], 1)) * 100
            },
            'error_count': len(self.error_log),
            'output_files': {
                'csv_file': str(self.csv_file_path),
                'jsonl_file': str(self.jsonl_file_path)
            }
        }
        
        try:
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(detailed_stats, f, ensure_ascii=False, indent=2)
            self.logger.info(f"필터링 통계 저장 완료: {stats_path}")
        except Exception as e:
            self.logger.error(f"통계 파일 저장 실패: {str(e)}")
    
    def run(self, save_json: bool = True):
        """
        전체 프로세스 실행 (CSV와 JSONL은 자동으로 즉시 저장됨)
        
        Args:
            save_json: JSONL을 표준 JSON 형식으로도 저장할지 여부
        """
        try:
            # 뉴스 수집, 필터링 및 즉시 저장
            self.collect_all_news()
            
            # JSONL을 표준 JSON 형식으로 변환 (선택사항)
            json_path = None
            if save_json:
                json_path = self._convert_jsonl_to_json()
            
            # 통계 및 에러 로그 저장
            self.save_filter_stats()
            self._save_error_log()
            
            # 수집 결과 요약
            self._print_summary()
            
            return {
                'csv_path': str(self.csv_file_path),
                'jsonl_path': str(self.jsonl_file_path),
                'json_path': json_path,
                'stats': self.filter_stats,
                'error_count': len(self.error_log)
            }
            
        except Exception as e:
            self.logger.error(f"프로세스 실행 중 오류 발생: {str(e)}")
            raise
    
    def _print_summary(self):
        """수집 결과 요약 출력"""
        print("\n" + "="*60)
        print("뉴스 수집 및 필터링 결과 요약")
        print("="*60)
        print(f"총 처리된 이벤트 수: {self.filter_stats['processed_events']}")
        print(f"수집된 전체 뉴스 수: {self.filter_stats['total_collected']}")
        print(f"날짜 불일치로 제외된 뉴스 수: {self.filter_stats['date_filtered']}")
        print(f"최종 전날 뉴스 수: {self.filter_stats['final_count']}")
        
        if self.filter_stats['total_collected'] > 0:
            keep_rate = (self.filter_stats['final_count'] / self.filter_stats['total_collected']) * 100
            print(f"필터링 유지율: {keep_rate:.1f}%")
        
        print(f"발생한 에러 수: {len(self.error_log)}")
        print(f"출력 파일:")
        print(f"  - CSV: {self.csv_file_path}")
        print(f"  - JSONL: {self.jsonl_file_path}")
        print(f"출력 디렉토리: {self.output_base_path}")
        print("="*60)


# 사용 예시
if __name__ == "__main__":
    # NewsCollector 인스턴스 생성
    collector = NewsCollector(
        event_data_path='../data/event_data.csv',
        stock_data_path='../data/stock_data.csv',
        output_base_path='../data/articles'
    )
    
    # 뉴스 수집, 필터링 및 즉시 저장 실행
    result = collector.run(save_json=True)
    
    print(f"\n처리 완료!")
    print(f"CSV 파일: {result['csv_path']}")
    print(f"JSONL 파일: {result['jsonl_path']}")
    print(f"JSON 파일: {result['json_path']}")
    print(f"최종 뉴스 수: {result['stats']['final_count']}")
    print(f"에러 수: {result['error_count']}")