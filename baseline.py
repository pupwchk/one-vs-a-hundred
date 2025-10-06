import json
import os
import random
from abc import ABC, abstractmethod
from typing import List, Dict, TypedDict, Optional, Any, Tuple
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import pandas as pd

# .env 파일에서 환경 변수 로드 (API 저장 변수 = OPENROUTER_API_KEY)
load_dotenv()

# --- 응답 형식 정의 ---
class PeriodPrediction(TypedDict):
    decision: str # 살지 팔지
    confidence: int # 신뢰도
    reason: Optional[str] # API 응답에 포함되는 reason 필드 처리를 위해 추가

class AgentPrediction(TypedDict):
    day_1: PeriodPrediction
    day_3: PeriodPrediction
    day_7: PeriodPrediction
    day_15: PeriodPrediction
    day_30: PeriodPrediction

# --- 페르소나 데이터 로딩 함수 ---
def load_persona_data(file_path: str) -> List[Dict]:
    """페르소나 데이터를 로딩합니다."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"페르소나 데이터 로딩 오류: {e}")
        return []

def select_random_personas(persona_data: List[Dict], num_personas: int) -> List[Dict]:
    """페르소나 데이터에서 랜덤하게 선택합니다."""
    return random.sample(persona_data, min(num_personas, len(persona_data)))

def format_persona_profile(persona: Dict) -> str:
    """페르소나 프로필을 문자열로 포맷팅합니다."""
    profile = persona.get("PERSONA_PROFILE", {})
    
    identity = profile.get("Identity", {})
    socio = profile.get("Socio_Economics", {})
    outlook = profile.get("Outlook_and_Trust", {})
    info_habit = profile.get("Information_Habit", {})
    
    persona_text = f"""당신은 다음 페르소나를 가진 투자자입니다:
- 국적: {identity.get('Nationality', 'Unknown')}
- 나이: {identity.get('Age', 'Unknown')}
- 성별: {identity.get('Gender', 'Unknown')}
- 교육수준: {identity.get('Education', 'Unknown')}
- 직업: {socio.get('Occupation', 'Unknown')}
- 소득수준: {socio.get('Income_Decile', 'Unknown')}
- 정치성향: {socio.get('Left_Right_Position', 'Unknown')}
- 전반적 행복도: {outlook.get('Overall_Happiness', 'Unknown')}
- 국가경제 전망: {outlook.get('Natl_Econ_View', 'Unknown')}
- 가계재정 전망: {outlook.get('House_Fin_View', 'Unknown')}
- 제도신뢰도: {outlook.get('Inst_Trust', 'Unknown')}
- 대인신뢰도: {outlook.get('Interpersonal_Trust', 'Unknown')}
- 정치관심도: {info_habit.get('Political_Interest', 'Unknown')}
- 뉴스빈도: {info_habit.get('News_Frequency', 'Unknown')}
- 인터넷사용빈도: {info_habit.get('Internet_Frequency', 'Unknown')}"""
    
    return persona_text

# --- 프롬프트 생성 함수 ---
def create_stock_prompt(stock_data: str, persona_profile: Optional[str] = None) -> str:
    """주식 예측을 위한 표준 프롬프트를 생성합니다."""
    # 다중 기간 매수/매도 결정을 위한 프롬프트
    base_prompt = (
        "Based on the provided stock news data, decide whether you would 'buy' or 'sell' the stock for different time periods: 1 day, 3 days, 7 days, 15 days, and 30 days from the data date. "
        "For each period, provide your confidence level as a single integer from 0 to 100 and a brief reason for your decision. "
        "Your response MUST be a JSON object with five keys: 'day_1', 'day_3', 'day_7', 'day_15', 'day_30'. "
        "Each period should have 'decision' (buy/sell), 'confidence' (0-100), and 'reason' (brief explanation).\n\n"
        "Example: {\"day_1\": {\"decision\": \"buy\", \"confidence\": 75, \"reason\": \"Strong Q4 results\"}, \"day_3\": {\"decision\": \"sell\", \"confidence\": 60, \"reason\": \"Market volatility\"}, ...}\n\n"
        "Respond ONLY with a valid JSON object. Do not include any extra text.\n\n"
    )

    if persona_profile:
        base_prompt = f"{persona_profile}\n\n{base_prompt}"

    return f"{base_prompt}Data: {stock_data}"

# --- 주식 심볼 추출 함수 ---
def extract_stock_symbol(stock_data: str) -> str:
    """stock_data에서 주식 심볼을 추출합니다."""
    try:
        data_dict = json.loads(stock_data)
        # descriptions나 titles에서 NASDAQ:SYMBOL 형태를 찾아 추출
        text_to_search = data_dict.get('descriptions', '') + ' ' + data_dict.get('titles', '')
        
        import re
        # NASDAQ:SYMBOL 패턴 찾기
        nasdaq_match = re.search(r'NASDAQ:([A-Z]+)', text_to_search)
        if nasdaq_match:
            return nasdaq_match.group(1)
        
        # 괄호 안의 심볼 찾기 (ADBE) 형태
        paren_match = re.search(r'\(([A-Z]{2,5})\)', text_to_search)
        if paren_match:
            return paren_match.group(1)
            
        return "UNKNOWN"
    except:
        return "UNKNOWN"

# --- 날짜 계산 함수 ---
def calculate_prediction_dates(search_date: str) -> Dict[str, str]:
    """search_date를 기준으로 각 예측 기간의 정확한 날짜를 계산합니다."""
    try:
        base_date = datetime.strptime(search_date, '%Y-%m-%d')

        prediction_dates = {}
        periods = {
            'day_1': 1,
            'day_3': 3,
            'day_7': 7,
            'day_15': 15,
            'day_30': 30
        }

        for period_name, days in periods.items():
            prediction_date = base_date + timedelta(days=days)
            prediction_dates[period_name] = prediction_date.strftime('%Y-%m-%d')

        return prediction_dates
    except Exception as e:
        print(f"날짜 계산 오류: {e}")
        # 오류 발생 시 기본값 반환
        return {
            'day_1': 'UNKNOWN',
            'day_3': 'UNKNOWN',
            'day_7': 'UNKNOWN',
            'day_15': 'UNKNOWN',
            'day_30': 'UNKNOWN'
        }

# --- CSV 데이터 로딩 함수 ---
def load_events_from_csv(csv_path: str, symbols: Optional[List[str]] = None, dates: Optional[List[str]] = None) -> List[Tuple[str, str]]:
    """
    CSV 파일에서 이벤트 목록(symbol, search_date 조합)을 추출합니다.

    Args:
        csv_path: CSV 파일 경로
        symbols: 필터링할 종목 리스트 (None이면 전체)
        dates: 필터링할 날짜 리스트 (None이면 전체)

    Returns:
        [(symbol, search_date), ...] 형태의 유니크한 이벤트 튜플 리스트
    """
    try:
        df = pd.read_csv(csv_path)

        # 필터링
        if symbols is not None:
            df = df[df['symbol'].isin(symbols)]
        if dates is not None:
            df = df[df['search_date'].isin(dates)]

        # (symbol, search_date) 조합의 유니크한 값만 추출
        events = df[['symbol', 'search_date']].drop_duplicates().values.tolist()

        return [(str(symbol), str(date)) for symbol, date in events]
    except Exception as e:
        print(f"CSV 로딩 오류: {e}")
        return []

def convert_df_to_agent_format(csv_path: str, symbol: str, search_date: str) -> Dict[str, str]:
    """
    CSV 파일을 읽어서 특정 symbol과 search_date의 데이터를 LLM Agent에 전달할 수 있는 딕셔너리 형태로 변환
    (yujin/E_data_form_making.py의 함수를 baseline.py에 통합)

    Args:
        csv_path: CSV 파일 경로
        symbol: 필터링할 종목 심볼 (예: 'AAPL', 'MSFT')
        search_date: 필터링할 검색 날짜 (예: '2024-03-14')

    Returns:
        dict: LLM Agent에 전달할 데이터 딕셔너리
            - symbol: 종목 심볼
            - search_date: 검색 날짜
            - titles: title 열의 모든 값을 ' / '로 연결한 문자열
            - descriptions: description 열의 모든 값을 ' / '로 연결한 문자열
            - sector: 섹터 정보 (없으면 빈 문자열)
    """
    try:
        df = pd.read_csv(csv_path)

        # symbol과 search_date로 필터링
        df_filtered = df[(df['symbol'] == symbol) & (df['search_date'] == search_date)]

        # 데이터가 없는 경우 처리
        if len(df_filtered) == 0:
            return {
                'symbol': symbol,
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
    except Exception as e:
        print(f"데이터 포맷팅 오류 (symbol={symbol}, date={search_date}): {e}")
        return {
            'symbol': symbol,
            'search_date': search_date,
            'titles': "",
            'descriptions': "",
            'sector': ""
        }

# --- 에이전트 설계 ---
class BaseModelAgent(ABC):
    @abstractmethod
    def predict(self, stock_data: str) -> AgentPrediction:
        pass

class OpenAIAgent(BaseModelAgent):
    """OpenRouter를 통해 API를 사용하는 표준 에이전트"""
    # 기본값 설정: 에러 발생 시 반환할 안전한 값 
    DEFAULT_PREDICTION: AgentPrediction = {
        "day_1": {"decision": "buy", "confidence": 50, "reason": "API error or parsing failed."},
        "day_3": {"decision": "buy", "confidence": 50, "reason": "API error or parsing failed."},
        "day_7": {"decision": "buy", "confidence": 50, "reason": "API error or parsing failed."},
        "day_15": {"decision": "buy", "confidence": 50, "reason": "API error or parsing failed."},
        "day_30": {"decision": "buy", "confidence": 50, "reason": "API error or parsing failed."}
    }

    def __init__(self, model_name: str, client: OpenAI, site_url: str, app_name: str, persona_profile: Optional[str] = None):
        self.model_name = model_name
        self.client = client
        self.site_url = site_url
        self.app_name = app_name
        self.persona_profile = persona_profile

    def predict(self, stock_data: str) -> AgentPrediction:
        prompt = create_stock_prompt(stock_data, self.persona_profile)
        effort_level = "high" if self.model_name == 'gpt-5' else "low"
        
        try:
            response = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": self.site_url,
                    "X-Title": self.app_name,
                },
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ],
                        "effort" : effort_level
                    }
                ],
            )
            
            content = response.choices[0].message.content
            result: Dict[str, Any] = json.loads(content)
            
            # 다중 기간 예측 결과 파싱
            prediction = {}
            for period in ["day_1", "day_3", "day_7", "day_15", "day_30"]:
                if period in result:
                    period_data = result[period]
                    prediction[period] = {
                        "decision": period_data.get("decision", "buy"),
                        "confidence": int(period_data.get("confidence", 50)),
                        "reason": period_data.get("reason", "")
                    }
                else:
                    prediction[period] = {"decision": "buy", "confidence": 50, "reason": "Missing period data"}
            
            return AgentPrediction(**prediction)
        except Exception as e:
            print(f"Error during prediction for model '{self.model_name}': {e}")
            return self.DEFAULT_PREDICTION

# --- 예측 시스템 ---
class StockPredictor:
    def __init__(self, expert_agent: OpenAIAgent, crowd_agents: List[OpenAIAgent]):
        self.expert_agent = expert_agent
        self.crowd_agents = crowd_agents

    def predict(self, stock_data: str) -> Dict:
        # 전문가 예측은 단일 요청이므로 그냥 실행
        expert_result = self.expert_agent.predict(stock_data)

        # crowd_agents 병렬 실행
        crowd_results = []
        with ThreadPoolExecutor(max_workers=len(self.crowd_agents)) as executor:
            futures = [executor.submit(agent.predict, stock_data) for agent in self.crowd_agents]
            for future in as_completed(futures):
                try:
                    res = future.result()
                    crowd_results.append(res)
                except Exception as e:
                    print(f"Error in crowd agent: {e}")

        # --- 각 기간별 집계 ---
        periods = ["day_1", "day_3", "day_7", "day_15", "day_30"]
        aggregated_results = {}
        
        for period in periods:
            buy_cnt = sell_cnt = 0
            buy_sum = sell_sum = 0
            
            # 전문가 예측 포함
            expert_decision = expert_result[period]["decision"].strip().lower()
            expert_conf = expert_result[period]["confidence"]
            if expert_decision == "buy":
                buy_sum += expert_conf
                buy_cnt += 1
            elif expert_decision == "sell":
                sell_sum += expert_conf
                sell_cnt += 1
            
            # 크라우드 예측 집계
            for res in crowd_results:
                decision = res[period]["decision"].strip().lower()
                confidence = res[period]["confidence"]
                if decision == "buy":
                    buy_sum += confidence
                    buy_cnt += 1
                elif decision == "sell":
                    sell_sum += confidence
                    sell_cnt += 1

            # 최종 결정
            if buy_sum > sell_sum:
                final_decision = "buy"
                avg_confidence = buy_sum / buy_cnt if buy_cnt else 0
            else:
                final_decision = "sell"
                avg_confidence = sell_sum / sell_cnt if sell_cnt else 0

            aggregated_results[period] = {
                "buy_confidence_sum": buy_sum,
                "sell_confidence_sum": sell_sum,
                "average_confidence": avg_confidence,
                "final_decision": final_decision
            }

        return {
            "expert_prediction": expert_result,
            "crowd_predictions": crowd_results,
            "aggregated_results": aggregated_results
        }


if __name__ == "__main__":
    import csv

    # ========== 설정 구간 (사용자가 수정할 부분) ==========
    CSV_PATH = "data/articles/csv/news_with_market_cap_20250924_135112.csv"
    # FILTER_SYMBOLS = None  # None이면 전체, 특정 종목만: ['ADBE', 'AAPL']
    # FILTER_DATES = None    # None이면 전체, 특정 날짜만: ['2024-03-14']

    # 테스트: 상위 3개 이벤트만 실행
    FILTER_SYMBOLS = ['ADBE']
    FILTER_DATES = ['2024-03-14', '2024-06-13', '2024-12-11']

    EXPERT_MODEL = "gpt-5"
    CROWD_MODEL = "gpt-5-nano"
    NUM_CROWD_AGENTS = 3

    PERSONA_FILE_PATH = "data/persona/json/persona_profiles_1.json"

    YOUR_SITE_URL = "http://localhost:8000"
    YOUR_APP_NAME = "AI Stock Predictor"
    # ===================================================

    # --- 1. OpenRouter API 클라이언트 초기화 ---
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    # --- 2. 에이전트 초기화 ---
    expert_agent = OpenAIAgent(model_name=EXPERT_MODEL, client=client, site_url=YOUR_SITE_URL, app_name=YOUR_APP_NAME)

    # 페르소나 데이터 로딩
    persona_data = load_persona_data(PERSONA_FILE_PATH)

    # 랜덤하게 페르소나 선택
    selected_personas = select_random_personas(persona_data, NUM_CROWD_AGENTS)

    # crowd agents 생성 (각각 다른 페르소나 적용)
    crowd_agents = []
    for i, persona in enumerate(selected_personas):
        persona_profile = format_persona_profile(persona)
        agent = OpenAIAgent(
            model_name=CROWD_MODEL,
            client=client,
            site_url=YOUR_SITE_URL,
            app_name=YOUR_APP_NAME,
            persona_profile=persona_profile
        )
        crowd_agents.append(agent)
        print(f"Crowd Agent {i+1} 페르소나: {persona.get('PROMPT_ID', 'Unknown')} - {persona.get('PERSONA_PROFILE', {}).get('Identity', {}).get('Nationality', 'Unknown')}")

    predictor = StockPredictor(expert_agent, crowd_agents)

    # --- 3. CSV에서 이벤트 목록 로드 ---
    events = load_events_from_csv(CSV_PATH, symbols=FILTER_SYMBOLS, dates=FILTER_DATES)

    if not events:
        print("처리할 이벤트가 없습니다. CSV 경로와 필터 조건을 확인하세요.")
        exit(1)

    print(f"\n총 {len(events)}개의 이벤트를 처리합니다.")
    print(f"이벤트 목록: {events[:5]}{'...' if len(events) > 5 else ''}\n")

    # --- 4. 결과 저장 파일 초기화 ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"prediction_results_{timestamp}.csv"

    fieldnames = ["symbol", "search_date", "prediction_date", "model_type", "model_name", "decision", "confidence", "reason"]

    # CSV 파일 헤더 작성
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    # --- 5. 각 이벤트에 대해 예측 실행 ---
    periods = ["day_1", "day_3", "day_7", "day_15", "day_30"]

    for event_idx, (symbol, search_date) in enumerate(events, 1):
        print(f"\n{'='*60}")
        print(f"[{event_idx}/{len(events)}] 처리 중: {symbol} @ {search_date}")
        print(f"{'='*60}")

        # 데이터 로드 및 포맷팅
        agent_data = convert_df_to_agent_format(CSV_PATH, symbol, search_date)

        # 데이터가 비어있으면 스킵
        if not agent_data['titles'] and not agent_data['descriptions']:
            print(f"  ⚠ 뉴스 데이터가 없습니다. 스킵합니다.")
            continue

        stock_data = json.dumps(agent_data, ensure_ascii=False)

        # 예측 실행
        try:
            result = predictor.predict(stock_data)
        except Exception as e:
            print(f"  ❌ 예측 실패: {e}")
            continue

        # 예측 날짜 계산
        prediction_dates = calculate_prediction_dates(search_date)

        # 결과 출력
        expert_result = result['expert_prediction']
        print(f"\n전문가 의견 ({EXPERT_MODEL}):")
        for period in periods:
            expert_pred = expert_result[period]
            print(f"  {period}: {expert_pred['decision'].upper()} (신뢰도: {expert_pred['confidence']})")

        print(f"\n집계된 최종 결과:")
        for period in periods:
            agg_result = result['aggregated_results'][period]
            print(f"  {period}: {agg_result['final_decision'].upper()} (평균 신뢰도: {agg_result['average_confidence']:.1f})")

        # --- 6. 결과를 CSV에 append ---
        rows = []

        # 전문가 결과
        for period in periods:
            expert_pred = expert_result[period]
            rows.append({
                "symbol": symbol,
                "search_date": search_date,
                "prediction_date": prediction_dates[period],
                "model_type": "expert",
                "model_name": EXPERT_MODEL,
                "decision": expert_pred["decision"],
                "confidence": expert_pred["confidence"],
                "reason": expert_pred.get("reason", "")
            })

        # 크라우드 결과
        for idx, crowd in enumerate(result['crowd_predictions']):
            for period in periods:
                crowd_pred = crowd[period]
                rows.append({
                    "symbol": symbol,
                    "search_date": search_date,
                    "prediction_date": prediction_dates[period],
                    "model_type": f"crowd_{idx+1}",
                    "model_name": CROWD_MODEL,
                    "decision": crowd_pred["decision"],
                    "confidence": crowd_pred["confidence"],
                    "reason": crowd_pred.get("reason", "")
                })

        # 크라우드 평균
        for period in periods:
            # 크라우드 모델들의 buy/sell 가중치 합 계산
            buy_sum = 0
            sell_sum = 0
            for crowd in result['crowd_predictions']:
                decision = crowd[period]["decision"].strip().lower()
                confidence = crowd[period]["confidence"]
                if decision == "buy":
                    buy_sum += confidence
                elif decision == "sell":
                    sell_sum += confidence

            # 최종 결정: 가중치 합이 큰 쪽
            crowd_final_decision = "buy" if buy_sum > sell_sum else "sell"

            # 평균 신뢰도 계산
            crowd_confidences = [crowd[period]["confidence"] for crowd in result['crowd_predictions']]
            crowd_avg_confidence = sum(crowd_confidences) / len(crowd_confidences)

            rows.append({
                "symbol": symbol,
                "search_date": search_date,
                "prediction_date": prediction_dates[period],
                "model_type": "crowd_average",
                "model_name": "crowd_models",
                "decision": crowd_final_decision,
                "confidence": round(crowd_avg_confidence, 1),
                "reason": f"Aggregated decision from {len(crowd_confidences)} crowd models (buy_sum={buy_sum}, sell_sum={sell_sum})"
            })

        # CSV에 추가 (append 모드)
        with open(output_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerows(rows)

        print(f"  ✅ 결과가 {output_file}에 저장되었습니다.")

    print(f"\n{'='*60}")
    print(f"전체 실험 완료! 총 {len(events)}개 이벤트 처리됨.")
    print(f"결과 파일: {output_file}")
    print(f"{'='*60}")