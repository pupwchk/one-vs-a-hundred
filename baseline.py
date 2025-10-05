import json
import os
from abc import ABC, abstractmethod
from typing import List, Dict, TypedDict, Optional, Any 
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

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

# --- 프롬프트 생성 함수 ---
def create_stock_prompt(stock_data: str) -> str:
    """주식 예측을 위한 표준 프롬프트를 생성합니다."""
    # 다중 기간 매수/매도 결정을 위한 프롬프트
    return (
        "You are a stock analyst. Based on the provided data, decide whether to 'buy' or 'sell' the stock for different time periods: 1 day, 3 days, 7 days, 15 days, and 30 days from the data date. "
        "For each period, provide your confidence level as a single integer from 0 to 100 and a brief reason for your decision. "
        "Your response MUST be a JSON object with five keys: 'day_1', 'day_3', 'day_7', 'day_15', 'day_30'. "
        "Each period should have 'decision' (buy/sell), 'confidence' (0-100), and 'reason' (brief explanation).\n\n"
        "Example: {\"day_1\": {\"decision\": \"buy\", \"confidence\": 75, \"reason\": \"Strong Q4 results\"}, \"day_3\": {\"decision\": \"sell\", \"confidence\": 60, \"reason\": \"Market volatility\"}, ...}\n\n"
        "Respond ONLY with a valid JSON object. Do not include any extra text.\n\n"
        f"Data: {stock_data}"
    )

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

    def __init__(self, model_name: str, client: OpenAI, site_url: str, app_name: str):
        self.model_name = model_name
        self.client = client
        self.site_url = site_url
        self.app_name = app_name

    def predict(self, stock_data: str) -> AgentPrediction:
        prompt = create_stock_prompt(stock_data)
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
    # --- 1. OpenRouter API 클라이언트 초기화 (원본 유지) ---
    YOUR_SITE_URL = "http://localhost:8000"  # 로컬 테스트 URL 또는 실제 앱 주소
    YOUR_APP_NAME = "AI Stock Predictor"      # 프로젝트 이름

    client = OpenAI(
        base_url = "https://openrouter.ai/api/v1",
        api_key = os.getenv("OPENROUTER_API_KEY"),
    )

    EXPERT_MODEL = "gpt-5"
    expert_agent = OpenAIAgent(model_name=EXPERT_MODEL, client=client, site_url=YOUR_SITE_URL, app_name=YOUR_APP_NAME)
    
    CROWD_MODEL = "gpt-5-nano"
    NUM_CROWD_AGENTS = 3
    crowd_agents = [
        OpenAIAgent(model_name=CROWD_MODEL, client=client, site_url=YOUR_SITE_URL, app_name=YOUR_APP_NAME) for _ in range(NUM_CROWD_AGENTS)
    ]

    # --- 2. 예측 시스템 초기화 및 데이터 정의 (원본 유지) ---
    predictor = StockPredictor(expert_agent, crowd_agents)
    
    # stock_data 정의
    stock_data_dict = {'descriptions': 'Adobe’s (NASDAQ:ADBE) Q4 Sales Beat Estimates But Full-Year '
                 'Sales Guidance Misses Expectations Significantly  Yahoo '
                 'Finance / Adobe Stock Drops as Revenue Forecast Disappoints  '
                 'Investopedia / Adobe Reports Record Q4 and Fiscal 2024 '
                 'Revenue  Business Wire / Adobe (ADBE) Results Seen as '
                 '‘Make-or-Break’ to Prove AI Credentials  Bloomberg',
                 'search_date': '2024-12-11',
                 'sector': 'Information Technology',
                 'titles': 'Adobe’s (NASDAQ:ADBE) Q4 Sales Beat Estimates But Full-Year Sales '
                 'Guidance Misses Expectations Significantly - Yahoo Finance / Adobe '
                 'Stock Drops as Revenue Forecast Disappoints - Investopedia / Adobe '
                 'Reports Record Q4 and Fiscal 2024 Revenue - Business Wire / Adobe '
                 '(ADBE) Results Seen as ‘Make-or-Break’ to Prove AI Credentials - '
                 'Bloomberg'}
    
    stock_data = json.dumps(stock_data_dict, ensure_ascii=False)
    
    # --- 3. 예측 시스템 실행 ---
    result = predictor.predict(stock_data)
    
    # --- 주식 심볼 추출 ---
    stock_symbol = extract_stock_symbol(stock_data)
    
    # --- 예측 날짜 계산 ---
    prediction_dates = calculate_prediction_dates(stock_data_dict["search_date"])

    # --- 4. 결과 출력 ---
    expert_result = result['expert_prediction']
    
    print(f"--- 최종 예측 결과 (주식: {stock_symbol}) ---")
    print(f"전문가 의견 ({EXPERT_MODEL}):")
    
    periods = ["day_1", "day_3", "day_7", "day_15", "day_30"]
    for period in periods:
        expert_pred = expert_result[period]
        print(f"  {period}: {expert_pred['decision'].upper()} (신뢰도: {expert_pred['confidence']}) - {expert_pred['reason']}")
    
    print("-" * 50)
    
    print(f"참고: 소형 모델({CROWD_MODEL}) 결과 ({NUM_CROWD_AGENTS}개 샘플):")
    for idx, crowd in enumerate(result['crowd_predictions']):
        print(f"소형 모델 {idx+1}:")
        for period in periods:
            crowd_pred = crowd[period]
            print(f"  {period}: {crowd_pred['decision'].upper()} (신뢰도: {crowd_pred['confidence']})")

    print("-" * 50)
    print("집계된 최종 결과:")
    for period in periods:
        agg_result = result['aggregated_results'][period]
        print(f"  {period}: {agg_result['final_decision'].upper()} (평균 신뢰도: {agg_result['average_confidence']:.1f})")

    # --- 예측 결과를 CSV로 저장 ---
    import csv
    rows = []
    
    # 전문가 결과 (각 기간별로)
    for period in periods:
        expert_pred = expert_result[period]
        rows.append({
            "symbol": stock_symbol,
            "search_date": stock_data_dict["search_date"],
            "prediction_date": prediction_dates[period],
            "model_type": "expert",
            "model_name": EXPERT_MODEL,
            "decision": expert_pred["decision"],
            "confidence": expert_pred["confidence"],
            "reason": expert_pred.get("reason", "")
        })
    
    # 소형 모델 결과 (각 기간별로)
    for idx, crowd in enumerate(result['crowd_predictions']):
        for period in periods:
            crowd_pred = crowd[period]
            rows.append({
                "symbol": stock_symbol,
                "search_date": stock_data_dict["search_date"],
                "prediction_date": prediction_dates[period],
                "model_type": f"crowd_{idx+1}",
                "model_name": CROWD_MODEL,
                "decision": crowd_pred["decision"],
                "confidence": crowd_pred["confidence"],
                "reason": crowd_pred.get("reason", "")
            })
    
    # 집계 결과도 추가
    for period in periods:
        agg_result = result['aggregated_results'][period]
        rows.append({
            "symbol": stock_symbol,
            "search_date": stock_data_dict["search_date"],
            "prediction_date": prediction_dates[period],
            "model_type": "aggregated",
            "model_name": "ensemble",
            "decision": agg_result["final_decision"],
            "confidence": round(agg_result["average_confidence"], 1),
            "reason": f"Buy: {agg_result['buy_confidence_sum']}, Sell: {agg_result['sell_confidence_sum']}"
        })
    
    # 예측 날짜별 crowd agent들만의 평균 신뢰도 추가
    for period in periods:
        # 해당 기간의 크라우드 모델들만의 신뢰도 수집
        crowd_confidences = []
        
        # 크라우드 모델 신뢰도만 수집
        for crowd in result['crowd_predictions']:
            crowd_conf = crowd[period]["confidence"]
            crowd_confidences.append(crowd_conf)
        
        # 크라우드 모델들만의 평균 계산
        crowd_avg_confidence = sum(crowd_confidences) / len(crowd_confidences)
        
        rows.append({
            "symbol": stock_symbol,
            "search_date": stock_data_dict["search_date"],
            "prediction_date": prediction_dates[period],
            "model_type": "crowd_average",
            "model_name": "crowd_models",
            "decision": "N/A",
            "confidence": round(crowd_avg_confidence, 1),
            "reason": f"Average confidence across {len(crowd_confidences)} crowd models"
        })
    
    with open("prediction_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["symbol", "search_date", "prediction_date", "model_type", "model_name", "decision", "confidence", "reason"])
        writer.writeheader()
        writer.writerows(rows)