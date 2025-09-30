import json
import os
from abc import ABC, abstractmethod
from typing import List, Dict, TypedDict, Optional, Any 
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# .env 파일에서 환경 변수 로드 (API 저장 변수 = OPENROUTER_API_KEY)
load_dotenv()

# --- 응답 형식 정의 ---
class AgentPrediction(TypedDict):
    decision: str # 살지 말지
    confidence: int # 신뢰도
    reason: Optional[str] # API 응답에 포함되는 reason 필드 처리를 위해 추가

# --- 프롬프트 생성 함수 ---
def create_stock_prompt(stock_data: str) -> str:
    """주식 예측을 위한 표준 프롬프트를 생성합니다."""
    # reason을 포함하도록 프롬프트 지침을 명확히 강화
    return (
        "You are a stock analyst. Based on the provided data, decide whether to 'buy' or 'sell'. "
        "Also, provide your confidence level as a single integer from 0 to 100 with the reason of your choice in one sentence. "
        "Your response MUST be a JSON object with two keys: 'decision' and 'confidence'. The reason should be included in the response JSON as an additional key 'reason'.\n\n"
        "Example: {\"decision\": \"buy\", \"confidence\": 87, \"reason\": \"answering the reason in text\"}\n\n"
        "Respond ONLY with a valid JSON object. Do not include any extra text.\n\n"
        f"Data: {stock_data}"
    )

# --- 에이전트 설계 ---
class BaseModelAgent(ABC):
    @abstractmethod
    def predict(self, stock_data: str) -> AgentPrediction:
        pass

class OpenAIAgent(BaseModelAgent):
    """OpenRouter를 통해 API를 사용하는 표준 에이전트"""
    # 기본값 설정: 에러 발생 시 반환할 안전한 값 
    DEFAULT_PREDICTION: AgentPrediction = {"decision": "hold", "confidence": 50, "reason": "API error or parsing failed."}

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
            
            return AgentPrediction(
                decision=result["decision"],
                confidence=int(result["confidence"]),
                **({"reason": result["reason"]} if "reason" in result else {})
            )
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

        # --- 집계 ---
        buy_cnt = sell_cnt = 0
        buy_sum = sell_sum = 0
        for res in crowd_results:
            decision = res["decision"].strip().lower()
            if decision == "buy":
                buy_sum += res["confidence"]
                buy_cnt += 1
            elif decision == "sell":
                sell_sum += res["confidence"]
                sell_cnt += 1

        if buy_sum > sell_sum:
            final_decision = "buy"
            avg_confidence = buy_sum / buy_cnt if buy_cnt else 0
        else:
            final_decision = "sell"
            avg_confidence = sell_sum / sell_cnt if sell_cnt else 0

        return {
            "expert_prediction": expert_result,
            "crowd_predictions": crowd_results,
            "buy_confidence_sum": buy_sum,
            "sell_confidence_sum": sell_sum,
            "average_confience": avg_confidence,
            "final_decision": final_decision
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

    # --- 4. 결과 출력 ---
    expert_result = result['expert_prediction']
    
    # 전문가의 매수/매도 신뢰도 분리
    expert_buy_conf = expert_result["confidence"] if expert_result["decision"].strip().lower() == "buy" else 0
    expert_sell_conf = expert_result["confidence"] if expert_result["decision"].strip().lower() == "sell" else 0

    print(f"--- 최종 예측 결과 ---")
    print(f"전문가 의견 ({EXPERT_MODEL}): {expert_result}")
    
    # 요청하신 전문가 신뢰도 분리 출력
    print(f"매수 신뢰도: {expert_buy_conf}")
    print(f"매도 신뢰도: {expert_sell_conf}")
    print(f"최종 결정: {expert_result['decision'].upper()}")
    print("-" * 25)
    
    print(f"참고: 소형 모델({CROWD_MODEL}) 결과 ({NUM_CROWD_AGENTS}개 샘플):")
    for idx, crowd in enumerate(result['crowd_predictions']):
        # 소형 모델 결과 출력 (reason 내용 대신 '...'을 포함한 형식)
        reason_display = crowd.get('reason', '...')
        print(f"소형 모델 {idx+1} 결과: {{'decision': '{crowd['decision']}', 'confidence': {crowd['confidence']}, 'reason': {reason_display}}}")

    # 소형 모델 집단의 총합 신뢰도 출력
    print(f"매수 신뢰도 총합: {result['buy_confidence_sum']}")
    print(f"매도 신뢰도 총합: {result['sell_confidence_sum']}")
    print(f"평균 신뢰도: {result['average_confience']}")
    print(f"최종 결정: {result['final_decision'].upper()}")

    # --- 예측 결과를 CSV로 저장 ---
    import csv
    rows = []
    # 전문가 결과
    rows.append({
        "model_type": "expert",
        "model_name": EXPERT_MODEL,
        "decision": expert_result["decision"],
        "confidence": expert_result["confidence"],
        "reason": expert_result.get("reason", ""),
        "search_date": stock_data_dict["search_date"]
    })
    # 소형 모델 결과
    for idx, crowd in enumerate(result['crowd_predictions']):
        rows.append({
            "model_type": f"crowd_{idx+1}",
            "model_name": CROWD_MODEL,
            "decision": crowd["decision"],
            "confidence": crowd["confidence"],
            "reason": crowd.get("reason", ""),
            "search_date": stock_data_dict["search_date"]
        })
    with open("prediction_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model_type", "model_name", "decision", "confidence", "reason", "search_date"])
        writer.writeheader()
        writer.writerows(rows)