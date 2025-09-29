import os
from abc import ABC, abstractmethod
from typing import List, Dict, TypedDict
from openai import OpenAI
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드 (API 저장 변수 = OPENROUTER_API_KEY)
load_dotenv()

# --- 응답 형식 정의 ---
class AgentPrediction(TypedDict):
    decision: str # 살지 말지
    confidence: int # 신뢰도

# --- 프롬프트 생성 함수 ---
def create_stock_prompt(stock_data: str) -> str:
    """주식 예측을 위한 표준 프롬프트를 생성합니다."""
    return (
        "You are a stock analyst. Based on the provided data, decide whether to 'buy' or 'hold' for tomorrow. "
        "Also, provide your confidence level as a single integer from 0 to 100. "
        "Your response MUST be a JSON object with two keys: 'decision' and 'confidence'.\n\n"
        "Example: {\"decision\": \"buy\", \"confidence\": 87}\n\n"
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
    # 기댓값은 평균으로 설정하도록 수정 필요할듯 (오류 개수 카운트 하고, 해당 라벨의 기댓값 평균으로 설정)
    DEFAULT_PREDICTION: AgentPrediction = {"decision": "hold", "confidence": 50}

    def __init__(self, model_name: str, client: OpenAI):
        self.model_name = model_name
        self.client = client

    def predict(self, stock_data: str) -> AgentPrediction:
        prompt = create_stock_prompt(stock_data)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}, # JSON 응답 모드 활성화
                max_tokens=50,
                temperature=0, # ??
            )
            # JSON 파싱
            import json
            result = json.loads(response.choices[0].message.content)
            return AgentPrediction(decision=result["decision"], confidence=result["confidence"])
        except Exception as e:
            # OpenRouter 사용 시 모델명을 함께 출력
            print(f"Error during prediction for model '{self.model_name}': {e}")
            return self.DEFAULT_PREDICTION

# --- 예측 시스템 ---
class StockPredictor:
    def __init__(self, expert_agent: OpenAIAgent, crowd_agents: List[OpenAIAgent]):
        self.expert_agent = expert_agent
        self.crowd_agents = crowd_agents

    def predict(self, stock_data: str) -> Dict:
        expert_result = self.expert_agent.predict(stock_data)
        crowd_results = [agent.predict(stock_data) for agent in self.crowd_agents]

        buy_sum = sum(res["confidence"] for res in crowd_results if res["decision"] == "buy")
        hold_sum = sum(res["confidence"] for res in crowd_results if res["decision"] == "hold")

        final_decision = "buy" if buy_sum > hold_sum else "hold"

        return {
            "expert_prediction": expert_result,
            "crowd_predictions": crowd_results,
            "buy_confidence_sum": buy_sum,
            "hold_confidence_sum": hold_sum,
            "final_decision": final_decision
        }


if __name__ == "__main__":
    # --- 1. OpenRouter API 클라이언트 초기화 (가이드라인 적용) ---
    # 사이트 URL과 앱 이름을 헤더에 추가하여 OpenRouter 대시보드에서 요청을 식별합니다.
    YOUR_SITE_URL = "http://localhost:8000"  # 로컬 테스트 URL 또는 실제 앱 주소
    YOUR_APP_NAME = "AI Stock Predictor"      # 프로젝트 이름

    client = OpenAI(
        base_url = "https://openrouter.ai/api/v1",
        api_key = os.getenv("OPENROUTER_API_KEY"),
        default_headers = {                   
            "HTTP-Referer": YOUR_SITE_URL,
            "X-Title": YOUR_APP_NAME,
        },
    )

    
    # 전문가 에이전트 - GPT-5
    EXPERT_MODEL = "openai/gpt-5" # 2024년 9월 30일 cutoff -> 4: 2024년 6월 30일
    expert_agent = OpenAIAgent(model_name=EXPERT_MODEL, client=client)

    # 대중 에이전트 - GPT-5-nano
    CROWD_MODEL = "openai/gpt-5-nano"
    NUM_CROWD_AGENTS = 3 # 대중 에이전트 수 설정

    crowd_agents = [
        OpenAIAgent(model_name=CROWD_MODEL, client=client) for _ in range(NUM_CROWD_AGENTS)
    ]

    # --- 3. 예측 시스템 실행 ---
    predictor = StockPredictor(expert_agent, crowd_agents)
    stock_data = "문자열 or .csv" # 이벤트 텍스트 데이터 + S&P100 개요 + S&P100에 대한 주가 데이터
    # 일반 텍스트로 주자.
    
    result = predictor.predict(stock_data)

    print(f"--- 최종 예측 결과 ---")
    print(f"전문가 의견 ({expert_agent.model_name}): {result['expert_prediction']}")
    print(f"매수 신뢰도 총합: {result['buy_confidence_sum']}")
    print(f"보류 신뢰도 총합: {result['hold_confidence_sum']}")
    print(f"최종 결정: {result['final_decision'].upper()}")
    print("-" * 25)
    print(f"참고: 소형 모델({CROWD_MODEL}) 결과 (5개 샘플): {result['crowd_predictions'][:5]}")