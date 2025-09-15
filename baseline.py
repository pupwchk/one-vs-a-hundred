import os
from abc import ABC, abstractmethod
from typing import List, Dict, TypedDict
from openai import OpenAI
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드 (OPENAI_API_KEY="sk-...")
load_dotenv()

# --- 응답 형식 정의 ---
class AgentPrediction(TypedDict):
    decision: str
    confidence: int

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
    """모든 에이전트의 기본 설계도 (추상 클래스)"""
    @abstractmethod
    def predict(self, stock_data: str) -> AgentPrediction:
        pass

class OpenAIAgent(BaseModelAgent):
    """OpenAI API를 사용하는 표준 에이전트"""
    # 기본값 설정: 에러 발생 시 반환할 안전한 값
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
                temperature=0.5,
            )
            # JSON 파싱
            import json
            result = json.loads(response.choices[0].message.content)
            return AgentPrediction(decision=result["decision"], confidence=result["confidence"])
        except Exception as e:
            print(f"Error during prediction for model {self.model_name}: {e}")
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
    # 1. API 클라이언트 초기화
    # .env 파일의 OPENAI_API_KEY를 자동으로 읽어옵니다.
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # 2. 에이전트 생성
    # 전문가 에이전트 (고성능 모델 사용)
    expert_agent = OpenAIAgent(model_name="gpt-5", client=client)

    # 대중 에이전트 100명 (경량 모델 사용)
    crowd_agents = [
        OpenAIAgent(model_name="gpt-5-nano", client=client) for _ in range(100)
    ]

    # 3. 예측 시스템 실행
    predictor = StockPredictor(expert_agent, crowd_agents)
    stock_data = "제공할 데이터를 넣어주세요"
    
    result = predictor.predict(stock_data)

    print(f"--- 최종 예측 결과 ---")
    print(f"전문가 의견: {result['expert_prediction']}")
    print(f"매수 신뢰도 총합: {result['buy_confidence_sum']}")
    print(f"보류 신뢰도 총합: {result['hold_confidence_sum']}")
    print(f"최종 결정: {result['final_decision'].upper()}")
    print("-" * 25)
    print(f"참고: 소형 모델 결과 (5개 샘플): {result['crowd_predictions'][:5]}")