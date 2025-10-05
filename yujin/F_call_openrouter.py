'''사용하지 않는 스크립트'''

from openai import OpenAI
from dotenv import load_dotenv
import os
import json
from E_data_form_making import convert_df_to_agent_format

load_dotenv()

def create_expert_client():
    """전문가 모델(GPT-5) 클라이언트 생성"""
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        default_headers={
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Stock Analysis Expert"
        }
    )

def create_nano_client(client_id):
    """소형 모델(GPT-5-nano) 클라이언트 생성"""
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        default_headers={
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": f"Stock Analysis Nano Agent {client_id}"
        }
    )

def get_expert_prediction(client, data):
    """전문가 모델의 심층 분석 예측"""
    expert_prompt = f"""
    You are a senior financial analyst with extensive experience in stock market analysis.

    Analyze the provided data thoroughly and make an investment decision based on comprehensive reasoning.
    Consider market context, sector trends, news sentiment, and potential impacts.

    Data to analyze: {data}

    Provide your response as a JSON object with this exact structure:
    {{
        "decision": "BUY" or "SELL",
        "confidence": <integer 0-100>,
        "reason": "<detailed 2-3 sentence explanation of your reasoning>"
    }}

    Focus on providing deep analytical insights and thorough reasoning for your decision.
    """

    try:
        # GPT-5: 깊은 추론을 위한 high reasoning effort
        response = client.chat.completions.create(
            model="openai/gpt-5",
            messages=[{"role": "user", "content": expert_prompt}],
            reasoning_effort="high",  # 깊은 추론
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Expert model error: {e}")
        return {"decision": "SELL", "confidence": 50, "reason": "Analysis failed due to technical error"}

def get_nano_prediction(client, data, agent_id):
    """소형 모델의 빠른 분류 예측"""
    nano_prompt = f"""
    Quick stock analysis task. Based on the news data, decide BUY or SELL.

    Data: {data}

    Response format (JSON only):
    {{
        "decision": "BUY" or "SELL",
        "confidence": <integer 0-100>,
        "reason": "<brief one-sentence reason>"
    }}
    """

    try:
        # GPT-5-nano: 최소 추론을 위한 minimal reasoning effort
        response = client.chat.completions.create(
            model="openai/gpt-5-nano",
            messages=[{"role": "user", "content": nano_prompt}],
            reasoning_effort="minimal",  # 최소 추론
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Nano model {agent_id} error: {e}")
        return {"decision": "SELL", "confidence": 25, "reason": "Quick analysis suggests caution"}

def aggregate_nano_results(nano_results):
    """소형 모델들의 결과를 신뢰도 기반으로 집계"""
    buy_confidence = sum(r["confidence"] for r in nano_results if r["decision"] == "BUY")
    sell_confidence = sum(r["confidence"] for r in nano_results if r["decision"] == "SELL")

    final_decision = "BUY" if buy_confidence > sell_confidence else "SELL"

    return {
        "aggregated_decision": final_decision,
        "buy_confidence_sum": buy_confidence,
        "sell_confidence_sum": sell_confidence,
        "individual_results": nano_results
    }

def test_with_default_input():
    """기본 입력값으로 테스트 실행"""
    # 기본 테스트 데이터
    default_data = {
        'symbol': 'AAPL',
        'search_date': '2024-12-11',
        'titles': 'Apple Reports Strong Q4 Earnings Beat / Apple Announces New AI Features / Tech Stocks Rally on Market Optimism',
        'descriptions': 'Apple exceeded Q4 earnings expectations with strong iPhone sales / Apple unveils advanced AI capabilities in latest software update / Technology sector sees broad gains as investors show confidence',
        'sector': 'Technology'
    }

    # 클라이언트 생성
    expert_client = create_expert_client()
    nano_clients = [create_nano_client(i+1) for i in range(3)]

    print("=== 기본 입력값 테스트: 1 Expert vs 3 Nano Models ===\n")
    print(f"테스트 데이터: {default_data['symbol']} ({default_data['sector']})")
    print(f"뉴스 제목들: {default_data['titles'][:100]}...\n")

    # 전문가 예측 (깊은 추론)
    print("🧠 Expert Analysis (GPT-5 with HIGH reasoning effort):")
    expert_result = get_expert_prediction(expert_client, default_data)
    print(f"Decision: {expert_result['decision']}")
    print(f"Confidence: {expert_result['confidence']}%")
    print(f"Reasoning: {expert_result['reason']}\n")

    # 소형 모델 예측들 (최소 추론)
    print("⚡ Nano Models Analysis (GPT-5-nano with MINIMAL reasoning effort):")
    nano_results = []
    for i, client in enumerate(nano_clients):
        result = get_nano_prediction(client, default_data, i+1)
        nano_results.append(result)
        print(f"Nano Agent {i+1}: {result['decision']} (Confidence: {result['confidence']}%) - {result['reason']}")

    # 집계 결과
    aggregated = aggregate_nano_results(nano_results)
    print(f"\n📊 Aggregated Nano Results:")
    print(f"Final Decision: {aggregated['aggregated_decision']}")
    print(f"BUY Confidence Sum: {aggregated['buy_confidence_sum']}")
    print(f"SELL Confidence Sum: {aggregated['sell_confidence_sum']}")

    # 최종 비교
    print(f"\n🏆 Final Comparison:")
    print(f"Expert (Deep Reasoning): {expert_result['decision']} ({expert_result['confidence']}%)")
    print(f"Crowd (Minimal Reasoning): {aggregated['aggregated_decision']} (Total confidence difference: {abs(aggregated['buy_confidence_sum'] - aggregated['sell_confidence_sum'])})")

    agreement = "✅ AGREE" if expert_result['decision'] == aggregated['aggregated_decision'] else "❌ DISAGREE"
    print(f"Expert vs Crowd: {agreement}")

    return expert_result, aggregated

def main():
    """메인 실행 함수"""
    # 데이터 로드
    stock_data = convert_df_to_agent_format()

    # 클라이언트 생성
    expert_client = create_expert_client()
    nano_clients = [create_nano_client(i+1) for i in range(3)]

    print("=== 1 vs 3 LLM Stock Analysis ===\n")

    # 전문가 예측 (1개)
    print("🧠 Expert Analysis (GPT-5 with high reasoning):")
    expert_result = get_expert_prediction(expert_client, stock_data)
    print(f"Decision: {expert_result['decision']}")
    print(f"Confidence: {expert_result['confidence']}%")
    print(f"Reasoning: {expert_result['reason']}\n")

    # 소형 모델 예측들 (3개)
    print("⚡ Nano Models Analysis (GPT-5-nano with minimal reasoning):")
    nano_results = []
    for i, client in enumerate(nano_clients):
        result = get_nano_prediction(client, stock_data, i+1)
        nano_results.append(result)
        print(f"Nano Agent {i+1}: {result['decision']} (Confidence: {result['confidence']}%)")

    # 집계 결과
    aggregated = aggregate_nano_results(nano_results)
    print(f"\n📊 Aggregated Nano Results:")
    print(f"Final Decision: {aggregated['aggregated_decision']}")
    print(f"BUY Confidence Sum: {aggregated['buy_confidence_sum']}")
    print(f"SELL Confidence Sum: {aggregated['sell_confidence_sum']}")

    # 최종 비교
    print(f"\n🏆 Final Comparison:")
    print(f"Expert: {expert_result['decision']} ({expert_result['confidence']}%)")
    print(f"Crowd: {aggregated['aggregated_decision']} (Total confidence difference: {abs(aggregated['buy_confidence_sum'] - aggregated['sell_confidence_sum'])})")

    agreement = "✅ AGREE" if expert_result['decision'] == aggregated['aggregated_decision'] else "❌ DISAGREE"
    print(f"Expert vs Crowd: {agreement}")

if __name__ == "__main__":
    # 기본 입력값으로 테스트 실행
    print("기본 입력값 테스트를 실행합니다...\n")
    test_with_default_input()

    print("\n" + "="*60 + "\n")

    # 원래 메인 함수도 실행 (실제 데이터 사용)
    print("실제 데이터로 분석을 실행합니다...\n")
    main()