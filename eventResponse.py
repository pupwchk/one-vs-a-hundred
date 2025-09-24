# ============================================================
# LLM 실험 템플릿 (최종 결과 요약 버전 / Colab용)
# ============================================================

import json
import time
import requests
import textwrap
try:
    from getpass import getpass
except ImportError:
    getpass = input

# --- 상수 정의 ---
BUY = "산다"
SELL = "안산다"


class LLMExperiment:
    """LLM 투자 결정 실험을 관리하는 클래스."""

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
    # 공통 rule
    SYSTEM_PROMPT = textwrap.dedent("""
        당신은 주식 이벤트 해석을 돕는 분석가입니다.
        아래 사용자 이벤트를 읽고 '산다' 또는 '안산다' 중 하나로 결론을 내리세요.
        그리고 결론에 대한 신뢰도를 0.0~1.0 사이 숫자로, 이유를 1~2문장으로 설명하세요.
        출력은 JSON으로만 하세요.
        형식: {"decision": "산다" 또는 "안산다", "confidence": 0.0~1.0, "rationale": "짧은 이유"}
        주의: JSON 이외의 텍스트를 출력하지 마십시오.
    """)

    def __init__(self, api_key: str, high_model_cfg: dict, low_models_cfg: list):
        self.api_key = api_key
        self.high_model_cfg = high_model_cfg
        self.low_models_cfg = low_models_cfg
        self.results = {}

    def _call_api(self, model_cfg: dict, event_text: str, retries: int = 1) -> dict:
        headers = { "Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json" }
        user_prompt = f"이벤트 설명:\n{event_text}"
        payload = {
            "model": model_cfg["name"], "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": self.SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
            "temperature": model_cfg["temperature"], "top_p": model_cfg["top_p"],
            "max_tokens": model_cfg["max_tokens"], "seed": model_cfg.get("seed"),
        }
        for _ in range(retries + 1):
            try:
                resp = requests.post(self.BASE_URL, headers=headers, json=payload, timeout=60)
                resp.raise_for_status()
                parsed = json.loads(resp.json()["choices"][0]["message"]["content"])
                decision = parsed.get("decision", SELL).strip()
                decision = decision if decision in (BUY, SELL) else SELL
                confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
                return {"decision": decision, "confidence": confidence}
            except (requests.RequestException, json.JSONDecodeError):
                time.sleep(2)
        return {"decision": SELL, "confidence": 0.0}

    @staticmethod
    def _aggregate_weighted(items: list[dict]) -> str:
        sums = {BUY: 0.0, SELL: 0.0}
        for item in items:
            if item["decision"] in sums: sums[item["decision"]] += item["confidence"]
        return BUY if sums[BUY] > sums[SELL] else SELL

    def run(self, event_text: str):
        high_res = self._call_api(self.high_model_cfg, event_text)
        low_res_list = [self._call_api(cfg, event_text) for cfg in self.low_models_cfg]
        aggregated_res = self._aggregate_weighted(low_res_list)
        self.results = {
            "event_text": event_text, "high_performance": high_res,
            "low_performance": low_res_list, "low_performance_aggregated": aggregated_res
        }
        return self

    def display_results(self):
        """요청하신 간략한 형태로 결과를 출력합니다."""
        if not self.results: return

        print("\n" + "="*50)
        # 1. 입력 이벤트 출력
        print("📌 [입력 이벤트]")
        print(textwrap.indent(self.results['event_text'], '  '))

        # 2. 고성능 모델 응답 출력
        high = self.results['high_performance']
        print("\n🤖 [고성능 모델 응답]")
        print(f"  - 최종 결론: {high['decision']}")

        # 3. 저성능 모델 요약 출력
        low_results = self.results['low_performance']
        buy_count = sum(1 for r in low_results if r['decision'] == BUY)
        sell_count = sum(1 for r in low_results if r['decision'] == SELL)
        agg_final = self.results['low_performance_aggregated']

        print("\n⚙️ [저성능 모델 100개 요약]")
        print(f"  - '산다' 의견: {buy_count}개")
        print(f"  - '안산다' 의견: {sell_count}개")
        print(f"  - 신뢰도 가중 최종 결론: {agg_final}")
        print("="*50)


# ============================================================
# 아래에만 입력

# 1. 하이퍼파라미터 설정
HIGH_MODEL_CFG = {
    "name": "openai/gpt-4o",
    "temperature": 0.2, "top_p": 0.95, "max_tokens": 256, "seed": 42
}
LOW_MODELS_CFG = [
    {"name": "google/gemma-7b-it", "temperature": round(0.6 + (i % 4) * 0.1, 1),
     "top_p": 0.95, "max_tokens": 256, "seed": i + 1} for i in range(100)
]

# 2. API 키 및 이벤트 텍스트 입력
OPENROUTER_API_KEY = getpass("OpenRouter API Key 입력: ")
EVENT_TEXT = """**Election 2024: Presidential results**
Donald Trump became America’s 47th president after mounting the most momentous comeback in political history...
"""

# 3. 실험 실행 및 결과 출력
if OPENROUTER_API_KEY:
    experiment = LLMExperiment(
        api_key=OPENROUTER_API_KEY,
        high_model_cfg=HIGH_MODEL_CFG,
        low_models_cfg=LOW_MODELS_CFG
    )
    experiment.run(EVENT_TEXT).display_results()
else:
    print("API 키가 입력되지 않았습니다.")
