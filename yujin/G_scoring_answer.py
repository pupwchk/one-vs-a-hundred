import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

# -----------------------------
# 1. 예측 결과 불러오기
# -----------------------------
df_predict = pd.read_csv('../data/answer/prediction_results.csv')

df_predict['search_date'] = pd.to_datetime(df_predict['search_date'])
df_predict['prediction_date'] = pd.to_datetime(df_predict['prediction_date'])

# 예측 기간 계산 (days_diff)
df_predict['days_diff'] = (df_predict['prediction_date'] - df_predict['search_date']).dt.days

# 거래일 오프셋 설정 (미국 연방 공휴일 반영)
us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())

# 거래일 기준으로 prediction_date 보정
df_predict['prediction_date_adj'] = df_predict.apply(
    lambda row: row['search_date'] + us_bd * row['days_diff'], axis=1
)

# prediction_date_adj를 datetime으로 변환
df_predict['prediction_date_adj'] = pd.to_datetime(df_predict['prediction_date_adj'])

# 보정 여부 및 보정 일수 계산
df_predict['보정여부'] = (df_predict['prediction_date_adj'] - df_predict['prediction_date']).dt.days
df_predict['보정여부'] = df_predict['보정여부'].fillna(0).astype(int)

# 보정 후 prediction_date 사용
df_predict['prediction_date'] = df_predict['prediction_date_adj']
df_predict.drop(columns=['prediction_date_adj'], inplace=True)

# decision을 수치로 매핑
df_predict['predicted_sign'] = df_predict['decision'].map({'buy': 1, 'sell': -1})

# -----------------------------
# 2. 실제 주가 데이터 수집 (모든 심볼)
# -----------------------------
def get_price_data(symbol: str, start_date, end_date):
    """심볼별 주가 데이터 조회"""
    try:
        ticker = yf.Ticker(symbol)
        # 충분한 기간의 데이터 확보 (시작일 7일 전부터 종료일 7일 후까지)
        start_adj = pd.to_datetime(start_date) - pd.Timedelta(days=7)
        end_adj = pd.to_datetime(end_date) + pd.Timedelta(days=7)
        df_symbol = ticker.history(start=start_adj, end=end_adj)

        if len(df_symbol) == 0:
            print(f"경고: {symbol} 데이터 없음 ({start_date} ~ {end_date})")
            return pd.DataFrame()

        df_symbol = df_symbol[['Close']].reset_index()
        df_symbol['Date'] = pd.to_datetime(df_symbol['Date']).dt.tz_localize(None)
        df_symbol['symbol'] = symbol
        return df_symbol
    except Exception as e:
        print(f"오류: {symbol} 조회 실패 - {e}")
        return pd.DataFrame()

# 모든 심볼의 주가 데이터 수집
symbols = df_predict['symbol'].unique()
min_date = df_predict['search_date'].min()
max_date = df_predict['prediction_date'].max()

print(f"주가 데이터 수집 중: {len(symbols)}개 심볼 ({min_date.date()} ~ {max_date.date()})")

price_data_list = []
for symbol in symbols:
    df_symbol = get_price_data(symbol, min_date, max_date)
    if len(df_symbol) > 0:
        price_data_list.append(df_symbol)
    print(f"  - {symbol}: {len(df_symbol)}개 데이터")

if len(price_data_list) == 0:
    raise ValueError("주가 데이터를 조회할 수 없습니다.")

df_all_prices = pd.concat(price_data_list, ignore_index=True)

# -----------------------------
# 3. search_date / prediction_date 가격 매칭 (보정된 날짜 기준)
# -----------------------------
# search_date 가격 매칭
df_search = df_all_prices[['symbol', 'Date', 'Close']].rename(
    columns={'Date': 'search_date', 'Close': 'Close_search'}
)
df_merge = df_predict.merge(df_search, on=['symbol', 'search_date'], how='left')

# prediction_date 가격 매칭 (보정된 날짜)
df_pred = df_all_prices[['symbol', 'Date', 'Close']].rename(
    columns={'Date': 'prediction_date', 'Close': 'Close_pred'}
)
df_merge = df_merge.merge(df_pred, on=['symbol', 'prediction_date'], how='left')

# -----------------------------
# 4. NaN 처리 및 데이터 검증
# -----------------------------
# 가격 매칭 실패한 행 확인
missing_search = df_merge['Close_search'].isna().sum()
missing_pred = df_merge['Close_pred'].isna().sum()

if missing_search > 0:
    print(f"경고: search_date 가격 매칭 실패 {missing_search}건")
if missing_pred > 0:
    print(f"경고: prediction_date 가격 매칭 실패 {missing_pred}건")

# NaN이 있는 행 제거
df_merge = df_merge.dropna(subset=['Close_search', 'Close_pred'])
print(f"유효한 데이터: {len(df_merge)}건")

# -----------------------------
# 5. 실제 등락 계산
# -----------------------------
df_merge['price_change'] = df_merge['Close_pred'] - df_merge['Close_search']
df_merge['actual_sign'] = np.sign(df_merge['price_change'])

# -----------------------------
# 6. 예측 일치 여부 & 가중 점수
# -----------------------------
df_merge['correct'] = np.where(df_merge['predicted_sign'] == df_merge['actual_sign'], 1, 0)
df_merge['weighted_score'] = np.where(df_merge['correct'] == 1, df_merge['confidence'], -df_merge['confidence'])

# -----------------------------
# 7. Crowd 집단 통합 (투표 방식)
# -----------------------------
def aggregate_crowd_predictions(df):
    crowd_df = df[df['model_type'].str.startswith('crowd', na=False)].copy()
    if len(crowd_df) == 0:
        return pd.DataFrame()

    grouped = crowd_df.groupby(['symbol', 'search_date', 'prediction_date', 'days_diff'])
    crowd_results = []

    for name, group in grouped:
        buy_weight = group[group['decision'] == 'buy']['confidence'].sum()
        sell_weight = group[group['decision'] == 'sell']['confidence'].sum()

        crowd_decision = 'buy' if buy_weight > sell_weight else 'sell'
        crowd_confidence = max(buy_weight, sell_weight) / (buy_weight + sell_weight) * 100 if (buy_weight + sell_weight) > 0 else 50

        actual_sign = group['actual_sign'].iloc[0]
        predicted_sign = 1 if crowd_decision == 'buy' else -1
        is_correct = 1 if predicted_sign == actual_sign else 0
        weighted_score = crowd_confidence if is_correct else -crowd_confidence

        crowd_results.append({
            'symbol': name[0],
            'search_date': name[1],
            'prediction_date': name[2],
            'days_diff': name[3],
            'model_type': 'crowd_aggregate',
            'decision': crowd_decision,
            'confidence': crowd_confidence,
            'predicted_sign': predicted_sign,
            'actual_sign': actual_sign,
            'correct': is_correct,
            'weighted_score': weighted_score,
            'crowd_count': len(group)
        })
    return pd.DataFrame(crowd_results)

df_crowd_agg = aggregate_crowd_predictions(df_merge)
df_expert = df_merge[df_merge['model_type'] == 'expert'].copy()

# -----------------------------
# 8. 평가 1: 단순 정답 유무 채점표
# -----------------------------
print("="*80)
print("[평가 1] 단순 정답 유무 채점표")
print("="*80)

if len(df_expert) > 0:
    print(f"\nExpert 정답률: {df_expert['correct'].mean():.2%} ({df_expert['correct'].sum()}/{len(df_expert)})")
if len(df_crowd_agg) > 0:
    print(f"Crowd 집단지성 정답률: {df_crowd_agg['correct'].mean():.2%} ({df_crowd_agg['correct'].sum()}/{len(df_crowd_agg)})")

# -----------------------------
# 9. 평가 2: 예측일별 집단지성 vs 전문가
# -----------------------------
print("\n" + "="*80)
print("[평가 2] 예측일별 집단지성 vs 전문가 비교")
print("="*80)

prediction_periods = sorted(df_merge['days_diff'].unique())
comparison_results = []

for days in prediction_periods:
    expert_subset = df_expert[df_expert['days_diff'] == days]
    crowd_subset = df_crowd_agg[df_crowd_agg['days_diff'] == days]

    expert_acc = expert_subset['correct'].mean() if len(expert_subset) > 0 else 0
    expert_weighted = expert_subset['weighted_score'].mean() if len(expert_subset) > 0 else 0
    crowd_acc = crowd_subset['correct'].mean() if len(crowd_subset) > 0 else 0
    crowd_weighted = crowd_subset['weighted_score'].mean() if len(crowd_subset) > 0 else 0

    print(f"\n{days}일 후 예측:")
    print(f"  Expert   - 정답률: {expert_acc:.2%}, 가중점수: {expert_weighted:+.2f}")
    print(f"  Crowd    - 정답률: {crowd_acc:.2%}, 가중점수: {crowd_weighted:+.2f}")
    winner = 'Crowd' if crowd_acc > expert_acc else 'Expert' if expert_acc > crowd_acc else '동점'
    print(f"  승자     - {winner}")

    comparison_results.append({
        'days_after': days,
        'expert_accuracy': expert_acc,
        'expert_weighted_score': expert_weighted,
        'crowd_accuracy': crowd_acc,
        'crowd_weighted_score': crowd_weighted,
        'winner_by_accuracy': 'crowd' if crowd_acc > expert_acc else 'expert' if expert_acc > crowd_acc else 'tie'
    })

# -----------------------------
# 10. 평가 3: 종합 점수
# -----------------------------
print("\n" + "="*80)
print("[평가 3] 종합 점수")
print("="*80)

expert_total_acc = df_expert['correct'].mean() if len(df_expert) > 0 else 0
expert_total_weighted = df_expert['weighted_score'].mean() if len(df_expert) > 0 else 0
crowd_total_acc = df_crowd_agg['correct'].mean() if len(df_crowd_agg) > 0 else 0
crowd_total_weighted = df_crowd_agg['weighted_score'].mean() if len(df_crowd_agg) > 0 else 0

print(f"\nExpert 종합")
print(f"  정답률: {expert_total_acc:.2%}")
print(f"  평균 가중점수: {expert_total_weighted:+.2f}")

print(f"\nCrowd 집단지성 종합")
print(f"  정답률: {crowd_total_acc:.2%}")
print(f"  평균 가중점수: {crowd_total_weighted:+.2f}")

print(f"\n최종 승자: ", end="")
if crowd_total_acc > expert_total_acc:
    print("Crowd 집단지성")
elif expert_total_acc > crowd_total_acc:
    print("Expert 전문가")
else:
    print("동점")

# -----------------------------
# 11. 결과 저장
# -----------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

output_file = f"../data/answer/scoring_results_detailed_{timestamp}.csv"
df_merge.to_csv(output_file, index=False, encoding='utf-8')
print(f"\n상세 결과 저장: {output_file}")

if len(df_crowd_agg) > 0:
    crowd_output = f"../data/answer/scoring_results_crowd_{timestamp}.csv"
    df_crowd_agg.to_csv(crowd_output, index=False, encoding='utf-8')
    print(f"Crowd 집단 결과 저장: {crowd_output}")

comparison_df = pd.DataFrame(comparison_results)
comparison_output = f"../data/answer/scoring_comparison_{timestamp}.csv"
comparison_df.to_csv(comparison_output, index=False, encoding='utf-8')
print(f"예측일별 비교 결과 저장: {comparison_output}")

summary_data = {
    'metric': ['정답률', '평균 가중점수'],
    'expert': [expert_total_acc, expert_total_weighted],
    'crowd': [crowd_total_acc, crowd_total_weighted]
}
summary_df = pd.DataFrame(summary_data)
summary_output = f"../data/answer/scoring_summary_{timestamp}.csv"
summary_df.to_csv(summary_output, index=False, encoding='utf-8')
print(f"종합 요약 저장: {summary_output}")

print("="*80)