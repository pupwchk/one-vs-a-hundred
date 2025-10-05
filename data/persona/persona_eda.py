import pandas as pd

# 1. 파일 로드 (파일명이 'Persona_Data.csv'라고 가정합니다.)
# 인코딩 문제가 발생하면 encoding='utf-8' 대신 encoding='cp949' 또는 'euc-kr'을 시도해 보세요.
try:
    df = pd.read_csv("Persona_Data.csv")
except FileNotFoundError:
    print("오류: 'Persona_Data.csv' 파일을 찾을 수 없습니다. 파일 경로와 이름을 확인해 주세요.")
    exit()

# 2. 연령대 (Age Group) 처리
# 20-30대 (<40), 40-50대 (40-59), 60대 이상 (>=60)으로 분류
bins = [0, 40, 60, float('inf')]
labels = ['20-30대', '40-50대', '60대 이상']
# right=False 설정으로 40세와 60세가 각각 '40-50대', '60대 이상'에 포함되도록 합니다.
df['Age_Group'] = pd.cut(df['agea'], bins=bins, labels=labels, right=False, include_lowest=True)

# 3. 교육 수준 (Education Level) 단순화 함수
def simplify_education(edu):
    """'Education_Level' 컬럼의 값을 한국식 교육 수준으로 단순화합니다."""
    if 'Primary' in edu or 'Lower Secondary' in edu:
        return '초등학교/중학교 졸업 이하'
    elif 'Secondary' in edu or 'Vocational' in edu:
        return '고등학교/직업 학교 졸업'
    elif 'Tertiary' in edu or 'Master' in edu or 'Doctorate' in edu:
        return '대학교/대학원 졸업 이상'
    else:
        return '기타'

df['Education_Simplified'] = df['Education_Level'].apply(simplify_education)


# 4. 분포 계산 및 포맷팅 함수
def calculate_distribution(series, title):
    """주어진 시리즈의 값별 백분율 분포를 계산하고 포맷팅하여 반환합니다."""
    # 값별 개수를 백분율로 계산 (normalize=True)
    distribution = series.value_counts(normalize=True) * 100
    
    # 결과를 출력용 문자열로 포맷팅
    output = f"## {title}\n"
    # 소수점 첫째 자리까지 표시하고, 결과를 조합합니다.
    output += "\n".join([f"* {index}: {value:.1f}%" for index, value in distribution.items()])
    return output

# 5. 각 항목별 분포 계산 및 출력
print("### 데이터 분석 결과 ###")
print("-" * 30)

age_dist = calculate_distribution(df['Age_Group'], "연령대 (Age Group)")
print(age_dist)

print("-" * 30)

edu_dist = calculate_distribution(df['Education_Simplified'], "교육 수준 (Education Level)")
print(edu_dist)

print("-" * 30)

trust_dist = calculate_distribution(df['Interpersonal_Trust'], "대인 신뢰 수준 (Interpersonal Trust)")
print(trust_dist)

print("-" * 30)

fin_sat_dist = calculate_distribution(df['Household_Fin_Sat'], "가계 재정 만족도 (Household Financial Satisfaction)")
print(fin_sat_dist)

print("-" * 30)