import pandas as pd
import numpy as np

def generate_100_personas(input_file="ESS11.csv", n_samples=100, output_file="Persona_Data_.csv"):
    """
    ESS 데이터를 불러와 지정된 규칙에 따라 변수를 처리하고 100개의 페르소나를 추출합니다.
    (교육 수준 4단계 세분화 적용)
    """
    print("--- 1. Starting Persona Data Processing ---")
    
    # 1. 사용할 ESS 원본 변수 목록 정의
    # hincfel은 stfinc를 대체하는 가계 재정 만족도 변수입니다.
    raw_persona_cols = [
        'cntry', 'agea', 'gndr', 'eisced', 'isco08', 'hinctnta', 
        'polintr', 'lrscale', 'vote', 'ppltrst', 'trstprl', 
        'stfeco', 'hincfel', 'nwspol', 'netustm', 'happy'
    ]
    
    # 2. 데이터 로드 및 결측치 처리
    try:
        df = pd.read_csv(input_file, low_memory=False)
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return
    
    df_persona = df[raw_persona_cols].copy()
    
    # ESS 결측치 코드 정의 및 NaN으로 대체
    missing_values = [6, 7, 8, 9, 66, 77, 88, 99, 666, 777, 888, 999, 6666, 7777, 8888, 9999]
    df_persona.replace(missing_values, np.nan, inplace=True)
    
    # NaN이 있는 행 제거
    df_persona_clean = df_persona.dropna()
    
    if len(df_persona_clean) < n_samples:
        print(f"Warning: Only {len(df_persona_clean)} clean rows available.")
        n_samples = len(df_persona_clean)
        
    # 3. 100개 무작위 샘플 추출
    df_personas_100 = df_persona_clean.sample(n=n_samples).reset_index(drop=True)

    # 4. 지정된 규칙에 따라 변수 재범주화 및 변환
    
    # B. 2값 매핑 (Binary Mapping): gndr, vote
    df_personas_100['Gender'] = df_personas_100['gndr'].map({1.0: 'Male', 2.0: 'Female'})
    df_personas_100['Vote_Status'] = df_personas_100['vote'].map({1.0: 'Yes', 2.0: 'No'})
    
    # C. 학력 (eisced: 0~8) -> 4단계 세분화 적용 (요청 사항 반영)
    def recode_education(code):
        if code <= 2: return 'Low (Primary/Lower Secondary)'        # ISCED 0-2 (초/중등)
        elif code <= 4: return 'Upper Secondary/Vocational'         # ISCED 3-4 (고등/전문)
        elif code <= 6: return "Tertiary/Bachelor's Degree"         # ISCED 5-6 (전문대/학사)
        else: return "Postgraduate (Master's/Doctorate)"           # ISCED 7-8 (석사/박사)
    df_personas_100['Education_Level'] = df_personas_100['eisced'].apply(recode_education)

    # D. 직업 (isco08) -> 코드-직업명 사용 (상위 1자리 기준)
    isco_map = {
        1: '1-Managers', 2: '2-Professionals', 3: '3-Technicians and Associate Professionals',
        4: '4-Clerical Support Workers', 5: '5-Service and Sales Workers', 
        6: '6-Skilled Agricultural, Forestry and Fishery Workers', 
        7: '7-Craft and Related Trades Workers', 8: '8-Plant and Machine Operators, and Assemblers', 
        9: '9-Elementary Occupations', 0: '0-Armed Forces Occupations'
    }
    df_personas_100['Occupation_Code_Name'] = (
        df_personas_100['isco08'].astype(int) // 1000
    ).map(isco_map).fillna('9-Elementary Occupations')

    # E. 3값 범주화: polintr, ppltrst, trstprl
    
    def recode_3_level_trust(score): # Trust/Interest: Low (0-3), Medium (4-6), High (7-10)
        if score <= 3: return 'Low'
        elif score <= 6: return 'Medium'
        else: return 'High'
    
    # Political Interest: 1=Not at all interested to 4=Very interested
    df_personas_100['Pol_Interest_Level'] = df_personas_100['polintr'].map({1.0: 'Low', 2.0: 'Middle', 3.0: 'Middle', 4.0: 'High'}).fillna('Middle')
    df_personas_100['Interpersonal_Trust'] = df_personas_100['ppltrst'].apply(recode_3_level_trust)
    df_personas_100['Institutional_Trust'] = df_personas_100['trstprl'].apply(recode_3_level_trust)
    
    # F. 경제 만족도 (stfeco, hincfel) -> Pessimistic / Middle / Optimistic
    def recode_econ_sat(score): # stfeco (0-10)
        if score <= 3: return 'Pessimistic'
        elif score <= 6: return 'Middle'
        else: return 'Optimistic'
    
    # hincfel (1-5)
    def recode_hincfel(score):
        if score <= 2: return 'Optimistic' # 1, 2: Living comfortably/Doing alright
        elif score == 3: return 'Middle' # 3: Coping
        else: return 'Pessimistic' # 4, 5: Finding it difficult/very difficult
        
    df_personas_100['National_Econ_Sat'] = df_personas_100['stfeco'].apply(recode_econ_sat)
    df_personas_100['Household_Fin_Sat'] = df_personas_100['hincfel'].apply(recode_hincfel)

    # G. 시간 변수 범주화 (Quantile-based Categorization): nwspol, netustm
    
    # nwspol (정치 뉴스): Often, Middle, Rarely (분위수 기준)
    q_nwspol_low = df_personas_100['nwspol'].quantile(0.33)
    q_nwspol_high = df_personas_100['nwspol'].quantile(0.66)
    
    def recode_nwspol(minutes):
        if minutes <= q_nwspol_low: return 'Rarely'
        elif minutes <= q_nwspol_high: return 'Middle'
        else: return 'Often'
    df_personas_100['Pol_News_Freq'] = df_personas_100['nwspol'].apply(recode_nwspol)
    
    # netustm (인터넷 사용): Often, Rarely (분위수 기준)
    q_netustm_low = df_personas_100['netustm'].quantile(0.50)
    
    def recode_netustm(minutes):
        if minutes <= q_netustm_low: return 'Rarely'
        else: return 'Often'
    df_personas_100['Internet_Use_Freq'] = df_personas_100['netustm'].apply(recode_netustm)

    # 5. 최종 CSV 파일 생성
    final_cols = [
        'cntry', 'agea', 'Gender', 'Education_Level', 'Occupation_Code_Name', 'hinctnta', 
        'Pol_Interest_Level', 'lrscale', 'Vote_Status', 'Interpersonal_Trust', 
        'Institutional_Trust', 'National_Econ_Sat', 'Household_Fin_Sat', 'Pol_News_Freq', 
        'Internet_Use_Freq', 'happy'
    ]
    df_personas_final = df_personas_100[final_cols].copy()
    
    df_personas_final.to_csv(output_file, index=False)
    
    print(f"\n✅ Successfully generated {len(df_personas_final)} personas and saved to {output_file}")
    print("\n--- Example of Generated Data (4-Level Education Applied) ---")
    print(df_personas_final.head())
    
    return df_personas_final

# --- 메인 실행 ---
if __name__ == "__main__":
    generate_100_personas()