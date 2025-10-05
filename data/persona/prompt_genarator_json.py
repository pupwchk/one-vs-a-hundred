import pandas as pd
import json
import numpy as np

# --- 1. 국가 코드 매핑 정의 ---
COUNTRY_MAP = {
    'AL': 'Albania', 'AT': 'Austria', 'BE': 'Belgium', 'BG': 'Bulgaria', 'CH': 'Switzerland',
    'CY': 'Cyprus', 'CZ': 'Czechia', 'DE': 'Germany', 'DK': 'Denmark', 'EE': 'Estonia',
    'ES': 'Spain', 'FI': 'Finland', 'FR': 'France', 'GB': 'United Kingdom', 'GE': 'Georgia',
    'GR': 'Greece', 'HR': 'Croatia', 'HU': 'Hungary', 'IE': 'Ireland', 'IS': 'Iceland',
    'IL': 'Israel', 'IT': 'Italy', 'LT': 'Lithuania', 'LU': 'Luxembourg', 'LV': 'Latvia',
    'ME': 'Montenegro', 'MK': 'North Macedonia', 'NL': 'Netherlands', 'NO': 'Norway',
    'PL': 'Poland', 'PT': 'Portugal', 'RO': 'Romania', 'RS': 'Serbia', 'RU': 'Russian Federation',
    'SE': 'Sweden', 'SI': 'Slovenia', 'SK': 'Slovakia', 'TR': 'Turkey', 'UA': 'Ukraine',
    'XK': 'Kosovo'
}

# --- 2. LLM 출력 형식 및 프롬프트 생성 함수 정의 ---

def create_persona_profile_json(persona_data: dict) -> dict:
    """
    단일 페르소나의 상세 정보를 담은 JSON 객체를 생성합니다.
    """
    
    # 정수형 변환 및 국가명 매핑
    def clean_value(key, value):
        if key == 'cntry': return COUNTRY_MAP.get(value, value) # 국가 코드 -> 국가명 변환
        if key in ['agea', 'hinctnta', 'lrscale', 'happy'] and not pd.isna(value):
            return int(value)
        return value

    # 페르소나 프로필 구조 정의
    persona_profile = {
        # PROMPT_ID는 인덱스를 사용하여 생성
        "PROMPT_ID": f"P{persona_data.name}", 
        "PERSONA_PROFILE": {
            "Identity": {
                "Nationality": clean_value('cntry', persona_data['cntry']),
                "Age": f"{clean_value('agea', persona_data['agea'])} years old",
                "Gender": persona_data['Gender'],
                "Education": persona_data['Education_Level']
            },
            "Socio_Economics": {
                "Occupation": persona_data['Occupation_Code_Name'],
                "Income_Decile": f"{clean_value('hinctnta', persona_data['hinctnta'])} (1=Lowest, 10=Highest)",
                "Left_Right_Position": f"{clean_value('lrscale', persona_data['lrscale'])} (0=Left, 10=Right)"
            },
            "Outlook_and_Trust": {
                "Overall_Happiness": f"{clean_value('happy', persona_data['happy'])}/10",
                "Natl_Econ_View": persona_data['National_Econ_Sat'],
                "House_Fin_View": persona_data['Household_Fin_Sat'],
                "Inst_Trust": persona_data['Institutional_Trust'],
                "Interpersonal_Trust": persona_data['Interpersonal_Trust']
            },
            "Information_Habit": {
                "Political_Interest": persona_data['Pol_Interest_Level'],
                "News_Frequency": persona_data['Pol_News_Freq'],
                "Internet_Frequency": persona_data['Internet_Use_Freq'],
                "Voter": persona_data['Vote_Status']
            }
        }
    }
    
    return persona_profile

def generate_persona_json_file(persona_file="Persona_Data.csv", output_json_file="persona_profiles.json"):
    
    print("--- 2. Generating Final Persona JSON File ---")

    # 1. 페르소나 데이터 로드 (CSV 파일에서)
    try:
        df_personas = pd.read_csv(persona_file)
        df_personas.index.name = 'Persona_ID'
    except FileNotFoundError:
        print(f"Error: {persona_file} 파일을 찾을 수 없습니다. 페르소나 생성 코드를 먼저 실행하세요.")
        return

    all_profiles_list = []

    # 2. 프로필 반복 생성 루프
    for persona_id, persona_data in df_personas.iterrows():
        # 단일 프로필 JSON 객체 생성
        json_profile_object = create_persona_profile_json(persona_data)
        all_profiles_list.append(json_profile_object)

    # 3. JSON 파일로 저장
    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(all_profiles_list, f, indent=4, ensure_ascii=False)

    print(f"\n✅ 성공적으로 총 {len(all_profiles_list)}개의 페르소나 프로필이 {output_json_file} 파일에 JSON 형식으로 저장되었습니다.")

# --- 메인 실행 ---
if __name__ == "__main__":
    
    # 2단계: 생성된 CSV를 기반으로 JSON 프로필 파일을 생성합니다.
    generate_persona_json_file()