import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Windows에서 'NanumGothic' 폰트 경로 설정 (Windows 기본 폰트 폴더)
font_path = 'C:\\Windows\\Fonts\\malgun.ttf'  # 예: 'malgun.ttf'나 'NanumGothic.ttf'로 변경할 수 있음
font_prop = fm.FontProperties(fname=font_path)

# matplotlib에 폰트 설정
plt.rcParams['font.family'] = font_prop.get_name()

# 제목을 가운데로 정렬
st.markdown("<h1 style='text-align: center;'>MonoGuard DashBoard</h1>", unsafe_allow_html=True)


# 탭 생성
tab1, tab2,tab3 = st.tabs([
    "외부 환경",
    "AGV",
    "OHT"
])

agv_data = joblib.load('data/agv_dataframe.pkl')
oht_data = joblib.load('data/oht_dataframe.pkl')

with tab1:

    # 데이터 인덱스 설정
    agv_data['index'] = range(len(agv_data))
    oht_data['index'] = range(len(oht_data))

    # 시각화할 범위 설정
    agv_data = agv_data.iloc[-100:]
    oht_data = oht_data.iloc[-100:]
    
    st.subheader("최근 외부 환경")
    
    latest_ex_temperature_agv = int(agv_data['ex_temperature'].iloc[-1])
    latest_ex_himidity_agv = int(agv_data['ex_humidity'].iloc[-1])
    latest_ex_illuminance_agv = int(agv_data['ex_illuminance'].iloc[-1])
    
    latest_ex_temperature_oht = int(oht_data['ex_temperature'].iloc[-1])
    latest_ex_humidity_oht = int(oht_data['ex_humidity'].iloc[-1])
    latest_ex_illuminance_oht = int(oht_data['ex_illuminance'].iloc[-1])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"최근 AGV 외부 온도: {latest_ex_temperature_agv} °C")
        st.success(f"최근 AGV 외부 습도: {latest_ex_himidity_agv} %")
        st.success(f"최근 AGV 외부 조도: {latest_ex_illuminance_agv} lux")
    
    with col2:
        st.success(f"최근 OHT 외부 온도: {latest_ex_temperature_oht} °C")
        st.success(f"최근 OHT 외부 습도: {latest_ex_humidity_oht} %")
        st.success(f"최근 OHT 외부 조도: {latest_ex_illuminance_oht} lux")
    
    st.divider()
    
    st.subheader("외부 온도 변화")

    # AGV 외부 온도 변화
    fig, ax = plt.subplots(figsize=(24, 8))
    ax.plot(agv_data['index'], agv_data['ex_temperature'], label="외부 온도", color='blue')
    ax.axhline(y=agv_data['ex_temperature'].mean(), color='blue', linestyle='--', label="평균 외부 온도")
    ax.set_title('AGV 외부 온도 변화', fontsize=18)
    #ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('외부 온도 (°C)', fontsize=14)
    ax.legend()
    st.pyplot(fig)
    
    # OHT 외부 온도 변화
    fig, ax = plt.subplots(figsize=(24, 8))
    ax.plot(oht_data['index'], oht_data['ex_temperature'], label="외부 온도", color='orange')
    ax.axhline(y=oht_data['ex_temperature'].mean(), color='orange', linestyle='--', label="평균 외부 온도")
    ax.set_title('OHT 외부 온도 변화', fontsize=18)
    #ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('외부 온도 (°C)', fontsize=14)
    ax.legend()
    st.pyplot(fig)
    
    st.divider()
    
    st.subheader("외부 습도 변화")
    
    # AGV 외부 습도 변화
    fig, ax = plt.subplots(figsize=(24, 8))
    ax.plot(agv_data['index'], agv_data['ex_humidity'], label="외부 습도", color='blue')
    ax.axhline(y=agv_data['ex_humidity'].mean(), color='blue', linestyle='--', label="평균 외부 습도")
    ax.set_title('AGV 외부 습도 변화', fontsize=18)
    #ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('외부 습도 (%)', fontsize=14)
    ax.legend()
    st.pyplot(fig)
    
    # OHT 외부 습도 변화
    fig, ax = plt.subplots(figsize=(24, 8))
    ax.plot(oht_data['index'], oht_data['ex_humidity'], label="외부 습도", color='orange')
    ax.axhline(y=oht_data['ex_humidity'].mean(), color='orange', linestyle='--', label="평균 외부 습도")
    ax.set_title('OHT 외부 습도 변화', fontsize=18)
    #ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('외부 습도 (%)', fontsize=14)
    ax.legend()
    st.pyplot(fig)
    
    st.divider()
    
    st.subheader("외부 조도 변화")
    
    # AGV 외부 조도 변화
    fig, ax = plt.subplots(figsize=(24, 8))
    ax.plot(agv_data['index'], agv_data['ex_illuminance'], label="외부 조도", color='blue')
    ax.axhline(y=agv_data['ex_illuminance'].mean(), color='blue', linestyle='--', label="평균 외부 조도")
    ax.set_title('AGV 외부 조도 변화', fontsize=18)
    #ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('외부 조도 (lux)', fontsize=14)
    ax.legend()
    st.pyplot(fig)
    
    # OHT 외부 조도 변화
    fig, ax = plt.subplots(figsize=(24, 8))
    ax.plot(oht_data['index'], oht_data['ex_illuminance'], label="외부 조도", color='orange')
    ax.axhline(y=oht_data['ex_illuminance'].mean(), color='orange', linestyle='--', label="평균 외부 조도")
    ax.set_title('OHT 외부 조도 변화', fontsize=18)
    #ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('외부 조도 (lux)', fontsize=14)
    ax.legend()
    st.pyplot(fig)
    
    
    


 

    
    
# 1. AGV 데이터 불러오기
agv_data = joblib.load('data/agv_dataframe.pkl')  # 파일 경로에 맞게 수정
agv_data = agv_data.iloc[-500:]

with tab2:
    
    #데이터가 비어있는지 확인
    if not agv_data.empty:
        # 'index'를 순차적으로 추가 (필요한 경우)
        agv_data['index'] = range(len(agv_data))  # 새로운 인덱스를 추가하여 순차적인 시간 시계열처럼 사용

        # AGV에서 선택할 수 있는 항목
        selected_item = st.selectbox(
            "어떤 데이터를 확인하시겠습니까?",
            ["선택하세요", "미세먼지 농도(PM)", "온도측정값(NTC)", "전류측정값(CT)", "장치 최근 상태"],
            key="agv_selectbox"  #고유한 key를 지정정
        )
        
        
        # 선택된 항목이 "선택하세요"일 경우 아무것도 표시하지 않음
        if selected_item == "선택하세요":
            st.write("")  # 아무것도 출력하지 않음
        
        
        elif selected_item == "미세먼지 농도(PM)":
            # PM1.0, PM2.5, PM10 각각 시각화 (하나씩 나누기)
            fig, ax = plt.subplots(figsize=(24, 8))  
            ax.plot(agv_data['index'], agv_data['PM1.0'], label="PM1.0", color='blue')
            ax.plot(agv_data['index'], agv_data['PM2.5'], label="PM2.5", color='orange')
            ax.plot(agv_data['index'], agv_data['PM10'], label="PM10", color='green')

            # 평균선 추가
            ax.axhline(y=agv_data['PM1.0'].mean(), color='blue', linestyle='--', label="PM1.0 평균")
            ax.axhline(y=agv_data['PM2.5'].mean(), color='orange', linestyle='--', label="PM2.5 평균")
            ax.axhline(y=agv_data['PM10'].mean(), color='green', linestyle='--', label="PM10 평균")

            ax.set_title('미세먼지 농도', fontsize=18)
            #ax.set_xlabel('시간', fontsize=14)
            ax.set_ylabel('농도 (μg/m³)', fontsize=14)
            ax.legend()

            # x축 레이블 간격 조정: 레이블을 500 간격으로 표시
            plt.xticks(range(0, len(agv_data), len(agv_data)//10))

            st.pyplot(fig)

            # 개별 센서 시각화 (하나씩 나누기)
            st.header("개별 센서 그래프")
            
            fig, ax = plt.subplots(figsize=(24, 8))  
            ax.plot(agv_data['index'], agv_data['PM1.0'], label="PM1.0", color='blue')
            ax.axhline(y=agv_data['PM1.0'].mean(), color='blue', linestyle='--', label="PM1.0 평균")
            ax.set_title('PM1.0 농도', fontsize=18)
            #ax.set_xlabel('시간', fontsize=14)
            ax.set_ylabel('농도 (μg/m³)', fontsize=14)
            ax.legend()
            st.pyplot(fig)

            fig, ax = plt.subplots(figsize=(24, 8))  
            ax.plot(agv_data['index'], agv_data['PM2.5'], label="PM2.5", color='orange')
            ax.axhline(y=agv_data['PM2.5'].mean(), color='orange', linestyle='--', label="PM2.5 평균")
            ax.set_title('PM2.5 농도', fontsize=18)
            #ax.set_xlabel('시간', fontsize=14)
            ax.set_ylabel('농도 (μg/m³)', fontsize=14)
            ax.legend()
            st.pyplot(fig)

            fig, ax = plt.subplots(figsize=(24, 8)) 
            ax.plot(agv_data['index'], agv_data['PM10'], label="PM10", color='green')
            ax.axhline(y=agv_data['PM10'].mean(), color='green', linestyle='--', label="PM10 평균")
            ax.set_title('PM10 농도', fontsize=18)
            #ax.set_xlabel('시간', fontsize=14)
            ax.set_ylabel('농도 (μg/m³)', fontsize=14)
            ax.legend()
            
            # x축 레이블 간격 조정: 레이블을 500 간격으로 표시
            plt.xticks(range(0, len(agv_data), len(agv_data)//10))
            
            st.pyplot(fig)
            
        elif selected_item == "온도측정값(NTC)":
            
            # 온도 시각화
            fig, ax = plt.subplots(figsize=(24, 8))
            ax.plot(agv_data['index'], agv_data['NTC'], label="온도 (NTC)", color='red')

            # 평균선 추가
            ax.axhline(y=agv_data['NTC'].mean(), color='red', linestyle='--', label="NTC 평균")

            ax.set_title('온도측정값', fontsize=18)
            #ax.set_xlabel('시간', fontsize=14)
            ax.set_ylabel('온도 (°C)', fontsize=14)
            ax.legend()

            # x축 레이블 간격 조정: 레이블을 500 간격으로 표시
            plt.xticks(range(0, len(agv_data), len(agv_data)//10))

            st.pyplot(fig)
            
            
        elif selected_item == "전류측정값(CT)":
            
            # CT1, CT2, CT3, CT4 각각 시각화 (하나씩 나누기)
            fig, ax = plt.subplots(figsize=(24, 8))  
            ax.plot(agv_data['index'], agv_data['CT1'], label="CT1", color='blue')
            ax.plot(agv_data['index'], agv_data['CT2'], label="CT2", color='orange')
            ax.plot(agv_data['index'], agv_data['CT3'], label="CT3", color='green')
            ax.plot(agv_data['index'], agv_data['CT4'], label="CT4", color='purple')

            # 평균선 추가
            ax.axhline(y=agv_data['CT1'].mean(), color='blue', linestyle='--', label="CT1 평균")
            ax.axhline(y=agv_data['CT2'].mean(), color='orange', linestyle='--', label="CT2 평균")
            ax.axhline(y=agv_data['CT3'].mean(), color='green', linestyle='--', label="CT3 평균")
            ax.axhline(y=agv_data['CT4'].mean(), color='purple', linestyle='--', label="CT4 평균")

            ax.set_title('전류측정값', fontsize=18)
            #ax.set_xlabel('시간', fontsize=14)
            ax.set_ylabel('전류 (A)', fontsize=14)
            ax.legend()

            # x축 레이블 간격 조정: 레이블을 500 간격으로 표시
            plt.xticks(range(0, len(agv_data), len(agv_data)//10))

            st.pyplot(fig)
            
            # 개별 센서 시각화 (하나씩 나누기)
            st.header("개별 센서 그래프")

            # CT1 전류측정값 시각화
            fig, ax = plt.subplots(figsize=(24, 8))  
            ax.plot(agv_data['index'], agv_data['CT1'], label="CT1", color='blue')
            ax.axhline(y=agv_data['CT1'].mean(), color='blue', linestyle='--', label="CT1 평균")
            ax.set_title('CT1 전류측정값', fontsize=18)
            #ax.set_xlabel('시간', fontsize=14)
            ax.set_ylabel('전류 (A)', fontsize=14)
            ax.legend()
            # x축 레이블 간격 조정: 레이블을 500 간격으로 표시
            plt.xticks(range(0, len(agv_data), len(agv_data)//10))
            st.pyplot(fig)

            # CT2 전류측정값 시각화
            fig, ax = plt.subplots(figsize=(24, 8))  
            ax.plot(agv_data['index'], agv_data['CT2'], label="CT2", color='orange')
            ax.axhline(y=agv_data['CT2'].mean(), color='orange', linestyle='--', label="CT2 평균")
            ax.set_title('CT2 전류측정값', fontsize=18)
            #ax.set_xlabel('시간', fontsize=14)
            ax.set_ylabel('전류 (A)', fontsize=14)
            ax.legend()
            # x축 레이블 간격 조정: 레이블을 500 간격으로 표시
            plt.xticks(range(0, len(agv_data), len(agv_data)//10))
            st.pyplot(fig)

            # CT3 전류측정값 시각화
            fig, ax = plt.subplots(figsize=(24, 8))  
            ax.plot(agv_data['index'], agv_data['CT3'], label="CT3", color='green')
            ax.axhline(y=agv_data['CT3'].mean(), color='green', linestyle='--', label="CT3 평균")
            ax.set_title('CT3 전류측정값', fontsize=18)
            #ax.set_xlabel('시간', fontsize=14)
            ax.set_ylabel('전류 (A)', fontsize=14)
            ax.legend()
            # x축 레이블 간격 조정: 레이블을 500 간격으로 표시
            plt.xticks(range(0, len(agv_data), len(agv_data)//10))
            st.pyplot(fig)

            # CT4 전류측정값 시각화
            fig, ax = plt.subplots(figsize=(24, 8))  
            ax.plot(agv_data['index'], agv_data['CT4'], label="CT4", color='purple')
            ax.axhline(y=agv_data['CT4'].mean(), color='purple', linestyle='--', label="CT4 평균")
            ax.set_title('CT4 전류측정값', fontsize=18)
            #ax.set_xlabel('시간', fontsize=14)
            ax.set_ylabel('전류 (A)', fontsize=14)
            ax.legend()
            # x축 레이블 간격 조정: 레이블을 500 간격으로 표시
            plt.xticks(range(0, len(agv_data), len(agv_data)//10))
            st.pyplot(fig)
            
            
            
            
        elif selected_item == "장치 최근 상태":
            # AGV 데이터가 로드되었으면 상태 출력
            if not agv_data.empty:
                latest_state_agv = int(agv_data['state'].iloc[-1])
                st.subheader("AGV 최근 상태")
                if latest_state_agv == 0:
                    st.success("AGV 최근 상태: 정상입니다 ✅")
                elif latest_state_agv == 1:
                    st.warning("AGV 최근 상태: 관심입니다 ⚠️")
                elif latest_state_agv == 2:
                    st.warning("AGV 최근 상태: 경고입니다 ⚠️")
                elif latest_state_agv == 3:
                    st.error("AGV 최근 상태: 위험입니다 ❌")
            else:
                st.error("AGV 데이터가 비어 있습니다 ❌")
            
            
    else:
        st.error("AGV 데이터가 비어 있습니다 ❌")
        


# 2. OHT 데이터 불러오기
oht_data = joblib.load('data/oht_dataframe.pkl')  # 파일 경로에 맞게 수정
oht_data = oht_data.iloc[-500:]

with tab3:
    
    #데이터가 비어있는지 확인
    if not oht_data.empty:
        # 'index'를 순차적으로 추가 (필요한 경우)
        oht_data['index'] = range(len(oht_data))  # 새로운 인덱스를 추가하여 순차적인 시간 시계열처럼 사용

        # OHT에서 선택할 수 있는 항목
        selected_item = st.selectbox(
            "어떤 데이터를 확인하시겠습니까?",
            ["선택하세요", "미세먼지 농도(PM)", "온도측정값(NTC)", "전류측정값(CT)", "장치 최근 상태"]
        )
        
        
        # 선택된 항목이 "선택하세요"일 경우 아무것도 표시하지 않음
        if selected_item == "선택하세요":
            st.write("")  # 아무것도 출력하지 않음
        
        
        elif selected_item == "미세먼지 농도(PM)":
            # PM1.0, PM2.5, PM10 각각 시각화 (하나씩 나누기)
            fig, ax = plt.subplots(figsize=(24, 8))  
            ax.plot(oht_data['index'], oht_data['PM1.0'], label="PM1.0", color='blue')
            ax.plot(oht_data['index'], oht_data['PM2.5'], label="PM2.5", color='orange')
            ax.plot(oht_data['index'], oht_data['PM10'], label="PM10", color='green')

            # 평균선 추가
            ax.axhline(y=oht_data['PM1.0'].mean(), color='blue', linestyle='--', label="PM1.0 평균")
            ax.axhline(y=oht_data['PM2.5'].mean(), color='orange', linestyle='--', label="PM2.5 평균")
            ax.axhline(y=oht_data['PM10'].mean(), color='green', linestyle='--', label="PM10 평균")

            ax.set_title('미세먼지 농도 (시간에 따른 변화)', fontsize=18)
            #ax.set_xlabel('순차적 인덱스 (시간처럼 처리)', fontsize=14)
            ax.set_ylabel('농도 (μg/m³)', fontsize=14)
            ax.legend()

            # x축 레이블 간격 조정: 레이블을 500 간격으로 표시
            plt.xticks(range(0, len(oht_data), len(oht_data)//10))

            st.pyplot(fig)

            # 개별 센서 시각화 (하나씩 나누기)
            st.header("개별 센서 그래프")
            
            fig, ax = plt.subplots(figsize=(24, 8))  
            ax.plot(oht_data['index'], oht_data['PM1.0'], label="PM1.0", color='blue')
            ax.axhline(y=oht_data['PM1.0'].mean(), color='blue', linestyle='--', label="PM1.0 평균")
            ax.set_title('PM1.0 농도', fontsize=18)
            #ax.set_xlabel('순차적 인덱스 (시간처럼 처리)', fontsize=14)
            ax.set_ylabel('농도 (μg/m³)', fontsize=14)
            ax.legend()
            st.pyplot(fig)

            fig, ax = plt.subplots(figsize=(24, 8))  
            ax.plot(oht_data['index'], oht_data['PM2.5'], label="PM2.5", color='orange')
            ax.axhline(y=oht_data['PM2.5'].mean(), color='orange', linestyle='--', label="PM2.5 평균")
            ax.set_title('PM2.5 농도', fontsize=18)
            #ax.set_xlabel('순차적 인덱스 (시간처럼 처리)', fontsize=14)
            ax.set_ylabel('농도 (μg/m³)', fontsize=14)
            ax.legend()
            st.pyplot(fig)

            fig, ax = plt.subplots(figsize=(24, 8)) 
            ax.plot(oht_data['index'], oht_data['PM10'], label="PM10", color='green')
            ax.axhline(y=oht_data['PM10'].mean(), color='green', linestyle='--', label="PM10 평균")
            ax.set_title('PM10 농도', fontsize=18)
            #ax.set_xlabel('순차적 인덱스 (시간처럼 처리)', fontsize=14)
            ax.set_ylabel('농도 (μg/m³)', fontsize=14)
            ax.legend()
            
            # x축 레이블 간격 조정: 레이블을 500 간격으로 표시
            plt.xticks(range(0, len(oht_data), len(oht_data)//10))
            
            st.pyplot(fig)
            
        elif selected_item == "온도측정값(NTC)":
            
            # 온도 시각화
            fig, ax = plt.subplots(figsize=(24, 8))
            ax.plot(oht_data['index'], oht_data['NTC'], label="온도 (NTC)", color='red')

            # 평균선 추가
            ax.axhline(y=oht_data['NTC'].mean(), color='red', linestyle='--', label="NTC 평균")

            ax.set_title('온도측정값 (시간에 따른 변화)', fontsize=18)
            #ax.set_xlabel('순차적 인덱스 (시간처럼 처리)', fontsize=14)
            ax.set_ylabel('온도 (°C)', fontsize=14)
            ax.legend()

            # x축 레이블 간격 조정: 레이블을 500 간격으로 표시
            plt.xticks(range(0, len(oht_data), len(oht_data)//10))

            st.pyplot(fig)
            
        elif selected_item == "전류측정값(CT)":
            
            # CT1, CT2, CT3, CT4 각각 시각화 (하나씩 나누기)
            fig, ax = plt.subplots(figsize=(24, 8))  
            ax.plot(oht_data['index'], oht_data['CT1'], label="CT1", color='blue')
            ax.plot(oht_data['index'], oht_data['CT2'], label="CT2", color='orange')
            ax.plot(oht_data['index'], oht_data['CT3'], label="CT3", color='green')
            ax.plot(oht_data['index'], oht_data['CT4'], label="CT4", color='purple')

            # 평균선 추가
            ax.axhline(y=oht_data['CT1'].mean(), color='blue', linestyle='--', label="CT1 평균")
            ax.axhline(y=oht_data['CT2'].mean(), color='orange', linestyle='--', label="CT2 평균")
            ax.axhline(y=oht_data['CT3'].mean(), color='green', linestyle='--', label="CT3 평균")
            ax.axhline(y=oht_data['CT4'].mean(), color='purple', linestyle='--', label="CT4 평균")

            ax.set_title('전류측정값 (시간에 따른 변화)', fontsize=18)
            #ax.set_xlabel('순차적 인덱스 (시간처럼 처리)', fontsize=14)
            ax.set_ylabel('전류 (A)', fontsize=14)
            ax.legend()

            # x축 레이블 간격 조정: 레이블을 500 간격으로 표시
            plt.xticks(range(0, len(oht_data), len(oht_data)//10))

            st.pyplot(fig)
            
            # 개별 센서 시각화 (하나씩 나누기)
            st.header("개별 센서 그래프")

            # CT1 전류측정값 시각화
            fig, ax = plt.subplots(figsize=(24, 8))  
            ax.plot(oht_data['index'], oht_data['CT1'], label="CT1", color='blue')
            ax.axhline(y=oht_data['CT1'].mean(), color='blue', linestyle='--', label="CT1 평균")
            ax.set_title('CT1 전류측정값', fontsize=18)
            #ax.set_xlabel('순차적 인덱스 (시간처럼 처리)', fontsize=14)
            ax.set_ylabel('전류 (A)', fontsize=14)
            # x축 레이블 간격 조정: 레이블을 500 간격으로 표시
            plt.xticks(range(0, len(oht_data), len(oht_data)//10))
            st.pyplot(fig)

            # CT2 전류측정값 시각화
            fig, ax = plt.subplots(figsize=(24, 8))  
            ax.plot(oht_data['index'], oht_data['CT2'], label="CT2", color='orange')
            ax.axhline(y=oht_data['CT2'].mean(), color='orange', linestyle='--', label="CT2 평균")
            ax.set_title('CT2 전류측정값', fontsize=18)
            #ax.set_xlabel('순차적 인덱스 (시간처럼 처리)', fontsize=14)
            ax.set_ylabel('전류 (A)', fontsize=14)
            # x축 레이블 간격 조정: 레이블을 500 간격으로 표시
            plt.xticks(range(0, len(oht_data), len(oht_data)//10))
            st.pyplot(fig)

            # CT3 전류측정값 시각화
            fig, ax = plt.subplots(figsize=(24, 8))  
            ax.plot(oht_data['index'], oht_data['CT3'], label="CT3", color='green')
            ax.axhline(y=oht_data['CT3'].mean(), color='green', linestyle='--', label="CT3 평균")
            ax.set_title('CT3 전류측정값', fontsize=18)
            #ax.set_xlabel('순차적 인덱스 (시간처럼 처리)', fontsize=14)
            ax.set_ylabel('전류 (A)', fontsize=14)
            # x축 레이블 간격 조정: 레이블을 500 간격으로 표시
            plt.xticks(range(0, len(oht_data), len(oht_data)//10))
            st.pyplot(fig)

            # CT4 전류측정값 시각화
            fig, ax = plt.subplots(figsize=(24, 8))  
            ax.plot(oht_data['index'], oht_data['CT4'], label="CT4", color='purple')
            ax.axhline(y=oht_data['CT4'].mean(), color='purple', linestyle='--', label="CT4 평균")
            ax.set_title('CT4 전류측정값', fontsize=18)
            #ax.set_xlabel('순차적 인덱스 (시간처럼 처리)', fontsize=14)
            ax.set_ylabel('전류 (A)', fontsize=14)
            # x축 레이블 간격 조정: 레이블을 500 간격으로 표시
            plt.xticks(range(0, len(oht_data), len(oht_data)//10))
            st.pyplot(fig)
            
        elif selected_item == "장치 최근 상태":
            # OHT 데이터가 로드되었으면 상태 출력
            if not oht_data.empty:
                latest_state_oht = int(oht_data['state'].iloc[-1])
                st.subheader("OHT 최근 상태")
                if latest_state_oht == 0:
                    st.success("OHT 최근 상태: 정상입니다 ✅")
                elif latest_state_oht == 1:
                    st.warning("OHT 최근 상태: 관심입니다 ⚠️")
                elif latest_state_oht == 2:
                    st.warning("OHT 최근 상태: 경고입니다 ⚠️")
                elif latest_state_oht == 3:
                    st.error("OHT 최근 상태: 위험입니다 ❌")
            else:
                st.error("OHT 데이터가 비어 있습니다 ❌")
            
            
            
    else:
        st.error("OHT 데이터가 비어 있습니다 ❌")

