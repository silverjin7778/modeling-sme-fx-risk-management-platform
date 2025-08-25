# .\euroenv\Scripts\activate
# streamlit run C:/Users/campus4D035/Desktop/final/chat02/yotube/result_streamlit.py


import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import yfinance as yf
import matplotlib.pyplot as plt

# 페이지 설정
st.set_page_config(page_title="iM 환율적 참견 시점", layout="wide")

# 이미지 URL
image_url = "https://i.ibb.co/LXjc3FMM/image.png"

# 페이지 상태 초기화
if "selected_menu" not in st.session_state:
    st.session_state.selected_menu = "홈"

# 고급 스타일 정의
st.markdown("""
    <style>
    .sidebar-card {
        background-color: #f2f4f8;
        padding: 30px 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        font-family: 'Segoe UI', sans-serif;
    }
    .menu-title {
        font-size: 20px;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 20px;
    }
    div[data-testid="stButton"] > button {
        display: block;
        width: 100%;
        padding: 10px 20px;
        margin-bottom: 10px;
        border: none;
        border-radius: 8px;
        background-color: #ffffff;
        color: #2c3e50;
        font-size: 16px;
        text-align: left;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        transition: 0.2s;
    }
    div[data-testid="stButton"] > button:hover {
        background-color: #e3e8ef;
    }
    </style>
""", unsafe_allow_html=True)

# 전체 레이아웃 구성
left_col, right_col = st.columns([1, 3])

# 왼쪽 메뉴
with left_col:
    def styled_button(label, menu_name):
        if st.button(label):
            st.session_state.selected_menu = menu_name

    styled_button("\U0001F3E0 홈", "홈")
    styled_button("\U0001F4C8 환율 변동성지수", "환율 변동성지수")
    styled_button("\U0001F52E 환율 예측기", "환율예측해보기")
    styled_button("\U0001F4F0 통화정책 브리핑", "통화정책 브리핑")

# 오른쪽 콘텐츠
with right_col:
    menu = st.session_state.selected_menu

    if menu == "홈":
        col1, col2, col3 = st.columns([0.8, 2, 1.2])
        with col2:
            st.image(image_url, width=500)
            st.markdown("<div style='font-size:16px; color:gray; text-align:center;'>이건 다섯 번째 레슨, 좋은 건 너만 알기😉</div>", unsafe_allow_html=True)

    elif menu == "환율 변동성지수":
        import streamlit as st
        import pandas as pd
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime
        from pandas.tseries.offsets import DateOffset  # ✅ DateOffset을 따로 import

        # 📌 한글 폰트 설정
        plt.rcParams['font.family'] = 'Malgun Gothic'

        # 📄 페이지 설정
        st.set_page_config(page_title="미환율 & 한국 EPU 분석", layout="wide")

        # 📂 데이터 불러오기
        topic_df = pd.read_csv('C:/Users/campus4D035/Desktop/final/news/토픽모델링.csv')
        df = pd.read_csv('C:/Users/campus4D035/Desktop/final/news/어쩌면최종데이터_two_Korea_EPU.csv')

        # 📆 날짜 변환
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        topic_df['일자'] = pd.to_datetime(topic_df['일자'], errors='coerce')

        # 📉 EPU/환율 데이터 정리
        df = df[['Date', '미환율_종가', '한국EPU']].sort_values('Date').set_index('Date')
        ten_years_ago = pd.Timestamp.today() - DateOffset(years=10)
        df_filtered = df[df.index >= ten_years_ago]

        # 📅 날짜 선택
        st.markdown("## 📌 환율 변동성 지수&월별 주요 이슈")
        available_dates = topic_df['일자'].dropna().sort_values().dt.strftime('%Y-%m-%d').unique()[::-1]
        selected_date_str = st.selectbox("날짜를 선택하세요", available_dates, index=0)
        selected_date = pd.to_datetime(selected_date_str)

        # 📋 해당 날짜 토픽 추출
        topic_on_date = topic_df[topic_df['일자'] == selected_date]

        st.markdown("---")
        st.markdown(f"### 선택한 날짜: {selected_date.strftime('%Y년 %m월 %d일')}")
        if topic_on_date.empty:
            st.info("해당 날짜에 대한 토픽 정보가 없습니다.")
        else:
            for idx, row in topic_on_date.iterrows():
                st.markdown(f"- **토픽 {row['토픽번호']}**: {row['상위키워드']}")

        # 📆 선택한 날짜의 월 시작일과 다음 달 시작일 계산
        month_start = selected_date.replace(day=1)
        if month_start.month == 12:
            next_month_start = month_start.replace(year=month_start.year + 1, month=1)
        else:
            next_month_start = month_start.replace(month=month_start.month + 1)

        # 📈 그래프 출력
        fig, ax1 = plt.subplots(figsize=(10, 4))

        # 환율
        ax1.plot(df_filtered.index, df_filtered['미환율_종가'], color='#1f77b4', linewidth=1.5, label='미환율 종가')
        ax1.set_ylabel("미환율 종가", color='#1f77b4', fontsize=11)
        ax1.tick_params(axis='y', labelcolor='#1f77b4')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        # EPU
        ax2 = ax1.twinx()
        ax2.plot(df_filtered.index, df_filtered['한국EPU'], color='#d62728', linestyle='--', linewidth=1.5, label='한국 EPU')
        ax2.set_ylabel("환율 변동성 지수", color='#d62728', fontsize=11)
        ax2.tick_params(axis='y', labelcolor='#d62728')

        # ✅ 월 시작일 수직선 + 해당 월 음영
        ax1.axvline(month_start, color='gray', linestyle=':', linewidth=1)
        ax1.axvspan(month_start, next_month_start, color='gray', alpha=0.1)

        # 제목 및 여백
        fig.suptitle("미환율 및 환율 변동성 지수 추이", fontsize=14, fontweight='bold')
        fig.tight_layout()
        st.pyplot(fig)


    elif menu == "환율예측해보기":
        st.subheader("\U0001F52E 환율예측해보기")
        currency_selected = st.selectbox("예측할 화폐를 선택하세요:", ["", "유로", "달러", "위안", "엔"], index=0)

        if currency_selected == "유로" : 
            st.title("💶 유로 환율 예측 및 환전 계산기 + 선물환 전략 추천")
            시점 = st.radio("예측 시점 선택", ['하루', '일주일', '한달', '세달'])
            기업유형 = st.radio("당신의 기업은?", ['수출기업', '수입기업'])
            if 기업유형 == '수출기업':
                금액 = st.number_input("금액 입력 €", min_value=0.0, step=100.0)
            else :
                금액 = st.number_input("금액 입력 €", min_value=0.0, step=100.0)

            def 추천_전략(result, 기업유형):
                prev = result['이전 종가'].values[0]
                pred = result['예측 종가'].values[0]
                if 기업유형 == '수출기업':
                    return "환율 상승 시 수익 증대 효과, 추가 대응 불필요" if pred > prev else "선물환 매도 또는 Put옵션 매수"
                elif 기업유형 == '수입기업':
                    return "선물환 매수 또는 Call옵션 매수" if pred > prev else "환율 하락 시 원가 절감 가능, 추가 대응 불필요"
                return "❌ ERROR"

            def euro_indicator():
                tickers = {"DAX": "^GDAXI", "EUROSTOXX50": "^STOXX50E", "CAC": "^FCHI"}
                data_list = []
                for name, symbol in tickers.items():
                    df = yf.Ticker(symbol).history(period="25y", interval="1d").reset_index()
                    df = df[["Date", "Open", "High", "Low", "Close"]]
                    df.rename(columns={
                        "Open": f"{name}_Open", "High": f"{name}_High",
                        "Low": f"{name}_Low", "Close": f"{name}_Close"
                    }, inplace=True)
                    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
                    data_list.append(df)
                df_merged = data_list[0]
                for df in data_list[1:]:
                    df_merged = pd.merge(df_merged, df, on="Date", how="inner")
                return df_merged

            def real_times(symbol):
                df = yf.Ticker(symbol).history(period="25y", interval="1d")[["Open", "High", "Low", "Close"]]
                df["Date"] = df.index.date
                df.reset_index(drop=True, inplace=True)
                df["Change"] = df["Close"].diff()
                df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
                return df[['Date', 'Close', 'Open', 'High', 'Low', 'Change']]

            def euro_predict(timelength, model, scaler_X, scaler_y, df_base):
                seq_len = 20
                new_df = df_base.tail(seq_len + 1)

                X_new = new_df[[  # 동일 순서 주의
                    'DAX_Open', 'DAX_High', 'DAX_Low', 'DAX_Close',
                    'EUROSTOXX50_Open', 'EUROSTOXX50_High', 'EUROSTOXX50_Low', 'EUROSTOXX50_Close',
                    'CAC_Open', 'CAC_High', 'CAC_Low', 'CAC_Close',
                    'Close', 'Open', 'High', 'Low', 'Change'
                ]]
                X_scaled = scaler_X.transform(X_new)

                if timelength in ['일주일', '세달']:  # LSTM
                    X_input = np.expand_dims(X_scaled[:-1], axis=0)
                    pred_scaled = model.predict(X_input)
                    pred_close = scaler_y.inverse_transform(pred_scaled)[0][0]
                else:  # 하루, 한달 → RandomForest
                    X_input = X_scaled[seq_len:]
                    pred_scaled = model.predict(X_input)
                    pred_close = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]

                prev_close = new_df['Close'].iloc[-1]
                return pd.DataFrame({
                    "Date": [new_df['Date'].iloc[-1]],
                    "이전 종가": [prev_close],
                    "예측 종가": [pred_close],
                    "예측결과": ["상승" if pred_close > prev_close else "하락"]
                })

            if st.button("예측 및 환전 계산"):
                model_path = "C:/Users/campus4D035/Desktop/final/streamlit/euro/모델/유로"
                model = joblib.load(f"{model_path}/euro_{시점}.pkl")
                scaler_X = joblib.load(f"{model_path}/euro_scaler_X_{시점}.pkl")
                scaler_y = joblib.load(f"{model_path}/euro_scaler_y_{시점}.pkl")

                indi_df = euro_indicator()
                df_euro = real_times("EURKRW=X")
                df_base = pd.merge(indi_df, df_euro, on="Date", how="inner")

                result = euro_predict(시점, model, scaler_X, scaler_y, df_base)

                st.markdown("### 📊 예측 결과")
                st.write(result[['이전 종가', '예측 종가', '예측결과']])
                diff = result['예측 종가'].values[0] - result['이전 종가'].values[0]
                now_won = 금액 * result['이전 종가'].values[0]
                future_won = 금액 * result['예측 종가'].values[0]
                current_price = result['이전 종가'].values[0]
                future_price = result['예측 종가'].values[0]
                st.markdown("### 💱 환전 계산 결과")

                current_price = result['이전 종가'].values[0]     # 현재 환율
                future_price = result['예측 종가'].values[0]       # 예측 환율
                diff = future_price - current_price                # 환율 차이

                if 기업유형 == '수출기업':
                    # 외화 → 원화: 현재/미래 환율에 따라 환전 결과 다름
                    now_won = 금액 * current_price
                    future_won = 금액 * future_price
                    result_word = "이득" if future_won > now_won else "손해"
                    
                    st.markdown(f"- 현재 환전: **약 {now_won:,.0f} 원**")
                    st.markdown(f"- {시점} 후 환전: **약 {future_won:,.0f} 원**")
                    st.markdown(f"➞️ **약 {abs(future_won - now_won):,.0f} 원 {result_word}**")

                elif 기업유형 == '수입기업':
                    # 원화 → 외화: 같은 원화로 환전 가능한 외화량 변화
                    now_foreign = 금액 / current_price
                    future_foreign = 금액 / future_price
                    result_word = "이득" if future_foreign > now_foreign else "손해"

                    now_won = 금액 * current_price
                    future_won = 금액 * future_price
                    now_foreign = 금액 / current_price
                    future_foreign = 금액 / future_price
                    result_word = "이득" if future_foreign > now_foreign else "손해"

                    st.markdown(f"- 현재 환전: **약 {now_won:,.0f} 원**")
                    st.markdown(f"- {시점} 후 환전: **약 {future_won:,.0f} 원**")
                    st.markdown(f"➞️ **약 {abs(future_won - now_won):,.0f} 원 {result_word}**")

                # 환율 요약 출력
                st.markdown(
                    f"📝 현재 환율은 {current_price:,.2f}원이며, "
                    f"{시점} 후 {future_price:,.2f}원으로 "
                    f"{result['예측결과'].values[0]}이 예상됩니다."
                )

                # 전략 출력
                st.markdown("### 📌 추천 전략")
                전략 = 추천_전략(result, 기업유형)
                st.success(f"{기업유형}에게 추천되는 전략은 **{전략}** 입니다.")



        elif currency_selected == "달러" : 
            st.title("💵 달러 환율 예측 및 환전 계산기 + 선물환 전략 추천")
            시점 = st.radio("예측 시점 선택", ['하루', '일주일', '한달', '세달'])
            기업유형 = st.radio("당신의 기업은?", ['수출기업', '수입기업'])
            if 기업유형 == '수출기업':
                금액 = st.number_input("금액 입력 $", min_value=0.0, step=100.0)
            else :
                금액 = st.number_input("금액 입력 $", min_value=0.0, step=100.0)

            def 추천_전략(result, 기업유형):
                prev = result['이전 종가'].values[0]
                pred = result['예측 종가'].values[0]
                if 기업유형 == '수출기업':
                    return "환율 상승 시 수익 증대 효과, 추가 대응 불필요" if pred > prev else "선물환 매도 또는 Put옵션 매수"
                elif 기업유형 == '수입기업':
                    return "선물환 매수 또는 Call옵션 매수" if pred > prev else "환율 하락 시 원가 절감 가능, 추가 대응 불필요"
                return "❌ ERROR"

            def dollar_indicator():
                tickers = {
                    "S&P500": "^GSPC",
                    "DowJones": "^DJI",
                    "NASDAQ": "^IXIC"
                }
                data_list = []
                for name, symbol in tickers.items():
                    df = yf.Ticker(symbol).history(period="25y", interval="1d").reset_index()
                    df = df[["Date", "Open", "High", "Low", "Close"]]
                    df = df.rename(columns={
                        "Open": f"{name}_Open", "High": f"{name}_High",
                        "Low": f"{name}_Low", "Close": f"{name}_Close"
                    })
                    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
                    data_list.append(df)
                df_merged = data_list[0]
                for df in data_list[1:]:
                    df_merged = pd.merge(df_merged, df, on="Date", how="inner")
                df_merged['Date'] = pd.to_datetime(df_merged['Date'])
                return df_merged

            def real_times(symbol):
                df = yf.Ticker(symbol).history(period="25y", interval="1d")[["Open", "High", "Low", "Close"]]
                df["Date"] = df.index.date
                df.reset_index(drop=True, inplace=True)
                df["Change"] = df["Close"].diff()
                df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
                return df[['Date', 'Close', 'Open', 'High', 'Low', 'Change']]

            def dollar_predict(timelength, model, scaler_X, scaler_y, df_base):
                seq_len = 20
                new_df = df_base.tail(seq_len + 1)

                X_new = new_df[[
                    'S&P500_Open', 'S&P500_High', 'S&P500_Low', 'S&P500_Close',
                    'DowJones_Open', 'DowJones_High', 'DowJones_Low', 'DowJones_Close',
                    'NASDAQ_Open', 'NASDAQ_High', 'NASDAQ_Low', 'NASDAQ_Close',
                    'Close', 'Open', 'High', 'Low', 'Change'
                ]]

                X_scaled = scaler_X.transform(X_new)
                X_seq = np.expand_dims(X_scaled[:-1], axis=0)  # LSTM은 3D 입력 필요

                pred_scaled = model.predict(X_seq)
                pred_close = scaler_y.inverse_transform(pred_scaled)[0][0]

                prev_close = new_df['Close'].iloc[-1]

                return pd.DataFrame({
                    "Date": [new_df['Date'].iloc[-1]],
                    "이전 종가": [prev_close],
                    "예측 종가": [pred_close],
                    "예측결과": ["상승" if pred_close > prev_close else "하락"]
                })

            if st.button("예측 및 환전 계산"):
                model_path = "C:/Users/campus4D035/Desktop/final/streamlit/euro/모델/달러"
                model = joblib.load(f"{model_path}/dollar_{시점}.pkl")
                scaler_X = joblib.load(f"{model_path}/dollar_scaler_X_{시점}.pkl")
                scaler_y = joblib.load(f"{model_path}/dollar_scaler_y_{시점}.pkl")

                indi_df = dollar_indicator()
                df_dollar = real_times("USDKRW=X")
                df_base = pd.merge(indi_df, df_dollar, on="Date", how="inner")

                result = dollar_predict(시점, model, scaler_X, scaler_y, df_base)

                st.markdown("### 📊 예측 결과")
                st.write(result[['이전 종가', '예측 종가', '예측결과']])


                st.markdown("### 💱 환전 계산 결과")

                current_price = result['이전 종가'].values[0]     # 현재 환율
                future_price = result['예측 종가'].values[0]       # 예측 환율
                diff = future_price - current_price                # 환율 차이

                if 기업유형 == '수출기업':
                    # 외화 → 원화: 현재/미래 환율에 따라 환전 결과 다름
                    now_won = 금액 * current_price
                    future_won = 금액 * future_price
                    result_word = "이득" if future_won > now_won else "손해"
                    
                    st.markdown(f"- 현재 환전: **약 {now_won:,.0f} 원**")
                    st.markdown(f"- {시점} 후 환전: **약 {future_won:,.0f} 원**")
                    st.markdown(f"➞️ **약 {abs(future_won - now_won):,.0f} 원 {result_word}**")

                elif 기업유형 == '수입기업':
                    # 원화 → 외화: 같은 원화로 환전 가능한 외화량 변화
                    now_won = 금액 * current_price
                    future_won = 금액 * future_price
                    now_foreign = 금액 / current_price
                    future_foreign = 금액 / future_price
                    result_word = "이득" if future_foreign > now_foreign else "손해"

                    st.markdown(f"- 현재 환전: **약 {now_won:,.0f} 원**")
                    st.markdown(f"- {시점} 후 환전: **약 {future_won:,.0f} 원**")
                    st.markdown(f"➞️ **약 {abs(future_won - now_won):,.0f} 원 {result_word}**")

                # 환율 요약 출력
                st.markdown(
                    f"📝 현재 환율은 {current_price:,.2f}원이며, "
                    f"{시점} 후 {future_price:,.2f}원으로 "
                    f"{result['예측결과'].values[0]}이 예상됩니다."
                )

                # 전략 출력
                st.markdown("### 📌 추천 전략")
                전략 = 추천_전략(result, 기업유형)
                st.success(f"{기업유형}에게 추천되는 전략은 **{전략}** 입니다.")

 
        elif currency_selected == "위안" : 
            st.title("🇨🇳💰 위안화 환율 예측 및 환전 계산기 + 선물환 전략 추천")
            시점 = st.radio("예측 시점 선택", ['하루','일주일','한달','세달'])
            기업유형 = st.radio("당신의 기업은?", ['수출기업', '수입기업'])
            if 기업유형 == '수출기업':
                금액 = st.number_input("금액 입력 元", min_value=0.0, step=100.0)
            else :
                금액 = st.number_input("금액 입력 元", min_value=0.0, step=100.0)

            def 추천_전략(result, 기업유형):
                prev = result['이전 종가'].values[0]
                pred = result['예측 종가'].values[0]
                if 기업유형 == '수출기업':
                    return "환율 상승 시 수익 증대 효과, 추가 대응 불필요" if pred > prev else "선물환 매도 또는 Put옵션 매수"
                elif 기업유형 == '수입기업':
                    return "선물환 매수 또는 Call옵션 매수" if pred > prev else "환율 하락 시 원가 절감 가능, 추가 대응 불필요"
                return "❌ ERROR"

            def yuan_indicator():
                tickers = {
                    "Shanghai Composite": "000001.SS",
                    "Shenzhen Component": "399001.SZ",
                    "Hang Seng": "^HSI",
                    "Hang Seng China Enterprises": "^HSCE"
                }
                data_list = []
                for name, symbol in tickers.items():
                    df = yf.Ticker(symbol).history(period="25y", interval="1d").reset_index()
                    df = df[["Date", "Open", "High", "Low", "Close"]]
                    df = df.rename(columns={
                        "Open": f"{name}_Open", "High": f"{name}_High",
                        "Low": f"{name}_Low", "Close": f"{name}_Close"
                    })
                    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
                    data_list.append(df)
                df_merged = data_list[0]
                for df in data_list[1:]:
                    df_merged = pd.merge(df_merged, df, on="Date", how="inner")
                df_merged['Date'] = pd.to_datetime(df_merged['Date'])
                return df_merged

            def yuan_df_load():
                df = pd.read_csv('C:/Users/campus4D035/Desktop/final/streamlit/euro/모델/위안/yuan_df.csv')
                df['Date'] = pd.to_datetime(df['Date'])
                return df

            def yuan_predict(timelength, model, scaler_X, scaler_y, df_base):
                seq_len = 20
                new_df = df_base.tail(seq_len + 1)

                # 입력 피처 구성
                X_new = new_df[[
                    'Shanghai Composite_Open', 'Shanghai Composite_High', 'Shanghai Composite_Low', 'Shanghai Composite_Close',
                    'Shenzhen Component_Open', 'Shenzhen Component_High', 'Shenzhen Component_Low', 'Shenzhen Component_Close',
                    'Hang Seng_Open', 'Hang Seng_High', 'Hang Seng_Low', 'Hang Seng_Close',
                    'Hang Seng China Enterprises_Open', 'Hang Seng China Enterprises_High',
                    'Hang Seng China Enterprises_Low', 'Hang Seng China Enterprises_Close',
                    'Close', 'Open', 'High', 'Low', 'Change'
                ]]

                # 정규화
                X_scaled = scaler_X.transform(X_new)

                # 모델 타입별 입력 처리
                if "한달" in timelength or "세달" in timelength:
                    # 예: RandomForest, XGBoost 등 2D 입력 필요
                    X_input = X_scaled[seq_len:]  # shape: (n_samples, n_features)
                    pred_scaled = model.predict(X_input)
                    pred_close = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
                else:
                    # 예: LSTM 등 3D 입력 필요
                    X_input = np.expand_dims(X_scaled[:-1], axis=0)  # shape: (1, seq_len, n_features)
                    pred_scaled = model.predict(X_input)
                    pred_close = scaler_y.inverse_transform(pred_scaled)[0][0]

                prev_close = new_df['Close'].iloc[-1]

                result_df = pd.DataFrame({
                    "Date": [new_df['Date'].iloc[-1]],
                    "이전 종가": [prev_close],
                    "예측 종가": [pred_close],
                    "예측결과": ["상승" if pred_close > prev_close else "하락"]
                })

                return result_df


            if st.button("예측 및 환전 계산"):
                model_path = "C:/Users/campus4D035/Desktop/final/streamlit/euro/모델/위안"
                model = joblib.load(f"{model_path}/yuan_{시점}.pkl")
                scaler_X = joblib.load(f"{model_path}/yuan_scaler_X_{시점}.pkl")
                scaler_y = joblib.load(f"{model_path}/yuan_scaler_y_{시점}.pkl")

                indi_df = yuan_indicator()
                df_yuan = yuan_df_load()
                df_base = pd.merge(indi_df, df_yuan, on='Date', how='inner')

                result = yuan_predict(시점, model, scaler_X, scaler_y, df_base)

                st.markdown("### 📊 예측 결과")
                st.write(result[['이전 종가', '예측 종가', '예측결과']])

                st.markdown("### 💱 환전 계산 결과")

                current_price = result['이전 종가'].values[0]     # 현재 환율
                future_price = result['예측 종가'].values[0]       # 예측 환율
                diff = future_price - current_price                # 환율 차이

                if 기업유형 == '수출기업':
                    # 외화 → 원화: 현재/미래 환율에 따라 환전 결과 다름
                    now_won = 금액 * current_price
                    future_won = 금액 * future_price
                    result_word = "이득" if future_won > now_won else "손해"
                    
                    st.markdown(f"- 현재 환전: **약 {now_won:,.0f} 원**")
                    st.markdown(f"- {시점} 후 환전: **약 {future_won:,.0f} 원**")
                    st.markdown(f"➞️ **약 {abs(future_won - now_won):,.0f} 원 {result_word}**")

                elif 기업유형 == '수입기업':
                    # 원화 → 외화: 같은 원화로 환전 가능한 외화량 변화
                    now_foreign = 금액 / current_price
                    future_foreign = 금액 / future_price
                    result_word = "이득" if future_foreign > now_foreign else "손해"

                    now_won = 금액 * current_price
                    future_won = 금액 * future_price
                    now_foreign = 금액 / current_price
                    future_foreign = 금액 / future_price
                    result_word = "이득" if future_foreign > now_foreign else "손해"

                    st.markdown(f"- 현재 환전: **약 {now_won:,.0f} 원**")
                    st.markdown(f"- {시점} 후 환전: **약 {future_won:,.0f} 원**")
                    st.markdown(f"➞️ **약 {abs(future_won - now_won):,.0f} 원 {result_word}**")

                # 환율 요약 출력
                st.markdown(
                    f"📝 현재 환율은 {current_price:,.2f}원이며, "
                    f"{시점} 후 {future_price:,.2f}원으로 "
                    f"{result['예측결과'].values[0]}이 예상됩니다."
                )

                # 전략 출력
                st.markdown("### 📌 추천 전략")
                전략 = 추천_전략(result, 기업유형)
                st.success(f"{기업유형}에게 추천되는 전략은 **{전략}** 입니다.")


        elif currency_selected == "엔" :
            st.title("💴 엔화 환율 예측 및 환전 계산기 + 선물환 전략 추천")
            시점 = st.radio("예측 시점 선택", ['하루','일주일','한달','세달'])  # 추후 확장 가능
            기업유형 = st.radio("당신의 기업은?", ['수출기업', '수입기업'])
            if 기업유형 == '수출기업':
                금액 = st.number_input("금액 입력 ¥", min_value=0.0, step=100.0)
            else :
                금액 = st.number_input("금액 입력 ¥", min_value=0.0, step=100.0)

            # 추천 전략 함수
            def 추천_전략(result, 기업유형):
                prev = result['이전 종가'].values[0]
                pred = result['예측 종가'].values[0]
                if 기업유형 == '수출기업':
                    return "환율 상승 시 수익 증대 효과, 추가 대응 불필요" if pred > prev else "선물환 매도 또는 Put옵션 매수"
                elif 기업유형 == '수입기업':
                    return "선물환 매수 또는 Call옵션 매수" if pred > prev else "환율 하락 시 원가 절감 가능, 추가 대응 불필요"
                return "❌ ERROR"

            # 지표 및 환율 데이터
            def yen_indicator():
                tickers = {"Nikkei225": "^N225", "TOPIX": "^TOPX", "Mothers": "^MTHR"}
                data_list = []
                for name, symbol in tickers.items():
                    df = yf.Ticker(symbol).history(period="25y", interval="1d").reset_index()
                    df = df[["Date", "Open", "High", "Low", "Close"]].rename(columns={
                        "Open": f"{name}_Open", "High": f"{name}_High", "Low": f"{name}_Low", "Close": f"{name}_Close"
                    })
                    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
                    data_list.append(df)
                df_merged = data_list[0]
                for df in data_list[1:]:
                    df_merged = pd.merge(df_merged, df, on="Date", how="inner")
                df_merged['Date'] = pd.to_datetime(df_merged['Date']).dt.strftime('%Y-%m-%d')
                return pd.to_datetime(df_merged['Date']), df_merged

            def real_times(symbol):
                df = yf.Ticker(symbol).history(period="25y", interval="1d")[["Open", "High", "Low", "Close"]]
                df["Date"] = df.index.date
                df.reset_index(drop=True, inplace=True)
                df["Change"] = df["Close"].diff()
                df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
                return df[['Date', 'Close', 'Open', 'High', 'Low', 'Change']]

            def yen_predict(timelength, real_times, yen_indicator, model, scaler_X, scaler_y):
                yen_df = real_times("JPYKRW=X")
                _, indi_df = yen_indicator()
                indi_df['Date'] = pd.to_datetime(indi_df['Date'])
                yen_df['Date'] = pd.to_datetime(yen_df['Date'])

                new_df = pd.merge(indi_df, yen_df, on="Date").tail(21)

                X_new = new_df[[
                    'Nikkei225_Open', 'Nikkei225_High', 'Nikkei225_Low', 'Nikkei225_Close',
                    'TOPIX_Open', 'TOPIX_High', 'TOPIX_Low', 'TOPIX_Close',
                    'Mothers_Open', 'Mothers_High', 'Mothers_Low', 'Mothers_Close',
                    'Close', 'Open', 'High', 'Low', 'Change'
                ]]
                X_scaled = scaler_X.transform(X_new)

                # 모델 타입 분기 처리
                try:
                    # 트리 기반 모델 (e.g., RandomForest, XGBoost): 2D 입력
                    if hasattr(model, "n_estimators") or hasattr(model, "feature_importances_"):
                        X_input = X_scaled[20:].reshape(1, -1)  # 마지막 한 줄만 예측에 사용
                    else:
                        # LSTM 등 딥러닝 모델: 3D 입력
                        X_input = np.expand_dims(X_scaled[:-1], axis=0)
                except:
                    X_input = np.expand_dims(X_scaled[:-1], axis=0)

                # 예측
                pred_scaled = model.predict(X_input)
                pred_close = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]

                prev_close = new_df['Close'].iloc[-1]

                result_df = pd.DataFrame({
                    "Date": [new_df['Date'].iloc[-1]],
                    "이전 종가": [prev_close],
                    "예측 종가": [pred_close],
                    "예측결과": ["상승" if pred_close > prev_close else "하락"]
                })

                return result_df

            if st.button("예측 및 환전 계산"):
                # 모델 불러오기
                model_path = "C:/Users/campus4D035/Desktop/final/streamlit/euro/모델/엔화"
                scaler_X = joblib.load(f"{model_path}/yen_scaler_X_{시점}.pkl")
                scaler_y = joblib.load(f"{model_path}/yen_scaler_y_{시점}.pkl")
                model_lstm = joblib.load(f"{model_path}/yen_{시점}.pkl")

                # 예측 실행
                result = yen_predict(시점, real_times, yen_indicator, model_lstm, scaler_X, scaler_y)

                # 결과 출력
                st.markdown("### 📊 예측 결과")
                st.write(result[['이전 종가', '예측 종가', '예측결과']])


                st.markdown("### 💱 환전 계산 결과")

                current_price = result['이전 종가'].values[0]     # 현재 환율
                future_price = result['예측 종가'].values[0]       # 예측 환율
                diff = future_price - current_price                # 환율 차이

                if 기업유형 == '수출기업':
                    # 외화 → 원화: 현재/미래 환율에 따라 환전 결과 다름
                    now_won = 금액 * current_price
                    future_won = 금액 * future_price
                    result_word = "이득" if future_won > now_won else "손해"
                    
                    st.markdown(f"- 현재 환전: **약 {now_won:,.0f} 원**")
                    st.markdown(f"- {시점} 후 환전: **약 {future_won:,.0f} 원**")
                    st.markdown(f"➞️ **약 {abs(future_won - now_won):,.0f} 원 {result_word}**")

                elif 기업유형 == '수입기업':
                    # 원화 → 외화: 같은 원화로 환전 가능한 외화량 변화
                    now_foreign = 금액 / current_price
                    future_foreign = 금액 / future_price
                    result_word = "이득" if future_foreign > now_foreign else "손해"

                    now_won = 금액 * current_price
                    future_won = 금액 * future_price
                    now_foreign = 금액 / current_price
                    future_foreign = 금액 / future_price
                    result_word = "이득" if future_foreign > now_foreign else "손해"

                    st.markdown(f"- 현재 환전: **약 {now_won:,.0f} 원**")
                    st.markdown(f"- {시점} 후 환전: **약 {future_won:,.0f} 원**")
                    st.markdown(f"➞️ **약 {abs(future_won - now_won):,.0f} 원 {result_word}**")

                # 환율 요약 출력
                st.markdown(
                    f"📝 현재 환율은 {current_price:,.2f}원이며, "
                    f"{시점} 후 {future_price:,.2f}원으로 "
                    f"{result['예측결과'].values[0]}이 예상됩니다."
                )

                # 전략 출력
                st.markdown("### 📌 추천 전략")
                전략 = 추천_전략(result, 기업유형)
                st.success(f"{기업유형}에게 추천되는 전략은 **{전략}** 입니다.")




    elif menu == "통화정책 브리핑":
        st.subheader("\U0001F4F0 최신 통화정책방향 브리핑")
        df = pd.read_csv('C:/Users/campus4D035/Desktop/final/chat02/yotube/sola_기자간담회_요약.csv')[['Date', 'sola_summary', 'default_summary']]
        selected_date = st.radio("\U0001F4C6 날짜 선택:", df['Date'].tolist()[::-1], horizontal=True)
        summary = df.loc[df['Date'] == selected_date, 'default_summary'].values[0]
        st.markdown(f"### \U0001F4DD 요약 내용 ({selected_date})")
        st.write(summary)