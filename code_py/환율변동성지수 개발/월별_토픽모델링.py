
# ------------------------------------------------
# 데이터 준비
# ------------------------------------------------

import pandas as pd
import numpy as np


df = pd.read_csv('df_2021_to_2025.csv')
df.shape




# ------------------------------------------------
# EPU에 포함되는 것만 필터링
# ------------------------------------------------


# 1. 경제(E).: 거시경기·실물경제를 직접 지칭
e = [
    "경제", "경기"
]

# 2. 정책(P): 정부·의회·당국 및 모든 정책·제도·규제·조치
p = [
    # ─ 기본 정책 키워드 ─
    "정부", "청와대", "국무회의", "국회", "의회", "당국",
    "한국은행", "한은", "중앙은행",
    "기획재정부", "기재부", "금융위원회", "금융위",
    "정책", "재정", "입법", "법안", "법률", "예산",
    "세금", "규제", "규정", "적자", "부채", "채무",
    "연방준비제도", "연준", "Fed", "FRB",
    "구조개혁", "구조조정",

    # ─ 통화·금융 정책 ─
    "통화정책", "금융완화", "추가완화", "양적완화", "질적완화",
    "통화긴축", "유동성긴축", "마이너스 금리", "통화할인율", "재할인율",
    "통화 운용", "공개시장조작", "물가안정목표", "물가목표",
    "금융통화위원회", "금통위", "금융통화운영위원회", "금융통의",
    "기준금리", "정책금리", "금리정책", "통화당국", "환금리",

    # ─ 재정 정책 ─
    "재정정책", "정부 예산", "추가경정예산", "추경", "일반 회계", "특별 회계",
    "재정적자", "재정수지", "정부지출", "재정지출",
    "사회보장성 지출", "국민연금 보험료", "국민건강 보험료",
    "의료비 지출", "간병비 지출", "의료보수수가", "의료수가",
    "공무원급여", "공적개발원조", "ODA",
    "국방비", "군비",
    "국채 발행 잔여", "공공부문 부채", "재정부채", "국채", "정부부채", "지방채",
    "경기부양", "경기부양책",

    # ─ 무역·통상 정책 ─
    "통상문제", "무역문제", "비관세장벽", "수입제한",
    "포괄통상법", "종합무역법",
    "무역정책", "통상정책", "무역협상",
    "세계무역기구", "WTO",
    "관세 및 무역에 관한 일반협정", "GATT", "가트",
    "관세 인하", "무역자유화", "수입자유화", "시장접근",
    "무역협정", "통상협정",
    "환태평양경제동반자협정", "TPP",
    "경제동반자협정", "경제파트너협정", "경제 파트너십 협정", "EPA",
    "자유무역협정", "FTA",
    "무역분쟁", "관세", "우루과이라운드", "도하라운드", "덤핑",

    # ─ 환율 정책 ─
    "외환정책", "환율정책", "시장개입", "외환시장 개입", "외환당국 개입",
    "미세조정", "환율 안정책", "환율 안정 조치", "환율 변동성 완화",
    "환율 조작", "외환보유고", "외환보유액", "달러매도", "원화 매입 개입"
]

# 3. 불확실(U): 모호성·위험·걱정·우려를 표현
u = [
    "불확실", "불확실성", "리스크", "불투명",
    "불안", "우려", "걱정"
]


import re

cols = ['키워드', '특성추출(가중치순 상위 50개.', '본문', '제목']          # 검색 대상 열
joined = df[cols].fillna(''..agg(' '.join, axis=1.                          # 네 열을 하나로 합침

cond1 = joined.str.contains('|'.join(map(re.escape, e.                    # 경제·경기
cond2 = joined.str.contains('|'.join(map(re.escape, p.                    # 불확실 관련
cond3 = joined.str.contains('|'.join(map(re.escape, u.                    # 정책(정부 등.
                 # 정책(정부 등.
df = df[cond1&cond2&cond3]




# ------------------------------------------------
# 시각화를 위한 데이터 생성
# ------------------------------------------------


df2 = df[['일자','키워드','언론사']]
df2['일자'] = pd.to_datetime(df2['일자'], format='%Y%m%d'.
df2['연'] = df2['일자'].dt.year
df2['월'] = df2['일자'].dt.month
df2 = df2[df2['연'] >= 2023]

# ------------------------------------------------
# 2023년부터 언론사별 기사 수 시각화
# ------------------------------------------------



import matplotlib.pyplot as plt
plt.rcParams['font.family']='Malgun Gothic'
# 언론사별 기사 수 집계
counts = df2['언론사'].value_counts()

# 색상 리스트 생성 (상위 3개는 초록색 계열, 나머지는 회색)
colors = ['skyblue', 'skyblue', 'skyblue'] + ['lightgray'] * (len(counts) - 3)

# 그래프 그리기
plt.figure(figsize=(10, 6))
bars = plt.bar(counts.index, counts.values, color=colors)

# 제목/레이블
plt.title('2023년부터 언론사별 기사 수', fontsize=18)
plt.xlabel('언론사')
plt.ylabel('기사 수')
plt.xticks(rotation=45, ha='right', fontsize=15)
plt.tight_layout()
plt.show()

# ------------------------------------------------
# 상위언론사 전체데이터 토픽모델링
# ------------------------------------------------

상위언론사데이터 = df[df['언론사'].str.contains('서울경제|매일경제|한국경제')][['일자','키워드','언론사','제목']]

상위언론사데이터['일자'] = pd.to_datetime(상위언론사데이터['일자'], format='%Y%m%d')
상위언론사데이터['연'] = 상위언론사데이터['일자'].dt.year
상위언론사데이터['월'] = 상위언론사데이터['일자'].dt.month
상위언론사데이터 = 상위언론사데이터[상위언론사데이터['연'] >= 2023]




# ------------------------------------------------
# LDA 모델링
# ------------------------------------------------

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd

# 1. 전처리: 키워드가 문자열이면 리스트로 변환
상위언론사데이터 = 상위언론사데이터.copy()
상위언론사데이터['키워드'] = 상위언론사데이터['키워드'].apply(
    lambda x: x if isinstance(x, list) else str(x).replace('...', '').split(',')
)

# 2. 키워드 문장화
상위언론사데이터['키워드문장'] = 상위언론사데이터['키워드'].apply(lambda x: ' '.join(x))

# 3. LDA 함수 정의
def lda_by_month(df, n_topics=5, top_n=10):
    result = []

    for (year, month), group in df.groupby(['연', '월']):
        texts = group['키워드문장'].tolist()

        if len(texts) < 2:
            continue  # 문서 수 너무 적으면 스킵

        vectorizer = CountVectorizer(max_df=0.95, min_df=1)
        try:
            X = vectorizer.fit_transform(texts)
        except ValueError:
            continue  # 단어 없음 → 스킵

        if X.shape[1] == 0:
            continue  # 유효 단어 없음 → 스킵

        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(X)
        words = vectorizer.get_feature_names_out()

        for idx, topic in enumerate(lda.components_):
            top_keywords = [words[i] for i in topic.argsort()[:-top_n - 1:-1]]
            result.append({
                '연': year,
                '월': month,
                '토픽번호': idx + 1,
                '상위키워드': ', '.join(top_keywords)
            })

    return pd.DataFrame(result)

# 4. LDA 함수 실행
monthly_topics = lda_by_month(상위언론사데이터, n_topics=3, top_n=10)


# datetime으로 변경
monthly_topics['일자'] = pd.to_datetime(monthly_topics['연'].astype(str) + '-' + monthly_topics['월'].astype(str) + '-01', format='%Y-%m-%d')

# 데이터 저장
monthly_topics[['일자','토픽번호','상위키워드']].to_csv('토픽모델링.csv', index=False)
