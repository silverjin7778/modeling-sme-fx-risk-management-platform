from openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI  # langchain-openai 설치 필요
import os
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI 

# ---------------------------------------------------------------------------------
# 데이터 불러오기
# ---------------------------------------------------------------------------------
df = pd.read_csv('기자간담회.csv')[['Date','Content']]
df = df.dropna()



# ---------------------------------------------------------------------------------
# API KEY : 코드 제출을 위해 환경변수 설정
# ---------------------------------------------------------------------------------

load_dotenv()
sola_api_key = os.getenv('sola_api_key')


# ---------------------------------------------------------------------------------
# Upstage API 클라이언트 생성
# ---------------------------------------------------------------------------------
client = OpenAI(
    api_key=sola_api_key,  # Use the sola_api_key from .env
    base_url="https://api.upstage.ai/v1"
)



# ---------------------------------------------------------------------------------
# 각 Content를 요약
# ---------------------------------------------------------------------------------

summaries = [] # 요약 결과 저장할 리스트

for content in df["Content"]:
    response = client.chat.completions.create(
        model="solar-pro2",
        messages=[
            {
                "role": "system",
                "content": "한국총재가 발표한 통화정책방향을 요약해주세요."
            },
            {
                "role": "user",
                "content": content
            }
        ],
        stream=False
    )
    summary = response.choices[0].message.content.strip()
    summaries.append(summary)

df["default_summary"] = summaries

import pandas as pd
from openai import OpenAI  # openai==1.52.2



# ---------------------------------------------------------------------------------
# COSTAR 프롬프트 엔지니어링
# ---------------------------------------------------------------------------------

system_prompt = """
당신은 글쓰기 전문가입니다. 아래 조건을 바탕으로 글을 작성해주세요:

1. Context:
- 주제는 한국은행 총재가 발표하는 통화정책방향입니다.

2. Objective:
- 독자가 장시간 발표된 내용을 요약본을 통해 편하게 인지할 수 있도록 합니다.

3. Style:
- 전문적인 글쓰기 스타일.

4. Tone:
- 격식 있고 신뢰감을 주는 톤.

5. Audience:
- 은행 직원으로 한국은행 총재의 발표 내용을 이해하고 요약할 수 있는 능력을 갖춘 사람들입니다.

6. Response:
- 가능한 한 심층 요약 글로 작성해주세요.
- 발표 내용을 주제별로 구분하여 항목별로 정리된 형태로 요약해주세요.
- 항목 제목은 발표 내용에 맞게 자유롭게 생성하되, 일관된 서식(숫자 또는 기호)을 사용해주세요.
- 발표에서 다룬 주제가 많을 경우 항목 수를 늘려도 좋습니다.

예시 형식:
1. ○○ 관련 발표 내용 요약  
2. △△ 배경 및 한국은행의 입장  
3. 향후 대응 방향
"""




# ---------------------------------------------------------------------------------
# 각 Content를 요약
# ---------------------------------------------------------------------------------

summaries = [] # 요약 결과 저장할 리스트
for content in df["Content"]:
    response = client.chat.completions.create(
        model="solar-pro2",
        messages=[
            {
                "role": "system",
                "content": system_prompt.strip()
            },
            {
                "role": "user",
                "content": content.strip()
            }
        ],
        stream=False
    )
    summary = response.choices[0].message.content.strip()
    summaries.append(summary)

df["sola_summary"] = summaries



# ---------------------------------------------------------------------------------
# 요약 결과 데이터파일로 저장
# ---------------------------------------------------------------------------------

df.to_csv('sola_기자간담회_요약.csv', index=False)  

