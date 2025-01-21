# 기본 Python 라이브러리
import os
import sys
import json
import requests
import xml.etree.ElementTree as ET
 
# 외부 라이브러리
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
 
# OpenAI 관련 라이브러리
import openai
from openai import OpenAI
 
# LangChain 관련 라이브러리
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI

import joblib 
from datetime import datetime
from langchain.chains import LLMChain
import pdfkit

from fastapi.responses import FileResponse



 
# FastAPI 앱 생성
app = FastAPI()
 
# CORS 미들웨어 설정 (프론트엔드와 통신을 위해 필요?)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# 전역 변수
qa_chain = None
openai_api_key = None
 
# 유틸리티 함수들
def load_file(filepath):
    with open(filepath, 'r') as file:
        return file.readline().strip()


def load_csv(file_path):
    try:
        df = pd.read_csv(file_path, encoding="euc-kr")
        print(f"Loaded {len(df)} rows and {len(df.columns)} columns.")
        
        return df
    except UnicodeDecodeError:
        raise ValueError("Error: Unable to decode the file. Please check the file encoding.")


 
def extract_text_data(df):
    if {'구분','내용'}.issubset(df.columns):
        return [f"구분: {row['구분']}\n내용: {row['내용']}" for _, row in df.iterrows()]
    else:
        raise ValueError("The required columns ('구분', '내용') are not found in the CSV file.")

def split_texts(text_data, chunk_size=1000, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = []
    for text in text_data:
        split_docs.extend(text_splitter.split_text(text))
    print(f"Split into {len(split_docs)} chunks.")
    return split_docs
 
def create_vectorstore(split_docs, embeddings_model):
    try:
        vectorstore = FAISS.from_texts(split_docs, embeddings_model)
        print("Embeddings created and vectorstore generated.")
        return vectorstore
    except Exception as e:
        raise RuntimeError(f"Error during embeddings creation: {e}")
 
def save_vectorstore(vectorstore, path):
    vectorstore.save_local(path)
    print(f"Vectorstore saved locally as '{path}'.")
 
def load_vectorstore(path, embeddings_model):
    return FAISS.load_local(path, embeddings_model, allow_dangerous_deserialization=True)
 
def setup_qa_chain(vectorstore):
    prompt = PromptTemplate.from_template(
        """당신은 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다. 당신의 임무는 주어진 문맥(context) 에서 주어진 질문(question) 에 답하는 것입니다.
        검색된 다음 문맥(context) 을 사용하여 질문(question) 에 답하세요. 만약, 주어진 문맥(context) 에서 답을 찾을 수 없다면, 답을 모른다면 `주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다` 라고 답하세요.
        한글로 답변해 주세요. 단, 기술적인 용어나 이름은 번역하지 않고 그대로 사용해 주세요. 
 
        #Question:
        {question}
 
        #Context:
        {context}
 
        #Answer:  (자세하게 작성해주세요)""" 
    )
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
 
# 앱 시작 시 초기화
@app.on_event("startup")
async def startup_event():
    global qa_chain, openai_api_key
    
    try:
        # API 키 로드 및 환경변수 설정
        openai_api_key = load_file('api_key.txt')
        os.environ['OPENAI_API_KEY'] = openai_api_key
 
        # CSV 로드 및 텍스트 추출
        file_path = "example.csv"
        df = load_csv(file_path)
        text_data = extract_text_data(df)
 
        # 텍스트 분할
        split_docs = split_texts(text_data)
 
        # 임베딩 모델 및 벡터스토어 설정
        embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectorstore = create_vectorstore(split_docs, embeddings_model)
        save_vectorstore(vectorstore, "faiss_index")
 
        # 벡터스토어 로드 및 QA 체인 설정
        vectorstore = load_vectorstore("faiss_index", embeddings_model)
        qa_chain = setup_qa_chain(vectorstore)
        
        print("Application successfully initialized!")
    except Exception as e:
        print(f"Error during initialization: {e}")
        sys.exit(1)
 
# API 엔드포인트
@app.get('/')
def read_root():
    return {"message": "Welcome to QA Chatbot API!"}
 

@app.get("/chat-bot")
async def chat_endpoint(question: str):
    global qa_chain
    try:
        if qa_chain is None:
            raise HTTPException(status_code=500, detail="QA chain not initialized")
        
        response = qa_chain.invoke({"query": question})
        
        return response["result"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    

# 🔥 PDF 저장 경로 설정
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)  # 📂 폴더가 없으면 자동 생성

# ✅ PDF 생성 설정
PDFKIT_CONFIG = pdfkit.configuration(wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe")
    


# ✅ 1️⃣ 장비 데이터 로드
def load_equipment_data():
    try:
        file_path = "agv_dataframe_version_2.pkl"
        df = joblib.load(file_path)
        print(f"✅ 장비 데이터 로드 완료: {len(df)} 개의 데이터")
        return df
    except Exception as e:
        raise ValueError(f"❌ 장비 데이터 로드 오류: {e}")

# ✅ 2️⃣ 사용자 입력 데이터 모델 (MM-DD 형식)
class EquipmentReportRequest(BaseModel):
    장비_ID: str
    시작_날짜: str  # MM-DD 형식
    종료_날짜: str  # MM-DD 형식

# ✅ 3️⃣ 날짜 변환 (MM-DD → 2024년 YYYY-MM-DD)
def convert_to_2024_date(mm_dd: str):
    return datetime.strptime(f"2024-{mm_dd}", "%Y-%m-%d")

# ✅ 4️⃣ 장비 데이터 조회
def fetch_equipment_data(장비_ID, 시작_날짜, 종료_날짜):
    df = load_equipment_data()

    if 장비_ID not in df["device_id"].unique():
        raise HTTPException(status_code=404, detail=f"❌ 장비 ID '{장비_ID}'를 찾을 수 없습니다.")

    시작 = convert_to_2024_date(시작_날짜)
    종료 = convert_to_2024_date(종료_날짜)

    df["collection_date"] = df["collection_date"].apply(lambda x: f"2024-{x}")
    df["collection_date"] = pd.to_datetime(df["collection_date"], format="%Y-%m-%d")

    filtered_df = df[
        (df["device_id"] == 장비_ID) & 
        (df["collection_date"] >= 시작) & 
        (df["collection_date"] <= 종료)
    ]

    if filtered_df.empty:
        raise HTTPException(status_code=404, detail="❌ 해당 기간의 데이터가 없습니다.")
    
    return filtered_df.to_dict(orient="records")

# ✅ 5️⃣ AI 기반 장비 보고서 생성
def create_ai_equipment_report(장비_ID, 시작_날짜, 종료_날짜, equipment_data):
    data_summary = "\n".join([
        f"- {d['collection_date']} {d['collection_time']}: 온도 {d['ex_temperature']}°C, 습도 {d['ex_humidity']}%, 조도 {d['ex_illuminance']} lx, 상태 {d['state']}"
        for d in equipment_data[:50]
    ])

    report_prompt = PromptTemplate.from_template(
        """당신은 운송 장비 정비 보고서를 작성하는 AI입니다. 
        아래 데이터를 분석하고 요약하여 정비 보고서를 작성하세요.

        # 장비 ID:
        {장비_ID}

        # 기간:
        {시작_날짜} ~ {종료_날짜}

        # 장비 상태 데이터 요약:
        {data_summary}

        # 보고서:
        """
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    chain = LLMChain(llm=llm, prompt=report_prompt)
    report_text = chain.invoke({
        "장비_ID": 장비_ID,
        "시작_날짜": 시작_날짜,
        "종료_날짜": 종료_날짜,
        "data_summary": data_summary
    })
    
    return report_text["text"]


# ✅ 1️⃣ PDF 보고서 생성
def generate_pdf_equipment_report(장비_ID, 시작_날짜, 종료_날짜, equipment_data, report_text):
    pdf_filename = f"{장비_ID}_정비보고서.pdf"
    pdf_path = os.path.join(REPORTS_DIR, pdf_filename)  # ✅ `reports/` 폴더에 저장

    html_content = f"""
    <html>
    <head><meta charset="UTF-8"></head>
    <body>
        <h1>{장비_ID} 정비 보고서</h1>
        <p><strong>운영 기간:</strong> {시작_날짜} ~ {종료_날짜}</p>

        <h2>📌 장비 데이터 요약</h2>
        <ul>
            {''.join([f"<li>{d['collection_date']} {d['collection_time']}: 온도 {d['ex_temperature']}°C, 습도 {d['ex_humidity']}%, 조도 {d['ex_illuminance']} lx, 상태 {d['state']}</li>" for d in equipment_data])}
        </ul>

        <h2>📌 AI 분석 보고</h2>
        <p>{report_text}</p>
    </body>
    </html>
    """

    pdfkit.from_string(html_content, pdf_path, configuration=PDFKIT_CONFIG)
    print(f"✅ PDF 보고서 생성 완료: {pdf_path}")  # 디버깅용 로그 추가
    return pdf_path

from fastapi.responses import FileResponse
import os

# ✅ 장비 보고서 생성 후 PDF 생성하지 않고 `download-report` 엔드포인트에서 생성
@app.post("/generate-equipment-report")
async def generate_equipment_report(request: EquipmentReportRequest):
    try:
        equipment_data = fetch_equipment_data(request.장비_ID, request.시작_날짜, request.종료_날짜)
        report_text = create_ai_equipment_report(request.장비_ID, request.시작_날짜, request.종료_날짜, equipment_data)
        pdf_filename = generate_pdf_equipment_report(request.장비_ID, request.시작_날짜, request.종료_날짜, equipment_data, report_text)
        return {"message": "보고서 생성 완료", "pdf_filename": pdf_filename}
    except Exception as e:
        return {"error": str(e)}

# ✅ 사용자가 다운로드 버튼을 클릭할 때 PDF를 생성하고 반환
# ✅ 파일이 존재하면 반환, 없으면 에러 메시지 출력
@app.get("/download-report/{filename}")
async def download_report(filename: str):
    file_path = os.path.join(REPORTS_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    return FileResponse(file_path, media_type='application/pdf', filename=filename)