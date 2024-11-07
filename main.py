from fastapi import FastAPI, Query
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import faiss
import torch
import os
import requests
from bs4 import BeautifulSoup
import numpy as np

# FastAPI 인스턴스 생성
app = FastAPI()

# 환경 변수 설정
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 웹 페이지 크롤링 함수
def crawl_specific_class(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    specific_content = soup.find(class_="page__inner-wrap")
    return specific_content.get_text(strip=True) if specific_content else "지정된 클래스를 찾을 수 없습니다."

# 임베딩을 위한 모델 및 토크나이저 로드
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
embedding_model = AutoModel.from_pretrained(embedding_model_name)

# 임베딩 함수 정의
def get_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = embedding_model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.numpy()

# 텍스트 벡터화 및 FAISS 인덱스 생성
url = "https://dehkartes.github.io/blog/resume/"
crawled_text = crawl_specific_class(url)
sentences = crawled_text.split(". ")
embeddings = get_embeddings(sentences)

d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)

# 텍스트 생성 모델 (EXAONE) 로드
model_name = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
generation_tokenizer = AutoTokenizer.from_pretrained(model_name)
generation_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    trust_remote_code=True,
    device_map="auto"
)

# Pydantic 모델 정의
class QueryRequest(BaseModel):
    query: str

# RAG 파이프라인 함수
def retrieve_and_generate(query, top_k=5):
    query_embedding = get_embeddings([query])
    distances, indices = index.search(query_embedding, top_k)
    retrieved_texts = [sentences[i] for i in indices[0]]
    
    prompt = (
        "이력서 내용을 기반으로 질문에 간단히 답해라 "
        "\n이력서 내용: " + " ".join(retrieved_texts) + "\n질문: " + query
    )
    
    inputs = generation_tokenizer(prompt, return_tensors="pt").to("cuda")
    output = generation_model.generate(
        **inputs, max_length=2000, max_new_tokens=500, temperature=0.7
    )
    generated_text = generation_tokenizer.decode(output[0], skip_special_tokens=True)
    answer = generated_text[len(prompt):].strip()
    
    return answer

# API 엔드포인트 정의
@app.post("/query")
async def query_endpoint(request: QueryRequest):
    result = retrieve_and_generate(request.query)
    return {"answer": result}