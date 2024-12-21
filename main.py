from fastapi import FastAPI, Request, status
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, TextIteratorStreamer, BitsAndBytesConfig
import faiss
import torch
import os
import requests
from allowIP import ALLOWED_IPS
from bs4 import BeautifulSoup
import configparser
from threading import Thread
from huggingface_hub import login

# FastAPI 인스턴스 생성
app = FastAPI()

config = configparser.ConfigParser()
config.read("secret.ini")
login(config["TOKEN"]["huggingface"])

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

# BitsAndBytesConfig 객체를 생성
quantization_config = BitsAndBytesConfig(
	load_in_4bit=True,
	bnb_4bit_compute_dtype=torch.float16
)

# 텍스트 생성 모델 로드
model_name = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
generation_tokenizer = AutoTokenizer.from_pretrained(model_name)
generation_model = AutoModelForCausalLM.from_pretrained(
	model_name,
	trust_remote_code=True,
	quantization_config=quantization_config,
	device_map="auto"
)

class QueryRequest(BaseModel):
	query: str

# RAG 파이프라인 함수
def retrieve_and_generate(query, top_k=5):
	print(query)
	query_embedding = get_embeddings([query])
	distances, indices = index.search(query_embedding, top_k)
	retrieved_texts = [sentences[i] for i in indices[0]]

	prompt = [
		{
			"role": "system", 
			"content": (
				"요청에 이력서 내용을 우선으로 단답형으로 응답해라. 단 이력서에 없는 내용은 모른다고 해라\n"
				"예시(요청: 이름이 뭐야? 응답: 허세진 입니다.)"
				"이력서 내용: " + " ".join(retrieved_texts))
		},
		{"role": "user", "content": query}
	]

	inputs = generation_tokenizer.apply_chat_template(
		prompt,
		tokenize=True,
		add_generation_prompt=True,
		return_tensors="pt"
	).to("cuda")

	streamer = TextIteratorStreamer(generation_tokenizer, True, skip_special_tokens=True)

	# 텍스트 생성을 스트리밍 방식으로 처리하는 generator
	def text_generator():
		generation_kwargs = {
			"inputs": inputs,
			"max_new_tokens": 500,
			"temperature": 1.0,
			"do_sample": False,
			"streamer": streamer
		}

		thread = Thread(target=generation_model.generate, kwargs=generation_kwargs)
		thread.start()
		return streamer

	return text_generator()

@app.middleware("http")
async def ip_filter_middleware(request: Request, call_next):
	client_ip = request.client.host
	if client_ip not in ALLOWED_IPS:
		return Response(status_code=status.HTTP_403_FORBIDDEN)
	response = await call_next(request)
	return response

# API 엔드포인트 정의
@app.post("/query")
async def query_endpoint(request: QueryRequest):
	return StreamingResponse(retrieve_and_generate(request.query), media_type="text/plain")