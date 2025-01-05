from transformers import TextIteratorStreamer
import faiss
import torch
import os
import requests
from core import model

from bs4 import BeautifulSoup
import configparser
from threading import Thread
from huggingface_hub import login

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
tokenizer = model.tokenizer
embedding_model = model.embedding_model

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

# 텍스트 생성 모델 로드
generation_tokenizer = model.generation_tokenizer
generation_model = model.generation_model

# RAG 파이프라인 함수
def pipeLine(query, top_k=5):
	print(query)
	query_embedding = get_embeddings([query])
	distances, indices = index.search(query_embedding, top_k)
	retrieved_texts = [sentences[i] for i in indices[0]]

	prompt = [
		{
			"role": "system", 
			"content": (
				"다음 절차에 따라 질문에 이력서 내용을 기반으로 응답해라\n"
				"1. 질문이 이력서에 있는 내용인지 판단해라, 이력서에 없는 내용이면 \"이력서에 없는 내용입니다.\"라고 해라.\n"
				"2. 이력서에 있는 내용이면 질문에 단답형으로 응답해라.\n"
				"예시(요청: 이메일이 뭐야? 응답: hsj3925@gmail.com 입니다.)\n"
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
