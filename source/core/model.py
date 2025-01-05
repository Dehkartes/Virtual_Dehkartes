from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# 임베딩을 위한 모델 및 토크나이저 로드
_embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(_embedding_model_name)
embedding_model = AutoModel.from_pretrained(_embedding_model_name)

# BitsAndBytesConfig 객체를 생성
quantization_config = BitsAndBytesConfig(
	load_in_4bit=True,
	bnb_4bit_compute_dtype=torch.float16
)

# 텍스트 생성 모델 로드
model_name = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
generation_tokenizer = AutoTokenizer.from_pretrained(model_name)
generation_model = AutoModelForCausalLM.from_pretrained(
	model_name,
	trust_remote_code=True,
	quantization_config=quantization_config,
	device_map="auto"
)