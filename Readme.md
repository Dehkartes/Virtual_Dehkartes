이력서 관련 서비스에 AI가 접목되는 것에 아이디어를 얻어 제 이력서를 질의응답 해주는 RAG 챗봇을 개발하였습니다. 이력서 내용을 임베딩하여 벡터 DB에 저장하고 사용자 질문으로 retrieval 하여 RAG를 구현하였습니다.  AWS 프리티어에서 웹 서버를 구동하고, On-premise 환경에서 LG EXAONE 모델을 구동하였습니다.

Tech stack: Hugging Face Transformer, FAISS, FastAPI, Streamlit, Route 53, EC2, ALB

Troubleshooting:

- **토큰 생성 시간 개선(26.6초 → 0.4초)**
    
    문제:
    
    토큰 생성 시간이 오래걸려 사용자 경험 저하
    
    해결:
    
    BitsAndBytesConfig 라이브러리를 통해 모델에 4bit 양자화 적용
    
- **토큰 생성 스트리밍**
    
    문제:
    
    토큰 생성 중 클라이언트에서 시각 피드백이 없어 사용자 경험 저하
    
    해결:
    
    TextIteratorStreamer 라이브러리를 사용하여 모델의 토큰 출력을 실시간으로 웹 클라이언트에 스트리밍
