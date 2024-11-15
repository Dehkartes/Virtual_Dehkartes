import streamlit as st
import requests

FASTAPI_URL = "http://hsj3925.iptime.org:8000/query"

st.title("허세진 챗봇 이력서")

if "messages" not in st.session_state:
	st.session_state.messages = []

for m in st.session_state["messages"]:
	with st.chat_message(m["role"]):
		st.markdown(m["content"])
		
if user_msg := st.chat_input("What is up?"):
	with st.chat_message("user"):
		st.markdown(user_msg)
		user_prompt = user_msg

	st.session_state.messages.append({"role": "user", "content": user_msg})

	with st.chat_message("assistant"):
		# FastAPI 서버에 요청 보내기
		response = requests.post(FASTAPI_URL, json={"query": user_prompt})
		if response.status_code == 200:
			result = response.json().get("answer", "응답을 가져오는 데 실패했습니다.")
		else:
			result = "FastAPI 서버 요청이 실패했습니다."

		st.markdown(result)
		
	st.session_state.messages.append({
		"role": "assistant", 
		"content": result
	})