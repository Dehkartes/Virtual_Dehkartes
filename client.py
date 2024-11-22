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
		# FastAPI 서버에 요청 보내기 (스트리밍 응답 처리)
		response = requests.post(FASTAPI_URL, json={"query": user_prompt}, stream=True)
		result = ""

		# 빈 요소를 생성하여 메시지를 동적으로 업데이트
		placeholder = st.empty()

		if response.status_code == 200:
			for chunk in response.iter_content(chunk_size=1, decode_unicode=True):
				if chunk:
					result += chunk
					placeholder.markdown(result)  # 동적으로 메시지를 업데이트
		else:
			result = "FastAPI 서버 요청이 실패했습니다."

	st.session_state.messages.append({
		"role": "assistant",
		"content": result
	})
