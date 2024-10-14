import streamlit as st
import random
import time

st.title("Simple Chat")

def get_response():
    responses = [
        "안녕하세요! 무엇을 도와드릴까요?",
        "오늘도 좋은 하루 되세요!",
        "저는 당신의 질문에 최선을 다해 답변드리겠습니다."
    ]
    return random.choice(responses)

def stream_response(text):
    for char in text:
        with st.chat_message("assistant"):
            st.markdown(char)
            time.sleep(0.05)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("메시지를 입력하세요"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = get_response()
    # stream_response(response)
    st.session_state.messages.append({"role": "assistant", "content": response})