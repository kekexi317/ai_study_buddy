import streamlit as st
import openai

st.set_page_config(page_title="AI学伴 · 轻量版")
st.title("📚 AI学伴（轻量版）")

with st.sidebar:
    provider = st.selectbox("AI平台", ["阿里百炼", "DeepSeek", "智谱GLM"])
    api_key = st.text_input("API Key", type="password")
    if provider == "阿里百炼":
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        model = "qwen-turbo"
    elif provider == "DeepSeek":
        base_url = "https://api.deepseek.com/v1"
        model = "deepseek-chat"
    else:
        base_url = "https://open.bigmodel.cn/api/paas/v4"
        model = "glm-4-flash"

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "你好！我是AI学伴，有什么可以帮助你的？"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("输入你的问题...")
if user_input:
    if not api_key:
        st.error("请先在左侧填入 API Key")
        st.stop()
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                client = openai.OpenAI(api_key=api_key, base_url=base_url)
                resp = client.chat.completions.create(
                    model=model,
                    messages=st.session_state.messages,
                    temperature=0.7
                )
                reply = resp.choices[0].message.content
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
            except Exception as e:
                st.error(f"调用失败：{e}")