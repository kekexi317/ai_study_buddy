import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import streamlit as st
import openai
import PyPDF2
import io
from sentence_transformers import SentenceTransformer
import chromadb
import uuid

# 加载文字转数字指纹的小模型（首次运行会下载，耐心等）
@st.cache_resource
def load_embedding():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding()

# 创建或连接向量数据库（小书架）
@st.cache_resource
def init_db():
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_or_create_collection(name="docs")

collection = init_db()

# 处理PDF：提取文字并切成小段
def parse_pdf(file_bytes):
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    # 每500字切成一段
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    return chunks

# 把段落存入小书架
def add_chunks(chunks):
    ids = [str(uuid.uuid4()) for _ in chunks]
    embeddings = embedding_model.encode(chunks).tolist()
    collection.add(ids=ids, embeddings=embeddings, documents=chunks)
    return len(chunks)

# 根据问题从小书架上找最相关的3段
def search(query):
    query_emb = embedding_model.encode([query]).tolist()
    results = collection.query(query_emb, n_results=3)
    return results['documents'][0] if results['documents'] else []

# ---------- 页面UI ----------
st.set_page_config(page_title="AI学伴", layout="wide")
st.title("📚 AI学伴 · 你的私人学习助手")

# 侧边栏：设置API Key和上传文件
with st.sidebar:
    st.header("⚙️ 设置")
    provider = st.selectbox("选择AI平台", ["阿里百炼", "DeepSeek", "智谱GLM"])
    api_key = st.text_input("API Key", type="password", help="免费获取：阿里百炼新用户送大量额度")
    
    if provider == "阿里百炼":
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        model = "qwen-turbo"
    elif provider == "DeepSeek":
        base_url = "https://api.deepseek.com/v1"
        model = "deepseek-chat"
    else:
        base_url = "https://open.bigmodel.cn/api/paas/v4"
        model = "glm-4-flash"
    
    uploaded_file = st.file_uploader("📄 上传PDF（让AI记住它）", type=["pdf"])
    if uploaded_file:
        with st.spinner("解析中..."):
            chunks = parse_pdf(uploaded_file.read())
            count = add_chunks(chunks)
        st.success(f"✅ 已存入 {count} 个知识点！可以提问了")
    
    st.info(f"📚 知识库现有 {collection.count()} 个段落")

# 聊天界面
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "上传PDF后，问我关于文档的问题吧！"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 接收用户输入
user_input = st.chat_input("输入你的问题...")
if user_input:
    if not api_key:
        st.error("请先在左侧填入API Key")
        st.stop()
    
    # 检索相关段落
    context_chunks = search(user_input)
    context = "\n\n".join(context_chunks) if context_chunks else "（没有找到相关文档内容）"
    
    # 构造系统提示
    sys_prompt = f"""你是一个学习助手。请根据以下文档片段回答用户问题。如果文档中没有，就说“文档里没提到”。
    
文档片段：
{context}
"""
    
    # 准备API消息
    msgs = [{"role": "system", "content": sys_prompt}]
    msgs.extend(st.session_state.messages)
    msgs.append({"role": "user", "content": user_input})
    
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # 调用AI
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                client = openai.OpenAI(api_key=api_key, base_url=base_url)
                resp = client.chat.completions.create(
                    model=model,
                    messages=msgs,
                    temperature=0.5
                )
                reply = resp.choices[0].message.content
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
            except Exception as e:
                st.error(f"调用失败：{e}")