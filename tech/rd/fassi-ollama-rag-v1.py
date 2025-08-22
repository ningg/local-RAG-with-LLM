import faiss
import numpy as np
import requests
import textwrap
import json

# ========= 1. Ollama API 封装 =========
OLLAMA_API = "http://localhost:11434/api"

def ollama_embed(text, model="deepseek-r1:8b"):
    """调用 Ollama embedding 接口"""
    resp = requests.post(f"{OLLAMA_API}/embed", json={"model": model, "input": text})
    data = resp.json()
    # print('ollama_embed result data:', data)
    return np.array(data["embeddings"][0], dtype="float32")

def ollama_chat(prompt, model="deepseek-r1:8b"):
    """调用 Ollama Chat 接口"""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    resp = requests.post(f"{OLLAMA_API}/generate", json=payload)
    data = resp.json()
    # print('ollama_chat result data:', data)
    return data["response"].strip()

# ========= 2. 加载文档并切分 =========
docs = [
    """我们的系统支持多种支付方式，包括支付宝、微信支付和银行卡支付。
    在支付过程中如遇到问题，可以联系客服协助处理。""",
    
    """用户可以通过点击登录页面的“忘记密码”，
    使用注册邮箱或手机号进行验证，即可重置密码。""",
    
    """完成订单后，您可以在“个人中心-订单管理”页面申请电子发票。
    系统将自动开具并发送到您的邮箱。"""
]

def split_into_chunks(text, chunk_size=200):
    text = text.replace("\n", " ")
    return textwrap.wrap(text, chunk_size)

chunks = []
for doc in docs:
    processed_doc = split_into_chunks(doc)
    chunks.extend(processed_doc)


chunk_embeddings = []
for chunk in chunks:
    emb = ollama_embed(chunk)
    chunk_embeddings.append(emb)

# ========= 3. 构建向量库 =========
dim = len(ollama_embed("测试"))  # 向量维度
index = faiss.IndexFlatL2(dim) 

index.add(np.array(chunk_embeddings))

# ========= 4. 检索 =========
def retrieve_chunks(query, top_k=3):
    query_emb = ollama_embed(query)
    D, I = index.search(np.array([query_emb]), top_k)
    return [chunks[i] for i in I[0]]

# ========= 5. 生成答案 =========
def answer_query(query):
    retrieved = retrieve_chunks(query, top_k=3)
    context = "\n".join(retrieved)

    prompt = f"""
你是一个文档问答助手。
只能根据以下提供的文档内容回答用户问题，
如果找不到，请回答：“抱歉，文档中没有相关内容”。

文档内容:
{context}

用户问题: {query}
请用自然语言总结或改写回答：
"""
    return ollama_chat(prompt)

# ========= 6. 测试 =========
if __name__ == "__main__":
    # q = "系统支持哪些支付方式？"
    q = "你是谁？"
    print("用户提问：", q)
    print("助手回答：", answer_query(q))
