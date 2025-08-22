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
    return data["response"].strip()

# ========= 2. 从URL加载文档并转换为Markdown格式 =========
def fetch_docs_from_url(url):
    """从指定URL获取文档内容"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"获取文档失败: {e}")
        return None

def html_to_markdown(html_content):
    """将HTML内容转换为Markdown格式"""
    import re
    
    # 处理标题标签
    text = re.sub(r'<h1[^>]*>(.*?)</h1>', r'# \1\n\n', html_content, flags=re.DOTALL)
    text = re.sub(r'<h2[^>]*>(.*?)</h2>', r'## \1\n\n', text, flags=re.DOTALL)
    text = re.sub(r'<h3[^>]*>(.*?)</h3>', r'### \1\n\n', text, flags=re.DOTALL)
    text = re.sub(r'<h4[^>]*>(.*?)</h4>', r'#### \1\n\n', text, flags=re.DOTALL)
    
    # 处理段落标签
    text = re.sub(r'<p[^>]*>(.*?)</p>', r'\1\n\n', text, flags=re.DOTALL)
    
    # 处理列表标签
    text = re.sub(r'<li[^>]*>(.*?)</li>', r'- \1\n', text, flags=re.DOTALL)
    text = re.sub(r'<ul[^>]*>|</ul>', '', text)
    text = re.sub(r'<ol[^>]*>|</ol>', '', text)
    
    # 处理链接标签
    text = re.sub(r'<a[^>]*href=["\']([^"\']*)["\'][^>]*>(.*?)</a>', r'[\2](\1)', text, flags=re.DOTALL)
    
    # 处理粗体和斜体标签
    text = re.sub(r'<strong[^>]*>(.*?)</strong>', r'**\1**', text, flags=re.DOTALL)
    text = re.sub(r'<b[^>]*>(.*?)</b>', r'**\1**', text, flags=re.DOTALL)
    text = re.sub(r'<em[^>]*>(.*?)</em>', r'*\1*', text, flags=re.DOTALL)
    text = re.sub(r'<i[^>]*>(.*?)</i>', r'*\1*', text, flags=re.DOTALL)
    
    # 处理代码标签
    text = re.sub(r'<code[^>]*>(.*?)</code>', r'`\1`', text, flags=re.DOTALL)
    text = re.sub(r'<pre[^>]*>(.*?)</pre>', r'```\n\1\n```', text, flags=re.DOTALL)
    
    # 处理换行标签
    text = re.sub(r'<br[^>]*>', '\n', text)
    
    # 移除剩余的HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    
    # 处理常见的HTML实体
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    text = text.replace('&apos;', "'")
    
    # 处理换行和段落
    text = re.sub(r'\n\s*\n', '\n\n', text)  # 多个空行合并为两个
    text = re.sub(r'[ \t]+', ' ', text)  # 多个空格合并为一个
    
    # 清理首尾空白
    text = text.strip()
    
    return text

def convert_to_markdown(content, content_type="html"):
    """将内容转换为Markdown格式"""
    if content_type == "html":
        return html_to_markdown(content)
    elif content_type == "text":
        return content
    else:
        # 尝试自动检测内容类型
        if "<" in content and ">" in content:
            return html_to_markdown(content)
        else:
            return content

# 配置文档源URL - 请替换为实际的文档URL
DOCS_URL = "https://ningg.top/about"

# 获取文档内容并转换为Markdown格式
docs_text = fetch_docs_from_url(DOCS_URL)
if docs_text is None:
    # 如果获取失败，使用默认文档作为备用
    docs_text = """未获取到正式内容，填充兜底内容."""
else:
    # 将获取到的内容转换为Markdown格式
    docs_text = convert_to_markdown(docs_text)
    print(f"成功获取文档内容，长度: {len(docs_text)} 字符")
    
    # 显示转换后的Markdown内容预览
    print("\n=== Markdown内容预览 ===")
    preview_length = min(500, len(docs_text))
    print(docs_text[:preview_length] + ("..." if len(docs_text) > preview_length else ""))
    print("=" * 50)

def split_into_chunks(text, chunk_size=200):
    """将文本切分成指定大小的块"""
    text = text.replace("\n", " ")
    return textwrap.wrap(text, chunk_size)

# 将文档文本切分成块
chunks = split_into_chunks(docs_text)

# 生成文档块的向量表示
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
    """检索与查询最相关的文档块"""
    query_emb = ollama_embed(query)
    D, I = index.search(np.array([query_emb]), top_k)
    return [chunks[i] for i in I[0]]

# ========= 5. 生成答案 =========
def answer_query(query):
    """根据检索到的文档块生成答案"""
    retrieved = retrieve_chunks(query, top_k=3)
    context = "\n".join(retrieved)

    prompt = f"""
你是一个文档问答助手。
只能根据以下提供的文档内容回答用户问题，
如果找不到，请回答："抱歉，文档中没有相关内容"。

文档内容:
{context}

用户问题: {query}
请用自然语言总结或改写回答：
"""
    return ollama_chat(prompt)

# ========= 6. 测试 =========
if __name__ == "__main__":
    print("正在从URL加载文档...")
    print(f"文档URL: {DOCS_URL}")
    print(f"文档块数量: {len(chunks)}")
    
    q = "作者是谁？"
    print(f"\n用户提问：{q}")
    print("助手回答：", answer_query(q))
