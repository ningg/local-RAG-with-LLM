import faiss
import numpy as np
import requests
import textwrap
import json
import os
import glob
import time
from pathlib import Path

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

# ========= 2. 从目录读取Markdown文件 =========
def load_md_files_from_directory(directory_path):
    """从指定目录读取所有.md文件"""
    md_files = []
    md_path = Path(directory_path)
    
    if not md_path.exists():
        print(f"目录不存在: {directory_path}")
        return []
    
    if not md_path.is_dir():
        print(f"路径不是目录: {directory_path}")
        return []
    
    # 查找所有.md文件
    md_pattern = md_path / "**/*.md"
    md_files = list(md_path.glob("**/*.md"))
    
    if not md_files:
        print(f"在目录 {directory_path} 中未找到.md文件")
        return []
    
    print(f"找到 {len(md_files)} 个Markdown文件:")
    for file_path in md_files:
        print(f"  - {file_path}")
    
    return md_files

def read_md_file(file_path):
    """读取单个Markdown文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        print(f"读取文件 {file_path} 失败: {e}")
        return None

def load_all_md_content(directory_path):
    """加载目录下所有Markdown文件的内容"""
    md_files = load_md_files_from_directory(directory_path)
    
    if not md_files:
        return ""
    
    all_content = []
    total_files = len(md_files)
    successful_files = 0
    
    # 定义要包含的年份前缀
    valid_years = ['2025']
    
    for file_path in md_files:
        # 检查文件名是否以指定年份开头
        file_name = file_path.name
        if not any(file_name.startswith(year) for year in valid_years):
            print(f"跳过文件: {file_name} (年份不符合要求)")
            continue
            
        content = read_md_file(file_path)
        if content:
            # 添加文件标识信息
            file_info = f"\n\n--- 文件: {file_name} ---\n\n"
            all_content.append(file_info + content)
            successful_files += 1
            print(f"✅ 成功读取: {file_name} ({len(content)} 字符)")
        else:
            print(f"❌ 跳过文件: {file_name} (读取失败)")
    
    print(f"\n总计: {successful_files}/{total_files} 个文件读取成功")
    
    return "\n".join(all_content)

# 配置文档目录路径
DOCS_DIR = "/Users/guoning/ningg/github/ningg.github.com/_posts/blog"  # 请替换为实际的目录路径

# 加载所有Markdown文件内容
print("=== 开始加载Markdown文件 ===")
start_time = time.time()

docs_text = load_all_md_content(DOCS_DIR)
if not docs_text:
    print("未获取到任何文档内容，程序退出")
    exit(1)

load_time = time.time() - start_time
print(f"✅ 文件加载完成，耗时: {load_time:.2f}秒")
print(f"总文档内容长度: {len(docs_text)} 字符")

# 显示内容预览
print("\n=== 文档内容预览 ===")
preview_length = min(800, len(docs_text))
print(docs_text[:preview_length] + ("..." if len(docs_text) > preview_length else ""))
print("=" * 60)

def split_into_chunks(text, chunk_size=500):
    """将文本切分成指定大小的块"""
    text = text.replace("\n", " ")
    return textwrap.wrap(text, chunk_size)

# 将文档文本切分成块
print("=== 开始文本切分 ===")
chunk_start_time = time.time()

chunks = split_into_chunks(docs_text)
chunk_time = time.time() - chunk_start_time
print(f"✅ 文本切分完成，耗时: {chunk_time:.2f}秒")
print(f"文档切分为 {len(chunks)} 个块")

# 生成文档块的向量表示
print("=== 开始生成向量表示 ===")
embedding_start_time = time.time()
chunk_embeddings = []
for i, chunk in enumerate(chunks):
    if i % 10 == 0:  # 每10个块显示一次进度
        print(f"处理进度: {i+1}/{len(chunks)}")
    emb = ollama_embed(chunk)
    chunk_embeddings.append(emb)

embedding_time = time.time() - embedding_start_time
print(f"✅ 向量生成完成，耗时: {embedding_time:.2f}秒")
print(f"平均每个块耗时: {embedding_time/len(chunks):.3f}秒")

# ========= 3. 构建向量库 =========
print("=== 开始构建向量库 ===")
index_start_time = time.time()

dim = len(ollama_embed("测试"))  # 向量维度
index = faiss.IndexFlatL2(dim) 
index.add(np.array(chunk_embeddings))

index_time = time.time() - index_start_time
print(f"✅ 向量库构建完成，耗时: {index_time:.2f}秒")
print(f"向量库维度: {dim}, 向量数量: {len(chunk_embeddings)}")

# ========= 4. 检索 =========
def retrieve_chunks(query, top_k=3):
    """检索与查询最相关的文档块"""
    query_emb = ollama_embed(query)
    D, I = index.search(np.array([query_emb]), top_k)
    return [chunks[i] for i in I[0]]

# ========= 5. 生成答案 =========
def answer_query(query):
    """根据检索到的文档块生成答案"""
    start_time = time.time()
    
    # 检索相关文档块
    retrieve_start = time.time()
    retrieved = retrieve_chunks(query, top_k=3)
    retrieve_time = time.time() - retrieve_start
    
    context = "\n".join(retrieved)

    # 生成答案
    chat_start = time.time()
    prompt = f"""
你是一个文档问答助手。
只能根据以下提供的文档内容回答用户问题，
如果找不到，请回答："抱歉，文档中没有相关内容"。

文档内容:
{context}

用户问题: {query}
请用自然语言总结或改写回答：
"""
    answer = ollama_chat(prompt)
    chat_time = time.time() - chat_start
    
    total_time = time.time() - start_time
    
    print(f"⏱️  检索耗时: {retrieve_time:.2f}秒, 生成耗时: {chat_time:.2f}秒, 总计: {total_time:.2f}秒")
    
    return answer

# ========= 6. 交互式问答 =========
def interactive_qa():
    """交互式问答模式"""
    print("\n" + "="*60)
    print("进入交互式问答模式 (输入 'quit' 或 'exit' 退出)")
    print("="*60)
    
    while True:
        try:
            query = input("\n请输入您的问题: ").strip()
            
            if query.lower() in ['quit', 'exit', '退出', 'q']:
                print("感谢使用，再见！")
                break
            
            if not query:
                print("请输入有效的问题")
                continue
            
            print(f"\n正在处理问题: {query}")
            answer = answer_query(query)
            print(f"答案: {answer}")
            
        except KeyboardInterrupt:
            print("\n\n程序被中断，退出...")
            break
        except Exception as e:
            print(f"处理问题时出错: {e}")

# ========= 7. 主程序 =========
if __name__ == "__main__":
    program_start_time = time.time()
    
    print("=== Markdown文件RAG问答系统 ===")
    print(f"文档目录: {DOCS_DIR}")
    
    # 测试问答
    test_question = "请介绍一下这些文档的主要内容"
    print(f"\n测试问题: {test_question}")
    print("正在生成答案...")
    
    try:
        answer = answer_query(test_question)
        print(f"测试答案: {answer}")
    except Exception as e:
        print(f"测试问答失败: {e}")
    
    # 计算初始化总时间
    init_time = time.time() - program_start_time
    print(f"\n🎯 系统初始化总耗时: {init_time:.2f}秒")
    print("=" * 60)
    
    # 进入交互式问答模式
    interactive_qa()
