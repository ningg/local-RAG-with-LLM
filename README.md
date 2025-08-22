# local-RAG-with-LLM

本地实现一套 RAG + LLM 系统，实现针对特定信息的小助手.

> Tips: 下面都是 cursor 总结的，我还没验证，不要相信他。
> 
> 我自己，手动进行的操作： [手动操作](./docs/design-manual.md)

## 🚀 快速开始

### 环境要求
- Python 3.11+
- Ollama (本地运行LLM)
- 至少8GB内存 (推荐16GB)

### 安装步骤

1. **克隆项目**
```bash
git clone <your-repo-url>
cd local-RAG-with-LLM
```

2. **安装Python依赖**
```bash
pip install -r requirements.txt
```

3. **安装并启动Ollama**
```bash
# macOS
brew install ollama

# 启动Ollama服务
ollama serve

# 下载模型 (新终端)
ollama pull deepseek-r1:8b
```

4. **运行RAG系统**
```bash
python tech/rd/fassi-ollama-rag-v1.py
```

## 🏗️ 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   用户查询       │    │   RAG系统        │    │   本地LLM        │
│                 │    │                 │    │                 │
│ - 自然语言问题    │◄──►│ - 文档检索       │◄──►│ - Ollama         │
│ - 文档选择       │    │ - 向量化         │    │ - 模型推理        │
│                 │    │ - 上下文构建      │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔧 核心功能

### 1. 文档向量化
- 支持多种文档格式 (PDF, Word, TXT, HTML)
- 使用FAISS进行高效向量检索
- 本地向量数据库存储

### 2. 智能检索
- 语义相似性搜索
- 关键词匹配
- 混合检索策略

### 3. 本地LLM对话
- 基于检索结果生成回答
- 支持多轮对话
- 完全本地化，保护隐私

## 📁 项目结构

```
local-RAG-with-LLM/
├── tech/
│   ├── rd/                    # 后端代码
│   │   └── fassi-ollama-rag-v1.py  # RAG核心实现
│   └── fe/                    # 前端代码 (待开发)
├── docs/                      # 设计文档
├── requirements.txt           # Python依赖
└── README.md                 # 项目说明
```

## 🎯 使用示例

### 基础问答
```python
# 系统会自动检索相关文档片段
q = "系统支持哪些支付方式？"
answer = answer_query(q)
print(answer)
```

### 自定义文档
```python
# 修改docs列表添加你的文档
docs = [
    "你的文档内容1",
    "你的文档内容2",
    # ... 更多文档
]
```

## ⚙️ 配置选项

### 模型配置
```python
# 在代码中修改模型名称
def ollama_embed(text, model="deepseek-r1:8b"):
    # 使用其他可用模型
    pass

def ollama_chat(prompt, model="deepseek-r1:8b"):
    # 使用其他可用模型
    pass
```

### 检索参数
```python
# 调整检索的文档片段数量
def retrieve_chunks(query, top_k=3):  # 修改top_k值
    pass
```

## 🔍 故障排除

### 常见问题

1. **ModuleNotFoundError: No module named 'requests'**
   ```bash
   pip install requests
   ```

2. **Ollama连接失败**
   ```bash
   # 确保Ollama服务正在运行
   ollama serve
   
   # 检查端口11434是否可用
   curl http://localhost:11434/api/tags
   ```

3. **模型下载失败**
   ```bash
   # 重新下载模型
   ollama pull deepseek-r1:8b
   ```

### 性能优化

- 增加内存: 推荐16GB以上
- 使用GPU: 支持CUDA的模型
- 调整chunk_size: 根据文档特点优化

## 🚧 开发计划

- [x] 基础RAG系统
- [x] FAISS向量检索
- [x] Ollama集成
- [ ] Web界面
- [ ] 文档上传功能
- [ ] 多模型支持
- [ ] 对话历史管理

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License

## 🙏 致谢

- [Ollama](https://ollama.ai/) - 本地LLM运行
- [FAISS](https://github.com/facebookresearch/faiss) - 向量检索
- [DeepSeek](https://www.deepseek.com/) - 开源模型
