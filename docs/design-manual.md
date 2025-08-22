手动学习版本

## 1.背景：

业务场景：

1.有几十篇 FAQ 的帮助文档 
2.需要做一个小助手，只针对文档中内容，进行答疑，不能超出文档范围


## 2.技术方案


技术方案：本地向量数据库 + 本地大模型（Ollama）

本地向量库：FAISS


1.conda 创建独立的环境 localRAG
2.安装 本地向量库：FAISS，参考 https://faiss.ai/

```
$ conda install -c pytorch faiss-cpu
```
* FAISS 的使用 demo： https://github.com/facebookresearch/faiss/wiki/Getting-started

3.安装 本地大模型：Ollama，参考 https://ningg.top/ai-series-deepseek-intro-202502/

* Ollama 的 API ： https://github.com/ollama/ollama 的 `REST API` 部分


## 3. 代码实现

| 文件名       | 作用简介        |
|-------------|----------------|
| [fassi-ollama-rag-v1.py](../tech/rd/fassi-ollama-rag-v1.py)        | 最基础的RAG实现：<br>1. 手动内置FAQ文档片段<br>2. 文档切分为块并向量化<br>3. 使用FAISS本地向量库检索<br>4. 通过Ollama本地大模型生成答案<br>适合快速本地测试和小规模文档。|
| [fassi-ollama-rag-v3.py](../tech/rd/fassi-ollama-rag-v3.py)        | 进阶RAG实现：<br>1. 支持从URL自动抓取文档（如网页）<br>2. 自动将HTML内容转为Markdown格式<br>3. 文档切分、向量化、检索流程自动化<br>4. 适合动态加载和处理较大规模文档。|
| [fassi-ollama-rag-v5.py](../tech/rd/fassi-ollama-rag-v5.py)        | 目录文档RAG实现：<br>1. 支持从指定目录自动读取所有Markdown文件<br>2. 自动过滤指定年份的文档（如2025年）<br>3. 文档切分、向量化、检索流程完全自动化<br>4. 提供交互式问答界面，适合批量处理本地文档集合。|
