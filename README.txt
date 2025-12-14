#环境部署
1. 安装 Python （本项目使用Miniconda + python3.9），并创建环境：
   ```
   conda create -n py39 python=3.9
   conda activate py39
   ```

2. 安装 Ollama  
   - 下载并安装：https://ollama.com/download  
   - 安装完成后保持 Ollama 服务运行（Windows 会自动启动）。

3. 拉取语言模型（在项目根目录 `RAG` 下执行）：
   ```
   cd /path/to/rag/RAG
   ollama pull qwen2.5:1.5b
   ```

4.向量模型与重排序模型（下载到 `SERVER/models/`；如需代理自行配置环境变量）：
    ```
    # 进入项目根目录
    cd /path/to/rag/RAG

    # 拉取向量模型（示例：bge-base-zh-v1.5）
    huggingface-cli download --local-dir SERVER/models/bge-base-zh-v1.5 \
        BAAI/bge-base-zh-v1.5

    # 拉取重排序模型（示例：bge-reranker-large）
    huggingface-cli download --local-dir SERVER/models/bge-reranker-large \
        BAAI/bge-reranker-large
    ```

5. 安装 Python 依赖（在项目根目录执行）：
   ```
   pip install -r requirements.txt
   ```

#启动服务
conda activate py39

##向量化与存储（预构建向量索引）
```
# 进入项目根目录
cd /path/to/rag/RAG

# 预下载模型并构建向量库，完成后退出
python SERVER/app.py --prepare

# 如需强制重建向量索引（覆盖 SERVER/storage/faiss_index/index.faiss & index.pkl）
python SERVER/app.py --prepare --rebuild-index
```
说明：
- 文档来源：`examples/docs/criminal-law/criminal-law/*.md`
- 索引位置：`SERVER/storage/faiss_index/`
- 运行服务时默认会预加载向量库；若仅需 `qa_simple` 且不想预加载，可设置环境变量 `SKIP_VECTOR_PRELOAD=1`。

##后端
cd /path/to/rag/RAG
ollama serve   
ollama run qwen2.5:1.5b  
python SERVER/app.py

##前端
cd /path/to/rag/RAG/UI
python -m http.server 8000

##浏览器访问应用
http://localhost:8000/index.html