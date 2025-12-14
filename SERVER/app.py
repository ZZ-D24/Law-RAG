from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request

import requests
from pathlib import Path
from typing import List
from urllib.parse import urlparse

from flask import Flask, request
from flask_cors import CORS
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from openai import OpenAI

try:
    import jieba
except Exception:
    jieba = None

app = Flask(__name__)
CORS(app)
app.config["JSON_AS_ASCII"] = False  # 返回中文时不转义为 \\uXXXX
app.config["JSONIFY_MIMETYPE"] = "application/json; charset=utf-8"

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR.parent / "examples" / "docs" / "criminal-law" / "criminal-law"
VECTOR_DIR = BASE_DIR / "storage" / "faiss_index"

# 强制离线：仅允许本地路径，避免自动拉取网络模型
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

DEFAULT_MODEL_DIR = BASE_DIR / "models"
EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", str(DEFAULT_MODEL_DIR / "bge-m3"))
RERANKER_MODEL = os.getenv("HF_RERANKER_MODEL", str(DEFAULT_MODEL_DIR / "bge-reranker-base"))
TOP_K = 5
GEN_MODEL_NAME = os.getenv("OPENAI_MODEL", "qwen2.5:1.5b")
# 本地 OpenAI 兼容推理（如 Ollama/LM Studio）无需真实密钥，使用占位符即可
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "ollama")
OPENAI_BASE_URL = os.getenv("OPENAI_API_BASE", "http://127.0.0.1:11434/v1")
ENABLE_LOCAL_LLM_AUTOSTART = os.getenv("ENABLE_LOCAL_LLM_AUTOSTART", "true").lower() in {
    "1",
    "true",
    "yes",
}
LOCAL_LLM_WAIT_SECONDS = int(os.getenv("LOCAL_LLM_WAIT_SECONDS", "10"))
MAX_NEW_TOKENS = 512
ALLOW_ON_DEMAND_BUILD = False  # 默认不在请求时临时构建，需预先准备

_vector_store: FAISS | None = None
_embeddings: HuggingFaceEmbeddings | None = None
_qa_client: OpenAI | None = None
_bm25_retriever: BM25Retriever | None = None
_reranker: HuggingFaceCrossEncoder | None = None
_documents_cache: List[Document] | None = None
_llm_service_checked: bool = False

STOP_WORDS = {
    "吗",
    "呢",
    "嘛",
    "吧",
    "啊",
    "是否",
    "是不是",
    "可以",
    "可否",
    "么",
    "什么",
    "如何",
    "怎么",
    "怎样",
    "请问",
    "请",
    "的",
    "了",
    "在",
}
PUNCT_PATTERN = re.compile(r"[，。！？、；：,.!?;:【】（）()《》“”\"'‘’\[\]{}<>]")

# 匹配刑法条文起始行，例如：
# **第一百零二条**　xxxx
# 第一百二十条之一　xxxx
ARTICLE_PATTERN = re.compile(
    r"^\s*\*{0,2}(第[一二三四五六七八九十百千万0-9零〇]+条(?:之[一二三四五六七八九十0-9]+)?)\*{0,2}\s*(.*)"
)


def _normalize_query(text: str) -> str:
    """基础清洗：去标点、压缩空格。"""
    if not text:
        return ""
    cleaned = PUNCT_PATTERN.sub(" ", text)
    return " ".join(cleaned.split())


def _segment_query(text: str) -> dict:
    """
    中文分词并保留核心关键词，兼容无 jieba 场景。
    优先用搜索分词保证细粒度（如“拐卖儿童犯罪”能得到“拐卖”）。
    返回 normalized/keywords/tokens 供检索使用。
    """
    normalized = _normalize_query(text)
    if not normalized:
        return {"normalized": "", "keywords": "", "tokens": []}

    if jieba:
        # 普通分词 + 搜索分词，去重保序，能拆出关键动词
        raw_tokens = jieba.lcut(normalized) + jieba.lcut_for_search(normalized)
        seen = set()
        tokens = []
        for t in raw_tokens:
            t = t.strip()
            if not t or t in seen or t in STOP_WORDS or len(t) <= 1:
                continue
            seen.add(t)
            tokens.append(t)
    else:
        # 无 jieba 时按中文连续片段+双字切分，兼顾“拐卖”这类动词
        segments = re.findall(r"[\u4e00-\u9fff]+|\w+", normalized)
        seen = set()
        tokens = []
        for seg in segments:
            if seg in STOP_WORDS or len(seg) <= 1:
                continue
            if seg not in seen:
                seen.add(seg)
                tokens.append(seg)
            # 针对较长中文词，补充双字窗口
            if re.fullmatch(r"[\u4e00-\u9fff]+", seg) and len(seg) >= 2:
                for i in range(len(seg) - 1):
                    bi = seg[i : i + 2]
                    if bi not in STOP_WORDS and bi not in seen:
                        seen.add(bi)
                        tokens.append(bi)

    keywords = " ".join(tokens)
    return {"normalized": normalized, "keywords": keywords, "tokens": tokens}


def _load_embeddings() -> HuggingFaceEmbeddings:
    """延迟加载本地嵌入模型，避免依赖外部额度。"""
    global _embeddings
    if _embeddings:
        return _embeddings
    model_path = Path(EMBEDDING_MODEL)
    if not model_path.exists():
        raise RuntimeError(
            f"未找到本地嵌入模型：{model_path}。请手动下载后，将 HF_EMBEDDING_MODEL 指向解压目录。"
        )
    _embeddings = HuggingFaceEmbeddings(model_name=str(model_path))
    return _embeddings


def _normalize_article_id(article_no: str) -> str:
    """将条号转换为 chunk_id 的安全标识。"""
    safe = re.sub(r"[^\w一-龥]+", "-", article_no).strip("-")
    return safe or "article"


def _split_law_articles(md_path: Path) -> List[Document]:
    """
    按“第××条”粒度拆分刑法 Markdown，确保一条法条对应一个 Document。
    保留章/节等标题到 metadata，便于前端展示与过滤。
    """
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    header_stack: dict[str, str] = {}
    documents: List[Document] = []

    current_article_no = ""
    current_article_title = ""
    current_headers: dict[str, str] = {}
    buffer: List[str] = []

    def flush():
        nonlocal buffer, current_article_no, current_article_title, current_headers
        if current_article_no and buffer:
            content = "\n".join(buffer).strip()
            if content:
                metadata = {
                    "source": md_path.name,
                    "article_no": current_article_no,
                    "article_title": current_article_title,
                    "chunk_id": f"{md_path.stem}-{_normalize_article_id(current_article_no)}",
                }
                # 将当前章/节等层级写入元数据
                metadata.update(current_headers)
                documents.append(Document(page_content=content, metadata=metadata))
        buffer = []
        current_article_no = ""
        current_article_title = ""
        current_headers = {}

    for line in lines:
        # 先捕获标题层级（#、## 等），用于元数据
        heading = re.match(r"^\s*(#{1,6})\s*(.+?)\s*$", line)
        if heading:
            level = len(heading.group(1))
            text = heading.group(2).strip()
            # 保留当前层级及以上，去掉更深层级
            header_stack = {k: v for k, v in header_stack.items() if int(k[1:]) < level}
            header_stack[f"H{level}"] = text
            continue

        # 匹配条文起始
        match = ARTICLE_PATTERN.match(line)
        if match:
            flush()
            current_article_no = match.group(1)
            current_article_title = match.group(2).strip("　 ").strip()
            current_headers = dict(header_stack)
            buffer.append(line)
            continue

        if current_article_no:
            buffer.append(line)

    flush()
    return documents


def _load_markdown_documents() -> List[Document]:
    """
    读取指定目录下的所有 Markdown 文件，按标题+固定长度双重切片，避免过长片段。
    metadata 中保留层级、来源及 chunk_id 方便去重与调试。
    """
    global _documents_cache
    if _documents_cache is not None:
        return _documents_cache

    documents: List[Document] = []
    if not DOCS_DIR.exists():
        raise FileNotFoundError(f"文档目录不存在: {DOCS_DIR}")

    for md_path in sorted(DOCS_DIR.glob("*.md")):
        articles = _split_law_articles(md_path)
        if not articles:
            raise RuntimeError(f"未能在文件中识别到法条: {md_path}")
        documents.extend(articles)

    _documents_cache = documents
    return documents


def _build_or_load_vector_store(
    allow_build: bool = ALLOW_ON_DEMAND_BUILD, force_rebuild: bool = False
) -> FAISS:
    """
    检查向量库是否存在，存在则加载。
    未找到时仅在 allow_build=True 或 force_rebuild=True 时重新构建，
    避免首个请求才触发下载/构建。
    """
    global _vector_store
    if _vector_store and not force_rebuild:
        return _vector_store

    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    index_file = VECTOR_DIR / "index.faiss"
    store_file = VECTOR_DIR / "index.pkl"

    if force_rebuild:
        if index_file.exists():
            index_file.unlink()
        if store_file.exists():
            store_file.unlink()

    if index_file.exists() and store_file.exists():
        try:
            _vector_store = FAISS.load_local(
                str(VECTOR_DIR),
                _load_embeddings(),
                allow_dangerous_deserialization=True,
            )
            return _vector_store
        except Exception as exc:  # 索引可能与新向量维度不兼容
            if not allow_build and not force_rebuild:
                raise RuntimeError(f"加载向量库失败，请重建索引: {exc}") from exc

    if not allow_build:
        raise RuntimeError(
            "未找到向量库，请先运行 `python SERVER/app.py --prepare` 预构建索引。"
        )

    documents = _load_markdown_documents()
    _vector_store = FAISS.from_documents(documents, _load_embeddings())
    _vector_store.save_local(str(VECTOR_DIR))
    return _vector_store


def _load_bm25_retriever() -> BM25Retriever:
    """基于相同切片构建 BM25，用于关键词补充召回。"""
    global _bm25_retriever
    if _bm25_retriever:
        return _bm25_retriever
    documents = _load_markdown_documents()
    _bm25_retriever = BM25Retriever.from_documents(documents)
    _bm25_retriever.k = TOP_K * 3
    return _bm25_retriever


def _load_reranker() -> HuggingFaceCrossEncoder | None:
    """交叉编码器重排，显著提升相关性。加载失败时回退为向量得分。"""
    global _reranker
    if _reranker:
        return _reranker
    model_path = Path(RERANKER_MODEL)
    if not model_path.exists():
        print(
            f"[RERANK] 未找到本地重排模型：{model_path}，将跳过重排，仅使用向量/BM25。",
            file=sys.stderr,
        )
        _reranker = None
        return _reranker
    try:
        _reranker = HuggingFaceCrossEncoder(model_name=str(model_path))
    except Exception as exc:
        print(f"[RERANK] 加载失败，跳过重排: {exc}", file=sys.stderr)
        _reranker = None
    return _reranker


def _ping_llm_service(models_url: str, timeout: float = 2.0) -> bool:
    try:
        req = urllib.request.Request(models_url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 500
    except Exception:
        return False


def _ensure_local_llm_service():
    """
    在默认本地地址上检测 LLM 服务，必要时尝试自动启动 Ollama。
    仅在 base_url 指向 127.0.0.1/localhost 时启用。
    """
    global _llm_service_checked
    if _llm_service_checked:
        return

    base_url = OPENAI_BASE_URL or ""
    parsed = urlparse(base_url)
    is_local = parsed.hostname in {"127.0.0.1", "localhost"}
    models_url = base_url.rstrip("/") + "/models"

    # 已经通了就直接返回
    if _ping_llm_service(models_url):
        _llm_service_checked = True
        return

    if not is_local:
        # 自定义远端或局域网地址，用户自行保证可用
        return

    if not ENABLE_LOCAL_LLM_AUTOSTART:
        raise RuntimeError(
            f"本地 LLM 服务未响应：{models_url}。请先运行 `ollama serve` 或在 LM Studio 开启 OpenAI 兼容服务。"
        )

    if shutil.which("ollama") is None:
        raise RuntimeError(
            "未找到 ollama 可执行文件，无法自动启动。请先安装并手动运行 `ollama serve`。"
        )

    try:
        # 后台启动 Ollama 服务
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"自动启动 Ollama 失败，请手动运行 `ollama serve`。错误: {exc}") from exc

    # 等待服务就绪
    deadline = time.time() + LOCAL_LLM_WAIT_SECONDS
    while time.time() < deadline:
        time.sleep(0.8)
        if _ping_llm_service(models_url):
            _llm_service_checked = True
            return

    raise RuntimeError(
        f"已尝试启动 Ollama，但 {models_url} 仍不可用。请检查端口/防火墙后重试。"
    )


def _retrieve_documents(question: str):
    """
    多路检索：向量 + BM25，并使用交叉编码器重排。
    返回格式同时附带各阶段分数，便于前端展示。
    """
    parsed_query = _segment_query(question)
    search_query = parsed_query["keywords"] or parsed_query["normalized"] or question.strip()
    vector_query = (
        f"{question} {search_query}"
        if search_query and search_query not in question
        else (search_query or question)
    )
    bm25_query = search_query or question

    store = _build_or_load_vector_store()
    bm25 = _load_bm25_retriever()

    # 适当扩大候选数量，后续再裁剪，避免关键词命中被截断
    vector_hits = store.similarity_search_with_score(vector_query, k=TOP_K * 6)
    bm25_hits = bm25.get_relevant_documents(bm25_query)[: TOP_K * 4]

    candidates = {}
    for doc, score in vector_hits:
        chunk_id = doc.metadata.get("chunk_id") or id(doc)
        candidates[chunk_id] = {
            "doc": doc,
            "vector_score": float(score),
            "bm25_score": None,
        }
    for doc in bm25_hits:
        chunk_id = doc.metadata.get("chunk_id") or id(doc)
        if chunk_id not in candidates:
            candidates[chunk_id] = {"doc": doc, "vector_score": None, "bm25_score": 1.0}
        else:
            candidates[chunk_id]["bm25_score"] = 1.0

    docs_for_rank = list(candidates.values())
    reranker = _load_reranker()
    if reranker:
        pairs = [[question, item["doc"].page_content] for item in docs_for_rank]
        try:
            scores = reranker.predict(pairs)
        except Exception:
            scores = None
    else:
        scores = None

    for item, score in zip(docs_for_rank, scores or []):
        item["rerank_score"] = float(score)
    if scores is None:
        for item in docs_for_rank:
            base = item.get("vector_score") or 0.0
            bonus = 0.2 if item.get("bm25_score") else 0.0
            item["rerank_score"] = base + bonus

    ranked = sorted(docs_for_rank, key=lambda x: x.get("rerank_score", 0), reverse=True)

    # 关键词命中优先：若片段包含分词关键词，强制前置，确保精确词不被截断
    query_terms = parsed_query["tokens"] or ([parsed_query["normalized"]] if parsed_query["normalized"] else [])
    if not query_terms and question.strip():
        query_terms = [question.strip()]
    if query_terms:
        hits = []
        others = []
        for item in ranked:
            content = item["doc"].page_content
            if any(term in content for term in query_terms):
                hits.append(item)
            else:
                others.append(item)
        ranked = hits + others

    return ranked[:TOP_K]


def _load_llm_pipeline():
    """
    加载 OpenAI 兼容客户端
    API Key/自定义 Base URL 从环境变量读取，可留空占位。
    """
    global _qa_client
    if _qa_client:
        return _qa_client
    _ensure_local_llm_service()
    # 兼容本地 Ollama/LM Studio 的无密钥场景，默认占位符并兜底本地端口
    api_key = OPENAI_API_KEY or "ollama"
    base_url = OPENAI_BASE_URL or "http://localhost:11434/v1"
    _qa_client = OpenAI(api_key=api_key, base_url=base_url)
    # 便于排查：打印一次实际使用的 LLM 入口
    print(f"[LLM] base_url={base_url} model={GEN_MODEL_NAME}")
    return _qa_client


def _call_llm_via_http(messages: list[dict], temperature: float = 0.0) -> str:
    """
    直接调用本地 OpenAI 兼容接口，绕过 openai SDK，减少 502 模糊错误。
    """
    base_url = OPENAI_BASE_URL.rstrip("/")
    url = f"{base_url}/chat/completions"
    payload = {
        "model": GEN_MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {OPENAI_API_KEY}"}
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
    except Exception as exc:
        raise RuntimeError(f"LLM HTTP 请求失败: {exc}") from exc
    if resp.status_code != 200:
        raise RuntimeError(f"LLM HTTP 非 200: {resp.status_code}, body={resp.text[:800]}")
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        raise RuntimeError(f"LLM 返回解析失败: {data}") from exc


def prepare_assets(force_rebuild: bool = False):
    """
    准备向量库，确保线上模型与嵌入服务可用。
    """
    _build_or_load_vector_store(allow_build=True, force_rebuild=force_rebuild)
    _load_llm_pipeline()


def _format_knowledge_base(docs: List[Document]) -> str:
    """将检索到的文档片段串联为 prompt 段落。"""
    parts = []
    for i, doc in enumerate(docs, start=1):
        src = doc.metadata.get("source", "unknown")
        header = ", ".join(
            f"{k}:{v}" for k, v in doc.metadata.items() if k != "source"
        )
        chunk_id = doc.metadata.get("chunk_id", f"chunk-{i}")
        parts.append(f"[{i}] ID:{chunk_id} 来源:{src} | {header}\n{doc.page_content}")
    return "\n\n".join(parts)


def _build_prompt(knowledge_base: str, user_query: str) -> str:
    return (
        "你是一名法律知识评估助手。你拥有一个外部知识库，知识库内容来自《中华人民共和国刑法》的结构化抽取与图谱化处理。\n\n"
        "请遵循以下回答原则：\n"
        "1. 结论：必须基于知识库片段，不得使用任何其他知识。\n"
        "2. 适用法条：引用检索到的条文原文，格式“【编号】【标题】+关键句”。\n"
        "3. 说明：指出满足条文要件的关键事实，并提示需要进一步查证的要件（如主观故意、因果关系）。\n"
        "4. 其它：若未命中特定条文或章节，请不要补充常识，不要提示“知识库未命中”。\n\n"
        "知识库片段：\n"
        f"{knowledge_base}\n\n"
        "问题是：\n"
        f"“{user_query}”\n\n"
        "请按照以下格式回答问题：\n"
        "【结论】行为是否违法，是否构成犯罪\n"
        "【适用法条】列出条文编号 + 条文标题\n"
        "【说明】解释为什么符合（或不符合）法条\n"
    )


def _build_direct_prompt(user_query: str) -> str:
    """简单直连 LLM 的提示词，不依赖知识库。"""
    return (
        f"“{user_query}”\n\n"
    )


def json_response(payload, status: int = 200):
    """统一 JSON 输出，确保中文不转义。"""
    return app.response_class(
        response=json.dumps(payload, ensure_ascii=False),
        status=status,
        mimetype="application/json",
    )


@app.route("/health", methods=["GET"])
def health_check():
    return json_response({"status": "ok"})


@app.route("/query", methods=["POST"])
def query():
    payload = request.get_json(silent=True) or {}
    question = payload.get("question") or payload.get("query")
    if not question:
        return json_response({"error": "缺少参数 question"}, status=400)

    try:
        ranked = _retrieve_documents(question)
    except Exception as exc:
        return json_response({"error": f"向量库不可用: {exc}"}, status=503)

    formatted = [
        {
            "text": item["doc"].page_content,
            "metadata": item["doc"].metadata,
            "vector_score": item.get("vector_score"),
            "bm25_score": item.get("bm25_score"),
            "rerank_score": item.get("rerank_score"),
        }
        for item in ranked
    ]
    return json_response({"question": question, "results": formatted})


@app.route("/qa", methods=["POST"])
def qa():
    payload = request.get_json(silent=True) or {}
    question = payload.get("question") or payload.get("query")
    if not question:
        return json_response({"error": "缺少参数 question"}, status=400)

    try:
        ranked = _retrieve_documents(question)
    except Exception as exc:
        return json_response({"error": f"向量库不可用: {exc}"}, status=503)
    docs = [item["doc"] for item in ranked]
    knowledge_base = _format_knowledge_base(docs)
    prompt = _build_prompt(knowledge_base, question)

    try:
        answer = _call_llm_via_http([{"role": "user", "content": prompt}], temperature=0)
    except Exception as exc:
        print("[LLM][qa] 调用失败", exc, file=sys.stderr)
        return json_response({"error": f"LLM 调用失败: {exc}"}, status=502)

    return json_response(
        {
            "question": question,
            "results": [
                {
                    "text": item["doc"].page_content,
                    "metadata": item["doc"].metadata,
                    "vector_score": item.get("vector_score"),
                    "bm25_score": item.get("bm25_score"),
                    "rerank_score": item.get("rerank_score"),
                }
                for item in ranked
            ],
            "answer": answer,
        }
    )


@app.route("/qa_simple", methods=["POST"])
def qa_simple():
    payload = request.get_json(silent=True) or {}
    question = payload.get("question") or payload.get("query")
    if not question:
        return json_response({"error": "缺少参数 question"}, status=400)

    prompt = _build_direct_prompt(question)
    try:
        answer = _call_llm_via_http([{"role": "user", "content": prompt}], temperature=0)
    except Exception as exc:
        print("[LLM][qa_simple] 调用失败", exc, file=sys.stderr)
        return json_response({"error": f"LLM 调用失败: {exc}"}, status=502)

    return json_response({"question": question, "answer": answer})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG 服务与资源准备工具")
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="预下载模型并构建向量库，完成后退出",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="强制重建向量索引（会覆盖原索引）",
    )
    parser.add_argument(
        "--allow-build",
        action="store_true",
        help="服务启动时如无索引则允许构建（需联网）",
    )
    parser.add_argument("--host", default="0.0.0.0", help="服务监听地址")
    parser.add_argument("--port", type=int, default=5000, help="服务端口")
    args = parser.parse_args()

    if args.prepare:
        prepare_assets(force_rebuild=args.rebuild_index)
        print("资源已准备完毕，退出。")
        raise SystemExit(0)

    # 服务启动前预加载，避免首次请求耗时/联网。
    # 若仅需 /qa_simple 且无法联网下载向量/嵌入，可设置环境变量 SKIP_VECTOR_PRELOAD=1 跳过预加载。
    skip_preload = os.getenv("SKIP_VECTOR_PRELOAD", "").lower() in {"1", "true", "yes"}
    try:
        if not skip_preload:
            _build_or_load_vector_store(
                allow_build=args.allow_build, force_rebuild=args.rebuild_index
            )
        _load_llm_pipeline()
    except Exception as exc:
        if not args.allow_build and not skip_preload:
            raise
        print(f"启动时资源加载失败: {exc}")

    app.run(host=args.host, port=args.port, debug=True)

