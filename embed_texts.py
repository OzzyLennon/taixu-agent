#!/usr/bin/env python3
"""
太虚大师全书 RAG 索引脚本
将1630份文本向量化后存储到ChromaDB
"""

import os
import hashlib
from pathlib import Path
import chromadb
from chromadb.config import Settings
import requests
import json
import time

# ============ 配置 ============
SILICONFLOW_API_KEY = "sk-hbxhxxjccuqnjdjhterivyluufacveaozsuurthhhqooejbi"
EMBEDDING_URL = "https://api.siliconflow.cn/v1/embeddings"
RERANKER_URL = "https://api.siliconflow.cn/v1/rerank"

RAW_TEXT_DIR = Path("D:/workspace/.claude/skills/taixu-perspective/sources/raw_text")
CHROMA_DIR = Path("D:/workspace/.claude/skills/taixu-perspective/rag/chromadb")
COLLECTION_NAME = "taixu_master"

# 分块参数
CHUNK_SIZE = 500  # 按字符数分块
CHUNK_OVERLAP = 50  # 重叠字符数

# ============ API调用 ============
def get_embedding(texts, model="Qwen/Qwen3-Embedding-8B", max_retries=5, retry_delay=3):
    """调用硅基流动API获取文本向量，带重试机制"""
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "input": texts
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(EMBEDDING_URL, headers=headers, json=payload, timeout=60)
            if response.status_code != 200:
                print(f"  [警告] API返回错误 {response.status_code}, 重试中...")
                time.sleep(retry_delay)
                continue
            result = response.json()
            return [item["embedding"] for item in result["data"]]
        except (requests.exceptions.ProxyError, requests.exceptions.ConnectionError) as e:
            print(f"  [警告] 连接错误: {e}, 重试中 ({attempt+1}/{max_retries})...")
            time.sleep(retry_delay * (attempt + 1))
            continue
        except Exception as e:
            print(f"  [警告] 未知错误: {e}, 重试中...")
            time.sleep(retry_delay)
            continue

    raise Exception("API调用失败，已达到最大重试次数")

def rerank(query, documents, model="Qwen/Qwen3-Reranker-8B", top_n=5):
    """调用硅基流动API进行Rerank"""
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": top_n
    }

    response = requests.post(RERANKER_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"Reranker API error: {response.status_code} {response.text}")

    result = response.json()
    return [(item["index"], item["relevance_score"]) for item in result["results"]]

# ============ 文本处理 ============
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """将长文本分块"""
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]

        # 找到最后一个句号或换行，避免句子中间断开
        if end < text_len:
            last_period = max(chunk.rfind('。'), chunk.rfind('\n'))
            if last_period > chunk_size // 2:
                end = start + last_period + 1
                chunk = text[start:end]

        chunks.append(chunk.strip())
        start = end - overlap if end < text_len else text_len

    return chunks

def load_all_texts():
    """加载所有文本文件"""
    texts = []
    metadatas = []
    ids = []

    for file_path in RAW_TEXT_DIR.rglob("*.txt"):
        rel_path = file_path.relative_to(RAW_TEXT_DIR)
        category = str(rel_path.parent.name)

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 跳过导航类和元信息（太短的）
        if len(content) < 100:
            continue

        # 分块
        chunks = chunk_text(content)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{rel_path.stem}_{i}"
            texts.append(chunk)
            metadatas.append({
                "source": str(rel_path),
                "category": category,
                "chunk_index": i
            })
            ids.append(chunk_id)

    return texts, metadatas, ids

# ============ ChromaDB 操作 ============
def init_chroma():
    """初始化ChromaDB"""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "太虚大师全书RAG知识库"}
    )
    return client, collection

def index_texts():
    """主索引流程，支持断点续传"""
    print("=" * 50)
    print("太虚大师全书 RAG 索引开始")
    print("=" * 50)

    # 1. 加载文本
    print("\n[1/4] 加载文本文件...")
    texts, metadatas, ids = load_all_texts()
    print(f"  共加载 {len(texts)} 个文本块")

    # 2. 初始化ChromaDB
    print("\n[2/4] 初始化ChromaDB...")
    client, collection = init_chroma()
    existing_count = collection.count()
    print(f"  Collection: {COLLECTION_NAME}")
    print(f"  已索引: {existing_count} 个文本块")

    # 3. 向量化（批量处理，避免API限制）
    print("\n[3/4] 向量化文本...")

    # 断点续传：找出已索引的ID
    existing_ids = set()
    if existing_count > 0:
        existing_results = collection.get()
        existing_ids = set(existing_results["ids"])
        print(f"  跳过已索引的 {len(existing_ids)} 个文本块")

    BATCH_SIZE = 32  # 每次处理的文本数
    total_indexed = existing_count

    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = []
        batch_ids = []
        batch_metas = []

        # 跳过已索引的
        for j in range(i, min(i + BATCH_SIZE, len(texts))):
            if ids[j] not in existing_ids:
                batch_texts.append(texts[j])
                batch_ids.append(ids[j])
                batch_metas.append(metadatas[j])

        # 如果批次为空（全跳过），继续下一批次
        if not batch_texts:
            progress = min(i + BATCH_SIZE, len(texts))
            print(f"  进度: {progress}/{len(texts)} ({100*progress/len(texts):.1f}%) - 已跳过")
            continue

        # 调用API获取embedding
        try:
            embeddings = get_embedding(batch_texts)

            # 存入ChromaDB
            collection.add(
                embeddings=embeddings,
                documents=batch_texts,
                ids=batch_ids,
                metadatas=batch_metas
            )

            total_indexed += len(batch_texts)
            progress = min(i + BATCH_SIZE, len(texts))
            print(f"  进度: {progress}/{len(texts)} ({100*progress/len(texts):.1f}%) - 本批 {len(batch_texts)} 个")

        except Exception as e:
            print(f"  [错误] 批次 {i}-{i+BATCH_SIZE} 失败: {e}")
            print(f"  将在下次运行时自动重试此批次")
            raise

        # API限流保护
        time.sleep(0.5)

    print("\n[4/4] 索引完成!")
    print(f"  Collection大小: {collection.count()}")
    print(f"  存储位置: {CHROMA_DIR}")

    return collection

if __name__ == "__main__":
    index_texts()
    print("\n索引脚本执行完毕。")
