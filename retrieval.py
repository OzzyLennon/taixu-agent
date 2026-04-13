#!/usr/bin/env python3
"""
太虚大师全书 RAG 检索脚本
使用Embedding + Reranker 混合检索
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

import chromadb
from chromadb.config import Settings
import requests
import json

# ============ 配置 ============
SILICONFLOW_API_KEY = "sk-hbxhxxjccuqnjdjhterivyluufacveaozsuurthhhqooejbi"
EMBEDDING_URL = "https://api.siliconflow.cn/v1/embeddings"
RERANKER_URL = "https://api.siliconflow.cn/v1/rerank"

CHROMA_DIR = Path("D:/workspace/.claude/skills/taixu-perspective/rag/chromadb")
COLLECTION_NAME = "taixu_master"

# 检索参数
INITIAL_TOP_K = 50  # 初始检索数量
RERANK_TOP_K = 10   # Rerank后返回数量

# ============ API调用 ============
def get_embedding(texts, model="Qwen/Qwen3-Embedding-8B"):
    """调用硅基流动API获取文本向量"""
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "input": texts
    }

    response = requests.post(EMBEDDING_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"Embedding API error: {response.status_code} {response.text}")

    result = response.json()
    return [item["embedding"] for item in result["data"]]

def rerank(query, documents, model="Qwen/Qwen3-Reranker-8B", top_n=RERANK_TOP_K):
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

# ============ ChromaDB 操作 ============
def get_chroma_collection():
    """获取ChromaDB collection"""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)
    return collection

# ============ 检索 ============
def retrieve(query, initial_k=INITIAL_TOP_K, top_k=RERANK_TOP_K):
    """
    检索函数：Embedding + Rerank 混合检索

    Args:
        query: 查询文本
        initial_k: 初始向量检索返回数量
        top_k: 最终返回数量

    Returns:
        list: 检索结果，每项包含 document, metadata, score
    """
    # 1. 获取query的embedding
    query_embedding = get_embedding([query])[0]

    # 2. 初始向量检索
    collection = get_chroma_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=initial_k
    )

    # 3. 提取检索到的文档
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]

    if not documents:
        return []

    # 4. Rerank优化
    reranked = rerank(query, documents, top_n=min(top_k, len(documents)))

    # 5. 整理结果
    final_results = []
    for idx, score in reranked:
        doc_idx = int(idx)
        final_results.append({
            "content": documents[doc_idx],
            "metadata": metadatas[doc_idx],
            "rerank_score": score,
            "vector_distance": distances[doc_idx],
            "id": ids[doc_idx]
        })

    return final_results

def format_results(results, max_length=500):
    """格式化检索结果为可读文本"""
    if not results:
        return "未找到相关结果。"

    output = []
    for i, r in enumerate(results, 1):
        content = r["content"]
        if len(content) > max_length:
            content = content[:max_length] + "..."

        output.append(f"--- 结果 {i} ---")
        output.append(f"来源: {r['metadata']['source']}")
        output.append(f"分类: {r['metadata']['category']}")
        output.append(f"相关度: {r['rerank_score']:.4f}")
        output.append(f"内容:\n{content}")
        output.append("")

    return "\n".join(output)

# ============ 主函数 ============
if __name__ == "__main__":
    # 测试检索
    test_queries = [
        "太虚大师对人间佛教的理解",
        "佛教与科学的关系",
        "太虚大师的三大革命是什么",
    ]

    print("=" * 60)
    print("太虚大师全书 RAG 检索测试")
    print("=" * 60)

    for query in test_queries:
        print(f"\n\n>>> 查询: {query}")
        print("-" * 40)
        results = retrieve(query)
        print(format_results(results))
