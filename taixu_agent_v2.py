#!/usr/bin/env python3
"""
太虚大师 Agent - 溯源分层增强版
区分三层来源：直接引用 / 分析推理 / 超越推演
"""

import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

# ============ 配置 ============
SILICONFLOW_API_KEY = "sk-hbxhxxjccuqnjdjhterivyluufacveaozsuurthhhqooejbi"
LLM_URL = "https://api.siliconflow.cn/v1/chat/completions"
EMBEDDING_URL = "https://api.siliconflow.cn/v1/embeddings"
RERANKER_URL = "https://api.siliconflow.cn/v1/rerank"

CHROMA_DIR = Path(__file__).parent / "chromadb"
COLLECTION_NAME = "taixu_master"
SKILL_PATH = Path(__file__).parent.parent / "SKILL.md"

# 检索参数
INITIAL_TOP_K = 30
RERANK_TOP_K = 8

# ============ API调用 ============
def get_embedding(texts, model="Qwen/Qwen3-Embedding-8B"):
    import requests
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"model": model, "input": texts}
    response = requests.post(EMBEDDING_URL, headers=headers, json=payload, timeout=180)
    if response.status_code != 200:
        raise Exception(f"Embedding API error: {response.status_code}")
    result = response.json()
    return [item["embedding"] for item in result["data"]]

def rerank(query, documents, model="Qwen/Qwen3-Reranker-8B", top_n=8):
    import requests
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"model": model, "query": query, "documents": documents, "top_n": top_n}
    for attempt in range(3):
        try:
            response = requests.post(RERANKER_URL, headers=headers, json=payload, timeout=180)
            if response.status_code != 200:
                continue
            result = response.json()
            return [(item["index"], item["relevance_score"]) for item in result["results"]]
        except:
            continue
    return [(i, 1.0 - i * 0.01) for i in range(min(top_n, len(documents)))]

def call_llm(messages, model="Pro/deepseek-ai/DeepSeek-V3.2"):
    import requests
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7
    }
    response = requests.post(LLM_URL, headers=headers, json=payload, timeout=300)
    if response.status_code != 200:
        raise Exception(f"LLM API error: {response.status_code} {response.text}")
    result = response.json()
    return result["choices"][0]["message"]["content"]

# ============ ChromaDB检索 ============
def retrieve_from_chroma(query, initial_k=INITIAL_TOP_K, top_k=RERANK_TOP_K):
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)

    query_embedding = get_embedding([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=initial_k
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]

    if not documents:
        return []

    reranked = rerank(query, documents, top_n=min(top_k, len(documents)))

    final_results = []
    for idx, score in reranked:
        doc_idx = int(idx)
        final_results.append({
            "content": documents[doc_idx],
            "metadata": metadatas[doc_idx],
            "rerank_score": score,
            "id": ids[doc_idx]
        })
    return final_results

# ============ 太虚Agent溯源分层版 ============
class TaixuAgentTracer:
    """
    太虚大师Agent（溯源分层版）

    三层来源：
    1. 【原文引用】直接来自太虚原文的表述
    2. 【分析推理】基于原文的分析和阐述
    3. 【超越推演】超出原文，基于太虚思维框架的推演
    """

    def __init__(self):
        self.conversation_history = []
        self.disclaimer_given = False

    def ask(self, question: str, top_k: int = 8) -> dict:
        """
        返回带溯源分层的回答
        """
        # 首次免责声明
        if not self.disclaimer_given:
            disclaimer = "我以太虚大师视角和你交流，基于公开言论推断，非本人观点。\n\n"
            self.disclaimer_given = True
        else:
            disclaimer = ""

        # RAG检索
        rag_results = retrieve_from_chroma(question, top_k=top_k)

        # 构建溯源分层Prompt
        context = self._format_context(rag_results)

        system_prompt = self._build_system_prompt()
        user_message = self._build_user_message(question, context)

        # 调用LLM
        messages = [
            {"role": "system", "content": system_prompt},
            *self.conversation_history,
            {"role": "user", "content": user_message}
        ]

        answer = call_llm(messages)

        # 更新历史
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": answer})

        return {
            "answer": disclaimer + answer,
            "sources": rag_results,
            "context": context
        }

    def _format_context(self, results) -> str:
        """格式化RAG结果，标注相关度"""
        context_parts = []
        for i, r in enumerate(results, 1):
            score = r["rerank_score"]
            # 标注相关度等级
            if score > 0.9:
                tag = "【高度相关】"
            elif score > 0.7:
                tag = "【中度相关】"
            else:
                tag = "【参考】"

            context_parts.append(
                f"{tag} {i}.\n"
                f"出处：{r['metadata']['source']}\n"
                f"内容：{r['content'][:600]}..."
            )
        return "\n\n".join(context_parts)

    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        return """你以太虚大师的身份回答问题。

身份：
我是太虚，俗姓吕，名淦森，法号太虚，浙江崇德人。十六岁出家，致力于佛教改革，是「人间佛教」首倡者，民国四大高僧之一。我已于1947年圆寂于上海玉佛寺。

核心思想：
- 即人成佛：做人即成佛，现实中求觉悟
- 契理契机：不脱离义理，又适应时代机宜
- 三大革命：教理革命、教制革命、教产革命
- 真现实论：现实是现前事实

表达风格：
- 文白相间，长句为主
- 善用偈语：「仰止唯佛陀，完成在人格；人圆佛即成，是名真现实」
- 确定性强，常用「必」「须」「当」

【重要】回答结构要求：
你的回答必须包含以下三层来源的明确标注：

1. 【原文引用】- 直接引用太虚原话的部分，用「"..."」标出，末尾注明出处
2. 【分析推理】- 基于原文的分析和阐述，说明"这意味着..."
3. 【超越推演】- 超出原文，基于太虚思维框架对新问题的推演，说明"以我的观点..."

最后用一句偈语收尾。

如果检索到的原文不直接相关，请基于太虚思想框架进行分析，明确标注各层来源。"""

    def _build_user_message(self, question: str, context: str) -> str:
        return f"""基于以下检索到的原文，用太虚大师的风格回答。

用户问题：{question}

--- 相关原文 ---
{context}

请严格按以下格式回答：

## 【原文引用】
（直接引用太虚原话，标明出处）

## 【分析推理】
（基于原文的分析）

## 【超越推演】
（超出原文的推演，基于太虚思维框架）

最后用偈语收尾。

请用太虚的口吻回答。"""

    def reset(self):
        self.conversation_history = []
        self.disclaimer_given = False


# ============ CLI ============
def main():
    if len(sys.argv) < 2:
        print("用法: python taixu_agent_v2.py \"你的问题\"")
        sys.exit(1)

    question = sys.argv[1]

    sys.stdout.reconfigure(encoding='utf-8')

    print("=" * 70)
    print("太虚大师 Agent (溯源分层增强版)")
    print("=" * 70)
    print(f"\n问题: {question}\n")

    agent = TaixuAgentTracer()
    result = agent.ask(question)

    print("【回答】")
    print("-" * 70)
    print(result["answer"])

    print("\n\n【参考原文】")
    print("-" * 70)
    for i, src in enumerate(result["sources"][:5], 1):
        print(f"{i}. {src['metadata']['source']} (score: {src['rerank_score']:.4f})")
        print(f"   {src['content'][:150]}...")

if __name__ == "__main__":
    main()
