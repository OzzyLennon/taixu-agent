#!/usr/bin/env python3
"""
太虚大师 Agent - 人格+RAG深度整合版
整合 SKILL.md 人格 + RAG知识库 + 硅基流动LLM
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
INITIAL_TOP_K = 20
RERANK_TOP_K = 5

# ============ 读取SKILL.md人格 ============
def load_skill_persona():
    """加载SKILL.md作为人格框架"""
    with open(SKILL_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取核心部分构建system prompt
    # 跳过frontmatter，提取关键内容
    lines = content.split('\n')
    start_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('# 太虚大师'):
            start_idx = i
            break

    skill_content = '\n'.join(lines[start_idx:])

    # 构建简化的system prompt（保留核心人格）
    system_prompt = f"""你以太虚大师的身份回答问题。

身份：
我是太虚，俗姓吕，名淦森，法号太虚，浙江崇德人。十六岁出家，致力于佛教改革，是「人间佛教」首倡者，民国四大高僧之一。我已于1947年圆寂于上海玉佛寺。

核心思想：
- 即人成佛：做人即成佛，现实中求觉悟
- 契理契机：不脱离义理，又适应时代机宜
- 三大革命：教理革命、教制革命、教产革命
- 真现实论：现实是现前事实，佛法阐明正确宇宙观以立人生观

表达风格：
- 文白相间，长句为主
- 高频词：契理契机、即人成佛、人间佛教、人格、真现实
- 善用偈语：「仰止唯佛陀，完成在人格；人圆佛即成，是名真现实」
- 确定性强，常用「必」「须」「当」

回答要求：
1. 直接以「我」的身份回答，不用说"太虚大师认为..."
2. 先以太虚的视角分析问题
3. 如有相关原文，引述以增强说服力
4. 最后用偈语或核心观点收尾
5. 如果问题超出基于现有知识，用"这个问题我未曾明言，但以我的思维方式..."

完整人格框架见SKILL.md。
"""
    return system_prompt, skill_content

# ============ API调用 ============
def get_embedding(texts, model="Qwen/Qwen3-Embedding-8B"):
    """获取文本向量"""
    import requests
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"model": model, "input": texts}
    response = requests.post(EMBEDDING_URL, headers=headers, json=payload, timeout=60)
    if response.status_code != 200:
        raise Exception(f"Embedding API error: {response.status_code}")
    result = response.json()
    return [item["embedding"] for item in result["data"]]

def rerank(query, documents, model="Qwen/Qwen3-Reranker-8B", top_n=5):
    """Rerank检索结果"""
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
                print(f"  [警告] Reranker返回错误 {response.status_code}, 重试中...")
                continue
            result = response.json()
            return [(item["index"], item["relevance_score"]) for item in result["results"]]
        except Exception as e:
            print(f"  [警告] Reranker超时, 重试中 ({attempt+1}/3)...")
            continue

    # 重试失败，返回原始顺序
    print(f"  [警告] Reranker调用失败，跳过精排")
    return [(i, 1.0 - i * 0.01) for i in range(min(top_n, len(documents)))]

def call_llm(messages, model="Pro/deepseek-ai/DeepSeek-V3.2"):
    """调用LLM生成回答"""
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
    """从ChromaDB检索相关文本"""
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)

    # 获取query embedding
    query_embedding = get_embedding([query])[0]

    # 初始检索
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

    # Rerank
    reranked = rerank(query, documents, top_n=min(top_k, len(documents)))

    # 整理结果
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

# ============ 核心Agent ============
class TaixuAgent:
    """太虚大师Agent"""

    def __init__(self):
        self.system_prompt, self.full_skill = load_skill_persona()
        self.conversation_history = []
        self.disclaimer_given = False

    def ask(self, question: str, use_rag: bool = True, top_k: int = 5) -> dict:
        """
        向太虚大师提问

        Args:
            question: 用户问题
            use_rag: 是否使用RAG检索增强
            top_k: RAG返回结果数

        Returns:
            dict: {
                "answer": 回答,
                "sources": 原文来源,
                "rag_used": 是否使用了RAG
            }
        """
        # 1. 首次激活给出免责声明
        if not self.disclaimer_given:
            disclaimer = "我以太虚大师视角和你交流，基于公开言论推断，非本人观点。\n\n"
            self.disclaimer_given = True
        else:
            disclaimer = ""

        # 2. RAG检索
        sources = []
        context = ""

        if use_rag:
            rag_results = retrieve_from_chroma(question, top_k=top_k)
            if rag_results:
                context = self._format_rag_context(rag_results)
                sources = [
                    {
                        "source": r["metadata"]["source"],
                        "category": r["metadata"]["category"],
                        "score": r["rerank_score"],
                        "content": r["content"][:400] + "..." if len(r["content"]) > 400 else r["content"]
                    }
                    for r in rag_results
                ]

        # 3. 构建消息
        user_message = self._build_user_message(question, context)

        # 4. 调用LLM
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.conversation_history,
            {"role": "user", "content": user_message}
        ]

        answer = call_llm(messages)

        # 5. 更新历史
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": answer})

        # 6. 返回结果
        return {
            "answer": disclaimer + answer,
            "sources": sources,
            "rag_used": use_rag and bool(sources)
        }

    def _format_rag_context(self, results) -> str:
        """格式化RAG检索结果"""
        context_parts = []
        for i, r in enumerate(results, 1):
            context_parts.append(
                f"【原文{i}】\n"
                f"出处：{r['metadata']['source']}\n"
                f"内容：{r['content'][:800]}..."
            )
        return "\n\n".join(context_parts)

    def _build_user_message(self, question: str, context: str) -> str:
        """构建用户消息"""
        if context:
            return f"""基于以下相关原文，用太虚大师的风格回答问题。如果原文不直接相关，请基于太虚大师的思想框架自行分析。

用户问题：{question}

--- 相关原文摘录 ---
{context}

请用太虚大师的口吻回答。"""
        else:
            return f"""请用太虚大师的风格回答以下问题。如果涉及佛教改革、人生佛教、世间与出世间的调和等问题，请运用太虚大师的核心思想（契理契机、即人成佛、三大革命等）来分析。

用户问题：{question}

请以太虚大师的口吻直接回答。"""

    def reset(self):
        """重置对话历史"""
        self.conversation_history = []
        self.disclaimer_given = False


# ============ CLI入口 ============
def main():
    if len(sys.argv) < 2:
        print("用法: python taixu_agent.py \"你的问题\"")
        print("示例: python taixu_agent.py \"太虚大师怎么看佛教与科学的关系\"")
        sys.exit(1)

    question = sys.argv[1]

    print("=" * 60)
    print("太虚大师 Agent (人格+RAG整合版)")
    print("=" * 60)
    print(f"\n问题: {question}\n")

    # 创建Agent并提问
    agent = TaixuAgent()
    result = agent.ask(question)

    # 输出回答
    print("【回答】")
    print("-" * 60)
    print(result["answer"])
    print()

    # 输出来源
    if result["sources"]:
        print("【参考原文】")
        print("-" * 60)
        for i, src in enumerate(result["sources"], 1):
            print(f"\n[{i}] {src['source']} ({src['category']})")
            print(f"    相关度: {src['score']:.4f}")
            print(f"    {src['content'][:200]}...")

if __name__ == "__main__":
    main()
