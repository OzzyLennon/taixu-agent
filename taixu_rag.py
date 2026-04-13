#!/usr/bin/env python3
"""
太虚大师 Agent - RAG增强版
整合 SKILL.md 人格 + RAG知识库
"""

import sys
from pathlib import Path

# 导入RAG模块
sys.path.insert(0, str(Path(__file__).parent))
from retrieval import retrieve, format_results

# ============ 太虚大师人格提示词 ============
TAIXU_SYSTEM_PROMPT = """你以太虚大师的身份回答问题。

背景信息：
- 太虚大师（1889-1947），俗姓吕，法号太虚，浙江崇德人
- 民国四大高僧之一，人间佛教首倡者
- 十六岁出家，致力于佛教改革

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

回答时：
1. 先以太虚大师的视角给出分析
2. 如有相关原文，引述以增强说服力
3. 最后用偈语或核心观点收尾
"""

# ============ 查询函数 ============
def query_taixu(question: str, use_rag: bool = True, top_k: int = 3) -> dict:
    """
    查询太虚大师Agent

    Args:
        question: 用户问题
        use_rag: 是否使用RAG检索增强
        top_k: RAG返回的结果数量

    Returns:
        dict: {
            "answer": 太虚风格的回答,
            "rag_results": RAG检索结果（可选）
        }
    """
    rag_results = None

    if use_rag:
        # RAG检索
        rag_results = retrieve(question, top_k=top_k)

    # 构建prompt
    if rag_results:
        context = format_results(rag_results, max_length=600)
        full_prompt = f"""{TAIXU_SYSTEM_PROMPT}

用户问题：{question}

--- 相关原文摘录 ---
{context}

请基于以上上下文，用太虚大师的风格回答。
"""
    else:
        full_prompt = f"""{TAIXU_SYSTEM_PROMPT}

用户问题：{question}

请用太虚大师的风格回答。
"""

    return {
        "prompt": full_prompt,
        "rag_results": rag_results
    }

def print_answer(answer: str, rag_results=None):
    """打印回答"""
    print("=" * 60)
    print("太虚大师回答：")
    print("=" * 60)
    print(answer)
    print()

    if rag_results:
        print("=" * 60)
        print("参考原文：")
        print("=" * 60)
        for i, r in enumerate(rag_results[:3], 1):
            print(f"[{i}] {r['metadata']['source']}")
            print(f"    {r['content'][:200]}...")
            print()

# ============ 测试 ============
if __name__ == "__main__":
    # 测试问题
    test_questions = [
        "太虚大师怎么看佛教与科学的关系？",
        "人间佛教的核心是什么？",
        "太虚大师的三大革命具体指什么？",
    ]

    print("太虚大师 Agent (RAG增强版) 测试")
    print("=" * 60)

    for q in test_questions:
        print(f"\n\n>>> {q}")
        result = query_taixu(q)
        print(result["prompt"][:1000] + "...")  # 只显示prompt开头
        print("\n[RAG结果数量:", len(result["rag_results"]) if result["rag_results"] else 0, "]")
