#!/usr/bin/env python3
"""
太虚大师 RAG 查询CLI
用法: python cli.py "你的问题"
"""

import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from retrieval import retrieve, format_results
import requests

# SiliconFlow API 配置
SILICONFLOW_API_KEY = "sk-hbxhxxjccuqnjdjhterivyluufacveaozsuurthhhqooejbi"
LLM_URL = "https://api.siliconflow.cn/v1/chat/completions"

# 太虚大师人格提示词
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

回答要求：
1. 先以太虚大师的视角给出分析
2. 如有相关原文，引述以增强说服力
3. 最后用偈语或核心观点收尾
"""

def call_llm(prompt, model="Pro/deepseek-ai/DeepSeek-V3.2"):
    """调用LLM生成回答"""
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": TAIXU_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    try:
        response = requests.post(LLM_URL, headers=headers, json=payload, timeout=120)
        if response.status_code != 200:
            return f"LLM API错误: {response.status_code} {response.text}"
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"调用失败: {e}"

def query_taixu(question, top_k=3, use_llm=True):
    """
    查询太虚大师Agent

    Args:
        question: 用户问题
        top_k: RAG返回的结果数量
        use_llm: 是否调用LLM生成回答

    Returns:
        dict: {"answer": 回答, "sources": 原文来源}
    """
    print("=" * 60)
    print(f"问题: {question}")
    print("=" * 60)

    # 1. RAG检索
    print("\n[1] RAG检索中...")
    results = retrieve(question, top_k=top_k)

    if not results:
        print("[警告] 未找到相关结果")
        return {"answer": "未找到相关结果", "sources": []}

    print(f"    找到 {len(results)} 条相关原文")

    # 2. 构建prompt
    context = format_results(results, max_length=800)

    if use_llm:
        print("\n[2] 调用LLM生成回答...")
        full_prompt = f"""基于以下相关原文，用太虚大师的风格回答问题。

用户问题：{question}

--- 相关原文摘录 ---
{context}

请基于以上上下文，用太虚大师的风格回答。"""
        answer = call_llm(full_prompt)
    else:
        # 只返回检索结果
        answer = context

    return {
        "answer": answer,
        "sources": [
            {
                "source": r["metadata"]["source"],
                "category": r["metadata"]["category"],
                "score": r["rerank_score"],
                "content": r["content"][:300] + "..."
            }
            for r in results
        ]
    }

def main():
    if len(sys.argv) < 2:
        print("用法: python cli.py \"你的问题\"")
        print("示例: python cli.py \"太虚大师怎么看佛教与科学的关系\"")
        sys.exit(1)

    question = sys.argv[1]
    result = query_taixu(question)

    print("\n" + "=" * 60)
    print("太虚大师回答：")
    print("=" * 60)
    print(result["answer"])

    print("\n" + "=" * 60)
    print("参考原文：")
    print("=" * 60)
    for i, src in enumerate(result["sources"], 1):
        print(f"\n[{i}] {src['source']} ({src['category']})")
        print(f"    相关度: {src['score']:.4f}")
        print(f"    {src['content']}")

if __name__ == "__main__":
    main()
