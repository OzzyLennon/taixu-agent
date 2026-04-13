#!/usr/bin/env python3
"""
太虚大师Agent测试脚本
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from taixu_agent import TaixuAgent

def test_agent():
    agent = TaixuAgent()

    test_questions = [
        "佛教应该如何面对AI时代的挑战？",
        "太虚大师的三大革命具体指什么？",
        "人间佛教与净土宗有什么区别？",
    ]

    print("=" * 70)
    print("太虚大师 Agent 测试")
    print("=" * 70)

    for q in test_questions:
        print(f"\n\n>>> {q}")
        print("-" * 70)
        result = agent.ask(q)
        print("【回答】")
        print(result["answer"])
        if result["sources"]:
            print("\n【参考原文】")
            for i, src in enumerate(result["sources"], 1):
                print(f"  [{i}] {src['source']} (score: {src['score']:.4f})")
                print(f"      {src['content'][:150]}...")

    print("\n\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)

if __name__ == "__main__":
    test_agent()
