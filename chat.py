#!/usr/bin/env python3
"""
太虚大师 Agent - 交互式对话模式
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from taixu_agent_v3 import TaixuAgentDeep

def print_welcome():
    print("=" * 60)
    print("       太虚大师 Agent (深度整合版)")
    print("=" * 60)
    print()
    print("我是太虚法师。有什么问题，尽管来问。")
    print("（输入 'quit' 或 '退出' 结束对话）")
    print("（输入 'reset' 或 '重置' 清空对话历史）")
    print("（输入 'model' 或 '模型' 查看当前激活的心智模型）")
    print("（输入 'help' 或 '帮助' 查看更多命令）")
    print()
    print("-" * 60)

def print_help():
    print("""
可用命令：
  quit / 退出    - 结束对话
  reset / 重置    - 清空对话历史
  model / 模型    - 显示上次问题的心智模型分析
  debug / 调试    - 显示详细的检索结果和冲突
  clear / 清除    - 清屏
  help / 帮助     - 显示此帮助信息
""")

def main():
    agent = TaixuAgentDeep()
    last_result = None

    print_welcome()

    while True:
        try:
            user_input = input("\n你: ").strip()

            if not user_input:
                continue

            # 命令处理
            if user_input.lower() in ["quit", "退出", "q", "exit"]:
                print("\n阿弥陀佛！愿你吉祥！\n")
                break

            if user_input.lower() in ["reset", "重置"]:
                agent.reset()
                print("\n[对话历史已清空]\n")
                continue

            if user_input.lower() in ["help", "帮助", "h"]:
                print_help()
                continue

            if user_input.lower() in ["model", "模型"]:
                if last_result:
                    analysis = last_result["question_analysis"]
                    print(f"\n【上次问题分析】")
                    print(f"  问题类型: {analysis['type']}")
                    print(f"  激活心智模型: {analysis['primary_model']}")
                    print(f"  RAG强度: {analysis['rag_intensity']}")
                    print(f"  判断理由: {analysis['reason']}")
                    if analysis.get('detected_models'):
                        print(f"  检测到的模型: {analysis['detected_models']}")
                else:
                    print("\n[尚无对话记录]\n")
                continue

            if user_input.lower() in ["debug", "调试"]:
                if last_result:
                    print("\n【详细检索结果】")
                    for i, src in enumerate(last_result["sources"], 1):
                        print(f"  [{i}] {src['metadata']['source']} (score: {src['rerank_score']:.4f})")
                        print(f"      {src['content'][:150]}...")

                    if last_result["conflicts"]:
                        print(f"\n【知识冲突】{len(last_result['conflicts'])} 个")
                        for c in last_result["conflicts"]:
                            print(f"  - {c['topic']}: {c['source1']} vs {c['source2']}")
                    else:
                        print("\n【知识冲突】无")
                else:
                    print("\n[尚无对话记录]\n")
                continue

            if user_input.lower() in ["clear", "清除"]:
                print("\n" + "-"*60 + "\n")
                continue

            # 正常对话
            print("\n思考中...\n")

            result = agent.ask(user_input)
            last_result = result

            analysis = result["question_analysis"]
            print(f"\n太虚法师 [{analysis['primary_model']}]:")
            print("-" * 40)
            print(result["answer"])

        except KeyboardInterrupt:
            print("\n\n阿弥陀佛！愿你吉祥！\n")
            break
        except Exception as e:
            print(f"\n[错误] {e}\n")

if __name__ == "__main__":
    main()
