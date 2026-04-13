#!/usr/bin/env python3
"""
太虚大师 Agent - Flask Web应用
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, render_template, request, jsonify
from taixu_agent_v3 import TaixuAgentDeep

app = Flask(__name__)

# 全局Agent实例
agent = TaixuAgentDeep()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "问题不能为空"}), 400

    try:
        result = agent.ask(question)
        return jsonify({
            "answer": result["answer"],
            "question_analysis": {
                "type": result["question_analysis"]["type"],
                "primary_model": result["question_analysis"]["primary_model"],
                "rag_intensity": result["question_analysis"]["rag_intensity"],
                "reason": result["question_analysis"]["reason"],
                "needs_web_search": result["question_analysis"].get("needs_web_search", False),
                "emotion": result["question_analysis"].get("emotion", {})
            },
            "sources": [
                {
                    "source": s["metadata"]["source"],
                    "score": s["rerank_score"],
                    "content": s["content"]
                }
                for s in result["sources"][:5]
            ],
            "web_results": result.get("web_results", []),
            "conflicts": result["conflicts"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/reset", methods=["POST"])
def reset():
    agent.reset()
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    print("=" * 60)
    print("太虚大师 Agent Web服务启动中...")
    print("请访问: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=True)
