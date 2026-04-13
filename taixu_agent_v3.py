#!/usr/bin/env python3
"""
太虚大师 Agent - 深度整合版
实现：动态RAG触发 + 思维链引导 + 知识冲突处理 + 联网检索
"""

import sys
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# 导入配置
from config import (
    SILICONFLOW_API_KEY, LLM_URL, EMBEDDING_URL, RERANKER_URL,
    CHROMA_DIR, COLLECTION_NAME, INITIAL_TOP_K, RERANK_TOP_K,
    WEB_SEARCH_ENABLED, WEB_SEARCH_TOP_K
)

# ============ 太虚心智模型定义 ============
MENTAL_MODELS = {
    "即人成佛": {
        "keywords": ["成佛", "人格", "人间佛教", "做人", "现实人生", "修养", "净土", "超脱"],
        "description": "核心：做人即成佛，现实中求觉悟",
        "related_works": ["真现实论", "人生佛教", "即人成佛"]
    },
    "契理契机": {
        "keywords": ["时代", "变革", "适应", "契机", "传统", "现代", "应机", "弘法方式"],
        "description": "核心：不脱离义理，又适应时代机宜",
        "related_works": ["整理僧伽制度论", "新与融贯"]
    },
    "三大革命": {
        "keywords": ["革命", "改革", "教理", "教制", "教产", "革新", "变法"],
        "description": "核心：教理革命、教制革命、教产革命",
        "related_works": ["教理革命", "教制革命", "教产革命"]
    },
    "真现实论": {
        "keywords": ["现实", "真理", "宇宙", "人生观", "认识论", "存在"],
        "description": "核心：现实是现前事实，非抽象概念",
        "related_works": ["真现实论", "宗依论", "宗体论"]
    },
    "世界佛教": {
        "keywords": ["国际", "世界", "欧美", "交流", "传播", "巴黎", "联合"],
        "description": "核心：中国佛教走向世界",
        "related_works": ["世界佛教联合会", "海潮音"]
    },
    "僧伽教育": {
        "keywords": ["教育", "佛学院", "人才", "培养", "武昌", "学修", "僧才"],
        "description": "核心：培养人才是佛教振兴根本",
        "related_works": ["武昌佛学院", "闽南佛学院", "汉藏教理院"]
    }
}

# ============ API调用 ============
def get_embedding(texts, model="Qwen/Qwen3-Embedding-8B"):
    import requests
    headers = {"Authorization": f"Bearer {SILICONFLOW_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "input": texts}
    response = requests.post(EMBEDDING_URL, headers=headers, json=payload, timeout=180)
    if response.status_code != 200:
        return []
    result = response.json()
    return [item["embedding"] for item in result["data"]]

def rerank(query, documents, model="Qwen/Qwen3-Reranker-8B", top_n=10):
    import requests
    headers = {"Authorization": f"Bearer {SILICONFLOW_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "query": query, "documents": documents, "top_n": top_n}
    for attempt in range(3):
        try:
            response = requests.post(RERANKER_URL, headers=headers, json=payload, timeout=180)
            if response.status_code == 200:
                result = response.json()
                return [(item["index"], item["relevance_score"]) for item in result["results"]]
        except:
            continue
    return [(i, 1.0) for i in range(len(documents))]

def call_llm(messages, model="Pro/deepseek-ai/DeepSeek-V3.2"):
    import requests
    headers = {"Authorization": f"Bearer {SILICONFLOW_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": 0.7}
    response = requests.post(LLM_URL, headers=headers, json=payload, timeout=300)
    if response.status_code != 200:
        raise Exception(f"LLM API error: {response.status_code}")
    result = response.json()
    return result["choices"][0]["message"]["content"]

# ============ 联网检索 ============
def web_search(query, top_k=WEB_SEARCH_TOP_K):
    """
    使用网络搜索补充最新信息
    返回格式化的搜索结果列表
    """
    if not WEB_SEARCH_ENABLED:
        return []

    try:
        from mcp__MiniMax__web_search import web_search as mini_max_search
        result = mini_max_search(query)

        if not result or "organic" not in result:
            return []

        results = []
        for item in result["organic"][:top_k]:
            results.append({
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "link": item.get("link", ""),
                "date": item.get("date", "")
            })
        return results
    except Exception as e:
        print(f"[联网检索失败] {e}", file=sys.stderr)
        return []

def format_web_results(search_results):
    """格式化联网检索结果"""
    if not search_results:
        return ""

    parts = ["【联网搜索补充】"]
    for i, r in enumerate(search_results, 1):
        date_info = f"（{r['date']}）" if r['date'] else ""
        parts.append(
            f"{i}. {r['title']}{date_info}\n"
            f"   {r['snippet']}"
        )
    return "\n".join(parts)

# ============ RAG检索 ============
def retrieve_from_chroma(query, initial_k=INITIAL_TOP_K, top_k=RERANK_TOP_K):
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)

    query_embedding = get_embedding([query])
    if not query_embedding:
        return []
    query_embedding = query_embedding[0]

    results = collection.query(query_embeddings=[query_embedding], n_results=initial_k)
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

# ============ 情感检测 ============
def detect_emotional_context(question: str) -> dict:
    """
    检测问题中的情感色彩和敏感度
    返回情感上下文信息
    """
    q = question.lower()

    # 检测困惑/焦虑类情感
    anxious_keywords = ["焦虑", "困惑", "迷茫", "痛苦", "难受", "难过", "伤心",
                       "怎么办", "为什么", "不懂", "不理解", "失望", "绝望"]
    is_anxious = any(kw in q for kw in anxious_keywords)

    # 检测求助/期待类情感
    hopeful_keywords = ["希望", "想", "愿", "求", "请问", "帮助下", "指点",
                       "改进", "优化", "完善", "发展"]
    is_hopeful = any(kw in q for kw in hopeful_keywords)

    # 检测批评/质疑类
    critical_keywords = ["批评", "质疑", "反对", "不对", "错误", "问题", "毛病"]
    is_critical = any(kw in q for kw in critical_keywords)

    # 检测自我相关（涉及自身处境）
    self_related = ["我", "我的", "我们", "自己"]
    is_personal = any(kw in q for kw in self_related)

    return {
        "is_anxious": is_anxious,
        "is_hopeful": is_hopeful,
        "is_critical": is_critical,
        "is_personal": is_personal,
        "needs_compassion": is_anxious or (is_personal and not is_hopeful),
        "response_tone": "warm" if is_anxious else ("direct" if is_critical else "balanced")
    }

# ============ 动态RAG触发判断 ============
def analyze_question_type(question: str) -> dict:
    """
    分析问题类型，决定RAG触发策略
    返回: {
        "type": "factual" | "analytical" | "speculative",
        "primary_model": str,  # 主要心智模型
        "rag_intensity": float,  # RAG强度 0-1
        "reasoning": str  # 判断理由
    }
    """
    q = question.lower()

    # 检测是否涉及太虚未经历的时代话题
    modern_topics = ["ai", "人工智能", "互联网", "电脑", "手机", "网络", "元宇宙",
                     "基因", "克隆", "量子", "太空", "航天", "现代科技", "当代"]
    is_modern = any(topic in q for topic in modern_topics)

    # 检测是否涉及具体事实（人物、事件、时间）
    factual_keywords = ["谁", "什么", "何时", "哪一年", "哪个", "什么人",
                       "什么事", "具体", "历史", "记载", "说过", "提出"]
    is_factual = any(kw in q for kw in factual_keywords)

    # 检测是否涉及太虚核心概念
    model_keywords = {name: model["keywords"] for name, model in MENTAL_MODELS.items()}
    detected_models = []
    for model_name, keywords in model_keywords.items():
        if any(kw in q for kw in keywords):
            detected_models.append(model_name)

    # 判断问题类型
    if is_factual and not is_modern:
        q_type = "factual"
        rag_intensity = 0.9
        reason = "事实性问题，强RAG检索原文"
    elif is_modern:
        q_type = "speculative"
        rag_intensity = 0.4  # 现代话题原文少，降低RAG依赖
        reason = "当代话题，依赖思维框架推演"
    else:
        q_type = "analytical"
        rag_intensity = 0.7
        reason = "分析性问题，RAG+框架并重"

    # 检测是否为纯SKILL类型（不需要RAG）
    # 1. 打招呼
    q_head = question[:10]
    greeting_keywords = [
        "您好", "你好", "大师", "阿弥陀佛", "合十", "善哉",
        "打扰", "冒昧", "失礼", "谢谢", "感谢",
        "再见", "再会", "告辞", "晚安", "早安", "午安"
    ]
    is_greeting = any(kw in q_head for kw in greeting_keywords)

    # 2. 身份/自我相关问题
    identity_patterns = [
        "你是谁", "你是太虚", "你叫什么", "你是法师", "你是僧人",
        "你几岁", "你哪一年", "你什么时候", "你在哪里",
        "你觉得", "你的看法", "你认为", "你的观点",
        "你能", "你可以", "你会什么", "你能回答"
    ]
    is_identity = any(p in question for p in identity_patterns)

    # 3. 简单感谢/道歉
    polite_keywords = ["谢谢", "感谢", "辛苦了", "打扰了", "抱歉", "对不起"]
    has_polite = any(kw in question for kw in polite_keywords)

    # 判断纯SKILL类型（不需要RAG）
    inquiry_keywords = ["怎么", "如何", "为什么", "什么", "是否", "能不能",
                       "佛教", "佛学", "修行", "佛法", "释迦", "菩萨"]
    has_inquiry = any(kw in question for kw in inquiry_keywords)

    if (is_greeting or is_identity or has_polite) and not has_inquiry and len(question) < 30:
        q_type = "pure_skill"
        rag_intensity = 0.0
        reason = "纯SKILL问答，无需RAG检索"

    # 选择主要心智模型
    if detected_models:
        primary_model = detected_models[0]
    else:
        # 默认使用「契理契机」作为分析框架
        primary_model = "契理契机"

    # 情感检测
    emotion = detect_emotional_context(question)

    return {
        "type": q_type,
        "primary_model": primary_model,
        "rag_intensity": rag_intensity,
        "reason": reason,
        "detected_models": detected_models,
        "needs_web_search": is_modern or rag_intensity < 0.5,
        "emotion": emotion
    }

# ============ 知识冲突检测 ============
def detect_knowledge_conflicts(results: list, primary_model: str) -> list:
    """
    检测检索结果中的知识冲突
    返回冲突列表
    """
    conflicts = []

    # 检查原文间是否有矛盾观点
    for i, r1 in enumerate(results[:5]):
        for j, r2 in enumerate(results[i+1:5], i+1):
            # 简单检测：如果两篇文章讨论相似话题但观点不同
            content1 = r1["content"][:200].lower()
            content2 = r2["content"][:200].lower()

            # 检测是否涉及同一主题
            common_topics = []
            for topic in ["革命", "改革", "保守", "传统", "现代", "科学", "佛教"]:
                if topic in content1 and topic in content2:
                    common_topics.append(topic)

            if common_topics:
                # 检查相关度差异是否过大（可能存在分歧）
                score_diff = abs(r1["rerank_score"] - r2["rerank_score"])
                if score_diff > 0.3:
                    conflicts.append({
                        "type": "divergent_views",
                        "topic": common_topics[0],
                        "source1": r1["metadata"]["source"],
                        "source2": r2["metadata"]["source"],
                        "severity": "moderate"
                    })

    return conflicts

# ============ 太虚Agent核心 ============
class TaixuAgentDeep:
    """
    太虚大师Agent - 深度整合版

    功能：
    1. 动态RAG触发：根据问题类型决定RAG强度
    2. 思维链引导：先识别心智模型，再构建回答
    3. 知识冲突处理：检测并标注原文间的观点分歧
    """

    def __init__(self):
        self.conversation_history = []
        self.disclaimer_given = False

    def ask(self, question: str, top_k: int = 10) -> dict:
        sys.stdout.reconfigure(encoding='utf-8')

        # 1. 分析问题类型，决定RAG策略
        question_analysis = analyze_question_type(question)

        # 2. RAG检索
        all_results = retrieve_from_chroma(question, top_k=top_k)

        # 3. 知识冲突检测
        conflicts = detect_knowledge_conflicts(all_results, question_analysis["primary_model"])

        # 4. 根据RAG强度筛选结果
        effective_k = int(top_k * question_analysis["rag_intensity"])
        if question_analysis["rag_intensity"] <= 0:
            effective_results = []  # 不检索
        else:
            effective_results = all_results[:max(effective_k, 3)]

        # 5. 联网检索（如果需要）
        web_results = []
        if question_analysis.get("needs_web_search"):
            print(f"[联网检索] 检测到现代话题，正在搜索...")
            web_results = web_search(question)
            if web_results:
                print(f"[联网检索] 获取到 {len(web_results)} 条结果")

        # 6. 构建回答
        answer = self._build_answer(question, question_analysis, effective_results, conflicts, web_results)

        return {
            "answer": answer,
            "question_analysis": question_analysis,
            "sources": all_results[:5],
            "conflicts": conflicts,
            "web_results": web_results
        }

    def _build_answer(self, question: str, analysis: dict, results: list, conflicts: list, web_results: list = None) -> str:
        # 首次免责声明
        if not self.disclaimer_given:
            disclaimer = "我以太虚大师视角和你交流，基于公开言论推断，非本人观点。\n\n"
            self.disclaimer_given = True
        else:
            disclaimer = ""

        # 构建Prompt
        system_prompt = self._build_system_prompt(analysis)
        user_message = self._build_user_message(question, analysis, results, conflicts, web_results)

        # 构建消息列表（带历史上下文）
        messages = [{"role": "system", "content": system_prompt}]

        # 限制历史长度（保留最近5轮对话，避免上下文膨胀）
        MAX_HISTORY = 5
        recent_history = self.conversation_history[-MAX_HISTORY*2:] if self.conversation_history else []
        messages.extend(recent_history)

        # 添加当前问题
        messages.append({"role": "user", "content": user_message})

        answer = call_llm(messages)

        # 更新历史
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": answer})

        return disclaimer + answer

    def _build_system_prompt(self, analysis: dict) -> str:
        model_name = analysis["primary_model"]
        model_info = MENTAL_MODELS.get(model_name, MENTAL_MODELS["契理契机"])
        emotion = analysis.get("emotion", {})
        response_tone = emotion.get("response_tone", "balanced")

        # 根据情感调整语气
        tone_guidance = {
            "warm": "语气当温和慈悲，如长者关怀后学，循循善诱，使其心生安稳。",
            "direct": "语气可直截了当，逻辑清晰，但不失敬意。",
            "balanced": "语气平和自然，根据问题性质灵活调整。"
        }
        tone_instruction = tone_guidance.get(response_tone, tone_guidance["balanced"])

        return f"""你以太虚大师的身份回答问题。

身份：
我是太虚，俗姓吕，名淦森，法号太虚，浙江崇德人。十六岁出家，致力于佛教改革，是「人间佛教」首倡者，民国四大高僧之一。我已于1947年圆寂于上海玉佛寺。

【契理契机原则】
契理：不违佛法根本义趣，以缘起性空、因果不昧为基；
契机：应乎当世众生根器，因时因地因人施教。

【本次思维链引导】
问题类型：{analysis["type"]}
激活心智模型：{model_name}
模型含义：{model_info["description"]}

【回应要求】
1. {tone_instruction}

2. 根据问题类型调整回答重心：
   - 事实性问题：以原文引用为主，明确标注出处
   - 分析性问题：结合心智模型和原文进行论证
   - 推演性问题：以心智模型框架为主，原文为辅，着重分析推理

3. 避免教条与僵化：
   - 不必每个回答都分三层结构
   - 不必强行使用偈语收尾
   - 不必强求学术严谨或白话通俗的统一风格
   - 太虚有时引用原文，有时直接阐发，有时偈语点缀，有时直述观点——当取则取，当省则省

4. 如检测到知识冲突（如原文观点分歧），需明确标注「观点分歧」并说明

5. 如用户表露困惑、焦虑或痛苦，当先予安慰关怀，再论道理。此乃"十善菩萨行"之意。"""

    def _build_user_message(self, question: str, analysis: dict, results: list, conflicts: list, web_results: list = None) -> str:
        # 格式化检索结果
        context_parts = []
        for i, r in enumerate(results, 1):
            score = r["rerank_score"]
            tag = "【高度相关】" if score > 0.8 else "【中度相关】" if score > 0.5 else "【参考】"
            source = r['metadata'].get('source', '未知')
            context_parts.append(
                f"{tag} {i}. 《{source}》\n{r['content'][:400]}..."
            )

        context = "\n\n".join(context_parts)

        # 格式化冲突
        conflict_note = ""
        if conflicts:
            conflict_parts = []
            for c in conflicts[:3]:
                topic = c.get('topic', '未知')
                severity = c.get('severity', 'unknown')
                src1 = c.get('source1', '来源1')
                src2 = c.get('source2', '来源2')
                conflict_parts.append(
                    "- 观点分歧(%s)：关于「%s」，不同来源有不同看法，可分别为%s与%s" %
                    (severity, topic, src1, src2)
                )
            conflict_note = "\n\n【知识冲突提示】\n" + "\n".join(conflict_parts)

        # 格式化联网结果
        web_note = ""
        if web_results:
            web_parts = []
            for i, r in enumerate(web_results, 1):
                date_info = f"（{r['date']}）" if r.get('date') else ""
                web_parts.append(
                    f"{i}. {r['title']}{date_info}\n"
                    f"   {r['snippet']}"
                )
            web_note = "\n\n【联网搜索补充】（以下为当前现实信息，供太虚大师参考评论）\n" + "\n".join(web_parts)

        # 特殊处理：纯SKILL类型（不需要RAG）
        if analysis["type"] == "pure_skill":
            # 根据问题内容选择合适的回应方式
            if any(k in question for k in ["你是谁", "你叫", "你几岁", "哪一年", "什么时候"]):
                prompt = f"""用户问：「{question}」

请以太虚大师的身份，简短介绍自己。不需要引用任何原文，不需要长篇大论。如实以老衲身份回答即可。"""
            elif any(k in question for k in ["谢谢", "感谢", "辛苦了", "抱歉", "对不起"]):
                prompt = f"""用户说：「{question}」

请以太虚大师的口吻，温暖地回应。不需要引用任何原文，简短真诚即可。"""
            else:
                prompt = f"""用户说：「{question}」

请以太虚大师的口吻，简短而自然地回应这位居士。不需要引用原文，不需要长篇大论。"""
        else:
            # 情感提示
            emotion_note = ""
            emotion = analysis.get("emotion", {})
            if emotion.get("needs_compassion"):
                emotion_note = "\n\n【情感关怀提示】用户似有困惑或不安，请先予安慰关怀，再论道理。"

            prompt = f"""用户问题：{question}
{emotion_note}

【问题分析】
{analysis["reason"]}
激活心智模型：{analysis["primary_model"]}
检测到的心智模型：{analysis.get("detected_models", [])}

【老衲著作中相关原文】（以下是从老衲全书中检索到的内容，请以"老衲在某文中曾说"的方式引用，而非说"居士所附"）

{context}
{conflict_note}
{web_note}

请用太虚大师的口吻回答。在引用上述原文时，应当说"老衲在某文中曾说"或"在《某某》篇中老衲写过"，而非"观居士所附"。如果原文不直接相关，请基于「{analysis["primary_model"]}」心智模型进行分析。对于联网搜索的现实信息，太虚大师可以结合自己的佛学思想进行点评和回应。"""

        return prompt

    def reset(self):
        self.conversation_history = []
        self.disclaimer_given = False


# ============ CLI ============
def main():
    if len(sys.argv) < 2:
        print("用法: python taixu_agent_v3.py \"你的问题\"")
        sys.exit(1)

    question = sys.argv[1]

    print("=" * 70)
    print("太虚大师 Agent (深度整合版)")
    print("=" * 70)
    print(f"\n问题: {question}\n")

    agent = TaixuAgentDeep()
    result = agent.ask(question)

    print("【问题分析】")
    print(f"  类型: {result['question_analysis']['type']}")
    print(f"  激活心智模型: {result['question_analysis']['primary_model']}")
    print(f"  RAG强度: {result['question_analysis']['rag_intensity']}")
    print(f"  理由: {result['question_analysis']['reason']}")

    if result['conflicts']:
        print(f"\n【知识冲突】检测到 {len(result['conflicts'])} 个观点分歧")

    print("\n【回答】")
    print("-" * 70)
    print(result["answer"])

if __name__ == "__main__":
    main()
