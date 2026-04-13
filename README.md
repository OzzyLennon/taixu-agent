# 太虚大师 Agent

以民国四大高僧·太虚法师视角的 AI 对话系统，基于 Skill + RAG 深度整合。

## 特性

- **人格框架**：太虚大师 SKILL.md 定义身份与思维模式
- **RAG 知识库**：基于太虚全书（15000+ 原文块）向量检索
- **动态 RAG 触发**：根据问题类型自动调整 RAG 强度
- **联网检索**：现代话题自动联网补充现实信息
- **思维链引导**：激活太虚六大心智模型（契理契机、即人成佛等）
- **历史会话**：类似 ChatGPT 的多会话管理

## 安装

### 1. 克隆仓库

```bash
git clone <repo-url>
cd taixu-perspective/rag
```

### 2. 安装依赖

```bash
pip install flask chromadb requests
```

### 3. 配置 API Key

复制配置文件并填入你的硅基流动 API Key：

```bash
cp config.py.example config.py
# 编辑 config.py，填入你的 API Key
```

### 4. 构建知识库（首次使用）

需要准备太虚全书原文文本，放置到 `../sources/raw_text/`，然后运行：

```bash
python embed_texts.py
```

### 5. 启动

```bash
python app.py
```

访问 http://127.0.0.1:5000

## 项目结构

```
rag/
├── app.py              # Flask Web 服务
├── chat.py             # 命令行对话模式
├── taixu_agent_v3.py  # 核心 Agent（推荐使用）
├── taixu_agent_v2.py  # 溯源分层版
├── taixu_agent.py     # 基础版
├── embed_texts.py     # 知识库索引脚本
├── retrieval.py       # 检索模块
├── config.py.example  # 配置文件模板
├── templates/
│   └── index.html     # Web 前端
└── chromadb/          # 向量数据库（不提交）
```

## API 配置

使用 [硅基流动](https://siliconflow.cn/) API：

- Embedding: `Qwen/Qwen3-Embedding-8B`
- Reranker: `Qwen/Qwen3-Reranker-8B`
- LLM: `Pro/deepseek-ai/DeepSeek-V3.2`

## 命令行使用

```bash
python chat.py
```

## 免责声明

本项目仅供学习和研究使用。太虚大师的回答基于公开言论和佛学著作推断，不代表历史真实观点。
