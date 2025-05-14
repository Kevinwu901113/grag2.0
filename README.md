# G-RAG: Graph-enhanced Retrieval-Augmented Generation System

G-RAG 是一个模块化、可拓展的图增强检索生成系统，结合了文档结构建图、向量检索、查询分类与大语言模型响应，支持灵活的精确或泛化问答。

---

## 🔧 项目结构

```
G-RAG/
├── main.py                          # 主入口
├── config.yaml                      # 配置文件
├── generate_training_data.py        # 利用LLM生成训练样本
├── requirements.txt                # 依赖清单
├── result/                          # 每次运行生成目录
│
├── document/
│   └── document_processor.py        # 文档清洗、分块
│
├── graph/
│   └── graph_builder.py             # 实体关系抽取 + 图构建
│
├── vector/
│   └── vector_indexer.py            # 嵌入生成 + FAISS索引
│
├── query/
│   ├── query_classifier.py          # 查询分类器（模式 + 精度）
│   └── query_handler.py             # 查询输入、检索、生成
│
├── utils/
│   ├── output_manager.py            # 运行目录控制
│   ├── misc.py                      # 日志封装
│   └── trainer.py                   # loss/optimizer 工具
│
├── llm/
│   ├── llm.py                       # LLM接口（生成 + 嵌入 + 抽取）
│   ├── prompt.py                    # 中文提示词模板
│   └── providers/
│       ├── ollama_strategy.py       # Ollama策略
│       └── openai_strategy.py       # OpenAI策略
```

---

## 🚀 使用方式

### 📁 步骤 1：准备数据
- 将 `.docx` 或 `.json` 文档放入 `./data/` 目录下

### 🧱 步骤 2：配置 config.yaml
- 设置 `llm` 调用模型（如 Ollama / OpenAI）
- 可设定 chunk 参数、输出路径、是否缓存

### 🔄 步骤 3：完整运行一次
```bash
python main.py
```

### 🛠️ 调试单模块
```bash
python main.py --debug doc       # 仅运行文档预处理
python main.py --debug graph     # 仅构建图结构
python main.py --debug classifier
```

### ✍️ 生成分类器训练数据
```bash
python generate_training_data.py
```

---

## 🧠 分类器逻辑（双判定）
| 结果 | 使用策略 |
|-------|-----------|
| norag + 不精确 | 仅图结构回答 |
| hybrid + 不精确 | 文本块 + 图结构联合回答 |
| hybrid + 精确   | 仅使用文本块精确回答 |

---

## 📦 中间产物
- `chunks.json`：文档分块结果
- `graph.json / graph.graphml`：实体关系图
- `vector.index`：向量检索索引
- `query_classifier.pkl`：查询分类模型
- `query_cache.jsonl`：历史查询缓存
- `continual_buffer.jsonl`：训练数据缓存

---

## 📌 依赖安装
```bash
pip install -r requirements.txt
```

---

## 🙌 作者提示
- 支持中文任务
- LLM选择灵活（可接入 Ollama / OpenAI / Huggingface）
- 后续可拓展剪枝、图嵌入、RAG微调、GNN支持等
