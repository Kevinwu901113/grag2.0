# G-RAG: Graph-enhanced Retrieval-Augmented Generation System

G-RAG 是一个模块化、可拓展的图增强检索生成系统，结合了文档结构建图、向量检索、查询分类与大语言模型响应，支持灵活的精确或泛化问答。

---

## 🔧 项目结构

```
├── main.py                          # 主入口
├── config.yaml                      # 配置文件
├── generate_training_data.py        # 利用LLM生成训练样本
├── finetune_from_document.py        # 从文档生成数据并微调分类器
├── requirements.txt                 # 依赖清单
├── result/                          # 每次运行生成目录
│
├── document/
│   └── document_processor.py        # 文档清洗、分块
│
├── graph/
│   ├── graph_builder.py             # 实体关系抽取 + 图构建
│   └── graph_utils.py               # 图操作工具函数
│
├── vector/
│   └── vector_indexer.py            # 嵌入生成 + FAISS索引
│
├── classifier/
│   ├── model.py                     # 分类器模型定义
│   ├── train_base_classifier.py     # 基础分类器训练
│   ├── finetune_classifier.py       # 分类器微调
│   └── evaluate_with_llm.py         # 使用LLM评估分类器
│
├── query/
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
- 参考 `config.yaml.example` 创建配置文件

### 🔄 步骤 3：完整运行一次
```bash
python main.py
```

### 🛠️ 调试单模块
```bash
python main.py --debug doc       # 仅运行文档预处理
python main.py --debug graph     # 仅构建图结构
python main.py --debug vector    # 仅构建向量索引
python main.py --debug classifier # 仅运行分类器
python main.py --debug query     # 仅运行查询模块
```

### ✍️ 生成分类器训练数据
```bash
python generate_training_data.py
```

### 📊 从文档生成数据并微调分类器
```bash
python finetune_from_document.py --base ./result/base --output ./result/fine_tuned
```

---

## 🧠 分类器逻辑（双判定）
| 结果 | 使用策略 |
|-------|-------------|
| norag + 不精确 | 仅图结构回答 |
| hybrid + 不精确 | 文本块 + 图结构联合回答 |
| hybrid + 精确   | 仅使用文本块精确回答 |

---

## 📦 中间产物
- `chunks.json`：文档分块结果
- `graph.json / graph.graphml`：实体关系图
- `vector.index`：向量检索索引
- `embedding_map.json`：向量索引映射
- `query_classifier.pkl`：查询分类模型
- `query_cache.jsonl`：历史查询缓存
- `continual_buffer.jsonl`：训练数据缓存
- `base_classifier_samples.jsonl`：基础分类器样本

---

## 📌 依赖安装
```bash
pip install -r requirements.txt
```

主要依赖包括：
- numpy, scikit-learn, faiss-cpu, joblib
- openai, requests
- python-docx
- torch
- networkx
- loguru

---

## 🙌 作者提示
- 支持中文任务
- LLM选择灵活（可接入 Ollama / OpenAI / Huggingface）
- 支持多种向量索引类型（Flat, IVF, HNSW）
- 支持分类器微调和评估
- 后续可拓展剪枝、图嵌入、RAG微调、GNN支持等
