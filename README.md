# G-RAG: 图增强检索生成系统

![版本](https://img.shields.io/badge/版本-1.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![许可证](https://img.shields.io/badge/许可证-MIT-orange)

G-RAG 是一个模块化、可拓展的图增强检索生成系统，结合了文档结构建图、向量检索、查询分类与大语言模型响应，支持灵活的精确或泛化问答。系统通过知识图谱增强传统RAG流程，提供更全面的上下文理解和推理能力。

## 🌟 核心特性

- **多模态文档处理**：支持 `.docx` 和 `.json` 格式文档的解析和分块
- **知识图谱构建**：自动从文本中提取实体和关系，构建结构化知识图谱
- **向量索引检索**：支持多种索引类型（Flat、IVF、HNSW），优化检索效率
- **查询意图分类**：基于BERT的查询分类器，支持精确/混合/无RAG多种模式
- **主题匹配**：基于语义相似度的主题匹配，提高检索精度
- **冗余过滤**：智能过滤重复内容，提高知识库质量
- **灵活LLM接口**：支持多种LLM提供商（OpenAI、Ollama等）
- **可微调分类器**：支持从现有文档生成训练数据并微调分类器

## 📋 项目结构

```
├── main.py                          # 主入口
├── config.yaml                      # 配置文件
├── generate_training_data.py        # 利用LLM生成训练样本
├── finetune_from_document.py        # 从文档生成数据并微调分类器
├── requirements.txt                 # 依赖清单
├── result/                          # 每次运行生成目录
│
├── document/                        # 文档处理模块
│   ├── document_processor.py        # 文档清洗、分块
│   ├── redundancy_buffer.py         # 冗余内容过滤
│   ├── sentence_splitter.py         # 句子分割
│   ├── topic_pool_manager.py        # 主题池管理
│   └── topic_summary_generator.py   # 主题摘要生成
│
├── graph/                           # 图构建模块
│   ├── graph_builder.py             # 实体关系抽取 + 图构建
│   └── graph_utils.py               # 图操作工具函数
│
├── vector/                          # 向量索引模块
│   └── vector_indexer.py            # 嵌入生成 + FAISS索引
│
├── classifier/                      # 分类器模块
│   ├── model.py                     # 分类器模型定义
│   ├── train_base_classifier.py     # 基础分类器训练
│   ├── finetune_classifier.py       # 分类器微调
│   └── evaluate_with_llm.py         # 使用LLM评估分类器
│
├── query/                           # 查询模块
│   ├── query_handler.py             # 查询输入、检索、生成
│   └── theme_matcher.py             # 主题匹配器
│
├── utils/                           # 工具模块
│   ├── output_manager.py            # 运行目录控制
│   ├── misc.py                      # 日志封装
│   └── trainer.py                   # loss/optimizer 工具
│
├── llm/                             # LLM接口模块
│   ├── llm.py                       # LLM接口（生成 + 嵌入 + 抽取）
│   ├── prompt.py                    # 中文提示词模板
│   └── providers/                   # LLM提供商策略
│       ├── ollama_strategy.py       # Ollama策略
│       └── openai_strategy.py       # OpenAI策略
```

## 🔧 安装指南

### 环境要求

- Python 3.8+
- CUDA (可选，用于GPU加速)

### 安装步骤

1. 克隆仓库

```bash
git clone https://github.com/yourusername/g-rag.git
cd g-rag
```

2. 创建虚拟环境（推荐）

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

3. 安装依赖

```bash
pip install -r requirements.txt
```

4. 配置

```bash
cp config.yaml.example config.yaml
# 编辑 config.yaml 设置您的配置
```

## 🚀 使用方法

### 准备数据

将 `.docx` 或 `.json` 文档放入 `./data/` 目录下（可在配置文件中修改）

### 配置系统

编辑 `config.yaml` 文件，设置：

- LLM提供商和模型（OpenAI或Ollama）
- 文档处理参数（分块大小、重叠度等）
- 向量索引类型和参数
- 分类器训练参数

### 运行完整流程

```bash
python main.py
```

这将依次执行：
1. 文档处理和分块
2. 分类器样本生成（如需要）
3. 分类器微调（如配置启用）
4. 知识图谱构建
5. 向量索引创建
6. 启动交互式查询界面

### 单模块调试

```bash
python main.py --debug doc       # 仅运行文档预处理
python main.py --debug graph     # 仅构建图结构
python main.py --debug vector    # 仅构建向量索引
python main.py --debug classifier # 仅运行分类器
python main.py --debug query     # 仅运行查询模块
```

### 生成分类器训练数据

```bash
python generate_training_data.py
```

### 从文档生成数据并微调分类器

```bash
python finetune_from_document.py --base ./result/base --output ./result/fine_tuned
```

## 🧠 查询分类逻辑

系统使用双判定机制来决定最佳回答策略：

| 分类结果 | 使用策略 |
|-------|-------------|
| norag + 不精确 | 仅使用图结构回答 |
| hybrid + 不精确 | 文本块 + 图结构联合回答 |
| hybrid + 精确   | 仅使用文本块精确回答 |

## 📦 中间产物

运行过程中会在工作目录生成以下文件：

- `chunks.json`：文档分块结果
- `graph.json` / `graph.graphml`：实体关系图
- `vector.index`：向量检索索引
- `embedding_map.json`：向量索引映射
- `query_classifier.pt`：查询分类器模型
- `label_encoder.pkl`：标签编码器
- `query_cache.jsonl`：查询历史缓存
- `log.txt`：运行日志

## 🔄 工作流程

1. **文档处理**：解析文档 → 分句 → 分块 → 生成主题摘要 → 过滤冗余
2. **图构建**：提取实体 → 提取关系 → 构建知识图谱
3. **向量索引**：生成文本嵌入 → 构建FAISS索引
4. **查询处理**：分类查询 → 检索相关文本 → 提取子图 → 生成回答

## 🛠️ 高级配置

### LLM提供商

支持多种LLM提供商：

- **OpenAI**：设置 `provider: "openai"` 和相应的 API 密钥
- **Ollama**：设置 `provider: "ollama"` 和本地服务地址

### 向量索引类型

支持三种索引类型：

- **Flat**：最精确但最慢
- **IVF**：平衡速度和精度
- **HNSW**：最快但精度略低

## 📊 性能优化

- 使用 GPU 加速模型推理
- 调整 `chunk_size` 和 `chunk_overlap` 优化文档分块
- 选择合适的向量索引类型和参数
- 启用缓存减少重复计算

## 🤝 贡献

欢迎贡献代码、报告问题或提出改进建议！

## 📄 许可证

本项目采用 MIT 许可证 - 详见 LICENSE 文件
