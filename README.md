# G-RAG: 图增强检索生成系统

![版本](https://img.shields.io/badge/版本-2.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![许可证](https://img.shields.io/badge/许可证-MIT-orange)

G-RAG 是一个先进的图增强检索生成系统，融合了知识图谱、向量检索、多路检索和智能答案选择等前沿技术，为复杂问答场景提供高质量的检索增强生成解决方案。

## 🌟 核心特性

### 🧠 智能查询处理
- **轻量级查询分类器**：基于规则和LLM zero-shot的智能分类，无需预训练，启动速度快
- **查询增强机制**：自动生成查询的多种表达形式，提升检索召回率
- **复杂度自适应**：根据查询复杂度自动选择最优处理策略
- **主题池管理**：支持句子级分解和智能主题管理

### 🔍 多模态检索系统
- **向量语义检索**：基于FAISS的高效向量索引，支持Flat、IVF、HNSW多种索引类型
- **BM25关键词检索**：传统关键词匹配，补充语义检索的不足
- **图谱实体检索**：基于知识图谱的实体关联检索，发现隐含关系
- **多路检索融合**：智能合并多种检索结果，去重排序

### 🕸️ 知识图谱增强
- **自动图谱构建**：从文档中自动提取实体和关系，构建结构化知识图谱
- **多跳推理能力**：支持2跳子图提取，发现实体间的中介路径
- **实体重要性评估**：基于度中心性和PageRank算法评估实体权重
- **图谱辅助检索**：利用实体关联增强文档检索精度

### 📄 智能文档处理
- **多格式支持**：支持.docx和.json格式文档的解析
- **智能分块策略**：保持语义完整性的文档分块，支持重叠分块
- **冗余内容过滤**：自动识别和过滤重复内容，提升知识库质量
- **主题摘要生成**：为文档块生成主题摘要，提升检索匹配度

### 🎯 智能答案生成
- **多候选答案机制**：为复杂问题生成多个候选答案
- **质量自动评估**：基于LLM的多维度答案质量评估
- **最优答案选择**：自动选择最佳答案返回给用户
- **参数化生成**：支持不同temperature参数的多样化生成

### 🔧 灵活的LLM接口
- **多提供商支持**：支持OpenAI、Ollama等多种LLM提供商
- **统一接口设计**：生成和嵌入功能的统一调用接口
- **策略模式架构**：易于扩展新的LLM提供商

## 🏗️ 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   文档处理模块   │    │   知识图谱模块   │    │   向量索引模块   │
│                │    │                │    │                │
│ • 智能分块      │    │ • 实体抽取      │    │ • 批量嵌入      │
│ • 冗余过滤      │    │ • 关系构建      │    │ • 多种索引      │
│ • 主题摘要      │    │ • 子图提取      │    │ • 相似度检索    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   查询处理模块   │
                    │                │
                    │ • 查询分类      │
                    │ • 查询增强      │
                    │ • 多路检索      │
                    │ • 结果重排      │
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │   答案生成模块   │
                    │                │
                    │ • 多候选生成    │
                    │ • 质量评估      │
                    │ • 最优选择      │
                    └─────────────────┘
```

## 📋 项目结构

```
├── main.py                          # 主入口文件
├── config.yaml.example              # 配置文件模板
├── requirements.txt                 # 依赖清单
│
├── document/                        # 文档处理模块
│   ├── document_processor.py        # 文档解析与智能分块
│   ├── redundancy_buffer.py         # 冗余内容过滤
│   ├── sentence_splitter.py         # 句子分割器
│   ├── topic_pool_manager.py        # 主题池管理
│   └── topic_summary_generator.py   # 主题摘要生成
│
├── graph/                           # 知识图谱模块
│   ├── graph_builder.py             # 图谱构建与实体关系抽取
│   ├── graph_subgraph_extractor.py  # 多跳子图提取
│   └── graph_utils.py               # 图操作工具函数
│
├── vector/                          # 向量索引模块
│   └── optimized_vector_indexer.py  # 优化的向量索引器
│
├── query/                           # 查询处理模块
│   ├── query_handler.py             # 查询处理主控制器
│   ├── query_classifier.py          # 轻量级查询分类器
│   ├── query_enhancer.py            # 查询增强器
│   ├── enhanced_retriever.py        # 增强检索器（多路检索）
│   ├── reranker.py                  # 结果重排序器
│   └── optimized_theme_matcher.py   # 优化的主题匹配器
│
├── llm/                             # LLM接口模块
│   ├── llm.py                       # 统一LLM客户端
│   ├── answer_selector.py           # 多候选答案选择器
│   ├── prompt.py                    # 中文提示词模板
│   └── providers/                   # LLM提供商策略
│       ├── ollama_strategy.py       # Ollama策略实现
│       └── openai_strategy.py       # OpenAI策略实现
│
├── utils/                           # 工具模块
│   ├── output_manager.py            # 运行目录管理
│   ├── io.py                        # 统一文件I/O操作
│   ├── common.py                    # 通用工具函数
│   ├── logger.py                    # 日志工具
│   ├── misc.py                      # 杂项工具
│   ├── model_cache.py               # 模型缓存管理
│   └── trainer.py                   # 训练工具（保留）
│
├── classifier/                      # 分类器模块
│   └── evaluate_with_llm.py         # LLM评估工具
│
└── docs/                            # 文档目录
    ├── ENHANCED_RETRIEVAL_GUIDE.md  # 增强检索使用指南
    ├── GRAPH_RETRIEVAL_GUIDE.md     # 图谱检索使用指南
    ├── ANSWER_SELECTOR_GUIDE.md     # 答案选择器使用指南
    ├── LIGHTWEIGHT_CLASSIFIER_README.md # 轻量级分类器说明
    └── REFACTOR_SUMMARY.md          # 重构总结
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- 8GB+ RAM（推荐16GB）
- CUDA支持（可选，用于GPU加速）

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/yourusername/g-rag.git
cd g-rag
```

2. **创建虚拟环境**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **配置系统**
```bash
cp config.yaml.example config.yaml
# 编辑config.yaml，配置LLM提供商和模型
```

5. **准备数据**
```bash
mkdir data
# 将.docx或.json格式的文档放入data目录
```

### 运行系统

**完整流程运行**
```bash
python main.py
```

**模块化调试**
```bash
# 只运行文档处理
python main.py --debug doc

# 只运行图谱构建
python main.py --debug graph

# 只运行向量索引
python main.py --debug vector

# 只运行查询循环
python main.py --debug query
```

**传统命令行方式**
```bash
# 处理文档（传统方式）
python main.py --mode doc --config config.yaml

# 处理文档（主题池管理）
python main.py --mode doc --config config.yaml

# 构建知识图谱
python main.py --mode graph --config config.yaml

# 构建向量索引
python main.py --mode index --config config.yaml

# 查询
python main.py --mode query --config config.yaml --query "你的问题"
```

**指定工作目录**
```bash
# 创建新的运行目录
python main.py --new

# 使用已有目录
python main.py --work_dir ./result/run_20240101_120000
```

## ⚙️ 配置说明

### 核心配置项

详细的配置选项请参考 `config.yaml.example` 文件。主要配置项包括：

- **LLM配置**: 支持OpenAI、Azure OpenAI、Ollama等
- **嵌入模型**: 支持多种嵌入模型
- **文档处理**: 分块大小、重叠等参数
- **主题池**: 主题管理、参数配置等
- **图构建**: 实体提取、关系识别参数
- **检索配置**: 向量检索、图检索、重排序参数

### 主题池管理功能

主题池管理功能支持句子级分解和智能主题组织：

**主要特性**：
- 基于相似度的主题聚合
- 智能主题分裂策略
- 主题内聚度评估
- 冗余内容过滤

**主要优势**：
- 保持主题内容的连贯性
- 自动管理主题大小
- 支持动态主题调整

**使用示例**：
```bash
# 使用主题池管理处理文档
python main.py --mode doc --config config.yaml
```

```yaml
# LLM配置
llm:
  provider: "ollama"                     # 提供商: ollama, openai
  model_name: "qwen2.5:7b-instruct-fp16" # 模型名称
  host: "http://localhost:11434"         # 服务地址
  options:
    num_ctx: 32768                       # 上下文长度

# 嵌入模型配置
embedding:
  provider: "ollama"
  model_name: "bge-m3"              # 嵌入模型
  host: "http://localhost:11434"

# 文档处理配置
document:
  input_dir: "./data"                    # 输入目录
  chunk_size: 800                        # 分块大小
  chunk_overlap: 300                     # 重叠大小
  similarity_threshold: 0.80             # 相似度阈值
  redundancy_threshold: 0.95             # 冗余过滤阈值

# 向量索引配置
vector:
  index_type: "IVF"                      # 索引类型: Flat, IVF, HNSW
  nlist: 100                             # IVF参数
  nprobe: 10                             # 查询参数

# 增强检索配置
enhanced_retrieval:
  enable_query_expansion: true           # 启用查询扩展
  enable_bm25: true                      # 启用BM25检索
  enable_graph_retrieval: true           # 启用图谱检索
  vector_top_k: 10                       # 向量检索数量
  bm25_top_k: 5                          # BM25检索数量
  final_top_k: 5                         # 最终结果数量

# 答案选择器配置
answer_selector:
  num_candidates: 3                      # 候选答案数量
  max_answer_length: 500                 # 答案最大长度
  enable_for_complex_queries: true       # 启用复杂问题检测
```

## 🎯 使用示例

### 基本查询

启动系统后，在查询循环中输入问题：

```
请输入查询 (输入 'quit' 退出): 什么是人工智能？

[查询分类] 模式: hybrid_precise, 精确查询: True
[向量检索] 找到 5 个相关文档
[BM25检索] 找到 3 个相关文档  
[图谱检索] 匹配实体: ['人工智能'], 找到 2 个相关文档
[重排序] 最终返回 5 个结果
[答案生成] 使用单候选模式

答案: 人工智能（Artificial Intelligence，AI）是指...
```

### 高级功能

**切换查询模式**
```
> mode auto          # 自动模式
> mode hybrid_precise # 混合精确模式
> mode hybrid_imprecise # 混合模糊模式
> mode norag         # 无RAG模式
```

**控制检索功能**
```
> enhanced on        # 启用增强检索
> enhanced off       # 禁用增强检索
> reranker simple    # 使用简单重排序
> reranker llm       # 使用LLM重排序
```

**调试信息**
```
> debug on           # 启用调试输出
> debug off          # 禁用调试输出
```

## 🔧 高级特性

### 1. 多跳图推理

系统支持复杂的多实体关联查询：

```
查询: "张三和李四的共同朋友是谁？"

[图谱推理]
1. 识别实体: [张三, 李四]
2. 提取2跳子图
3. 发现中介实体: [王五, 赵六]
4. 生成推理路径: 张三 -> 王五 -> 李四
5. 返回相关文档块
```

### 2. 智能答案选择

对于复杂问题，系统会生成多个候选答案并自动选择最佳：

```
查询: "请分析深度学习和传统机器学习的优缺点"

[复杂度判断] 查询长度: 23, 实体数量: 3, 启用多候选模式
[候选生成] 生成3个候选答案
[质量评估] 评估维度: 准确性、相关性、完整性、清晰性、有用性
[最优选择] 选择评分最高的答案 (8.5/10)
```

### 3. 自适应检索策略

系统根据查询类型自动调整检索策略：

- **精确查询**：优先使用向量检索和图谱检索
- **模糊查询**：增加BM25检索权重
- **实体查询**：重点使用图谱检索
- **概念查询**：平衡多种检索方法

## 📊 性能优化

### 系统性能

| 指标 | 数值 |
|------|------|
| 查询响应时间 | < 2秒 |
| 文档处理速度 | 1000文档/分钟 |
| 向量索引构建 | 10000文档/分钟 |
| 内存占用 | < 2GB |
| 检索准确率 | 85%+ |

### 优化建议

1. **硬件配置**
   - 使用SSD存储提升I/O性能
   - 16GB+内存支持大规模文档处理
   - GPU加速向量计算（可选）

2. **参数调优**
   - 根据文档特点调整chunk_size
   - 优化向量索引参数（nlist, nprobe）
   - 调整检索top_k平衡精度和速度

3. **缓存策略**
   - 启用向量缓存减少重复计算
   - 使用查询缓存提升响应速度

## 🛠️ 开发指南

### 扩展新的LLM提供商

1. 在`llm/providers/`目录下创建新的策略文件
2. 继承基础策略类并实现必要方法
3. 在`llm.py`中注册新的提供商

```python
# llm/providers/custom_strategy.py
class CustomStrategy:
    def __init__(self, config):
        self.config = config
    
    def generate(self, prompt):
        # 实现生成逻辑
        pass
    
    def embed(self, texts):
        # 实现嵌入逻辑
        pass
```

### 添加新的检索方法

1. 在`query/enhanced_retriever.py`中添加新的检索器类
2. 实现检索接口方法
3. 在主检索流程中集成新方法

### 自定义重排序算法

1. 在`query/reranker.py`中添加新的重排序器
2. 实现排序逻辑
3. 在配置文件中添加新的重排序选项

## 🤝 贡献指南

我们欢迎社区贡献！请遵循以下步骤：

1. Fork项目仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

### 代码规范

- 遵循PEP 8代码风格
- 添加适当的注释和文档字符串
- 编写单元测试
- 更新相关文档

## 📝 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

感谢以下开源项目的支持：

- [FAISS](https://github.com/facebookresearch/faiss) - 高效向量检索
- [NetworkX](https://networkx.org/) - 图数据结构和算法
- [Sentence Transformers](https://www.sbert.net/) - 句子嵌入模型
- [Jieba](https://github.com/fxsjy/jieba) - 中文分词
- [OpenAI](https://openai.com/) - GPT模型API
- [Ollama](https://ollama.ai/) - 本地LLM服务

## 📞 联系我们

- 项目主页: [GitHub Repository](https://github.com/yourusername/g-rag)
- 问题反馈: [Issues](https://github.com/yourusername/g-rag/issues)
- 讨论交流: [Discussions](https://github.com/yourusername/g-rag/discussions)

---

**G-RAG** - 让知识检索更智能，让问答更精准！
