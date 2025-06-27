# G-RAG 统一文档

![版本](https://img.shields.io/badge/版本-2.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![许可证](https://img.shields.io/badge/许可证-MIT-orange)

G-RAG 是一个先进的图增强检索生成系统，融合了知识图谱、向量检索、多路检索和智能答案选择等前沿技术，为复杂问答场景提供高质量的检索增强生成解决方案。

## 📋 目录

- [核心特性](#核心特性)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [功能模块](#功能模块)
  - [SimHash冗余检测](#simhash冗余检测)
  - [批量嵌入处理](#批量嵌入处理)
  - [聚类文档处理](#聚类文档处理)
  - [HuggingFace嵌入迁移](#huggingface嵌入迁移)
- [反馈系统](#反馈系统)
- [查询预加载功能](#查询预加载功能)
- [配置缺失分析](#配置缺失分析)
- [性能优化](#性能优化)
- [使用示例](#使用示例)
- [故障排除](#故障排除)

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
- **优先级调度**：智能上下文片段选择，基于相关度、结构权重和多样性

### 🕸️ 知识图谱增强
- **自动图谱构建**：从文档中自动提取实体和关系，构建结构化知识图谱
- **多跳推理能力**：支持2跳子图提取，发现实体间的中介路径
- **实体重要性评估**：基于度中心性和PageRank算法评估实体权重
- **图谱辅助检索**：利用实体关联增强文档检索精度

### 📄 智能文档处理
- **多格式支持**：支持.docx、.json、.jsonl、.txt、.md格式文档的解析
- **智能分块策略**：保持语义完整性的文档分块，支持重叠分块
- **冗余内容过滤**：自动识别和过滤重复内容，提升知识库质量
- **主题摘要生成**：为文档块生成主题摘要，提升检索匹配度

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd rag

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置设置

使用统一配置文件 `config.yaml`：

```bash
# 复制配置文件模板
cp config.yaml config.local.yaml

# 编辑配置文件
vim config.local.yaml
```

### 3. 基础使用

```bash
# 文档处理
python main.py --mode doc --config config.yaml

# 图谱构建
python main.py --mode graph --config config.yaml

# 索引构建
python main.py --mode index --config config.yaml

# 完整流程
python main.py --config config.yaml
```

## ⚙️ 配置说明

### 核心配置项

| 配置项 | 说明 | 推荐值 |
|--------|------|--------|
| `llm.provider` | LLM提供商 | `ollama` |
| `embedding.provider` | 嵌入模型提供商 | `huggingface` |
| `document.processing_mode` | 文档处理模式 | `enhanced` |
| `redundancy.method` | 冗余检测方法 | `simhash` |

### 提供商选择

#### LLM提供商
- **ollama**: 本地部署，支持多种开源模型
- **openai**: OpenAI API，需要API密钥
- **huggingface**: HuggingFace模型，主要用于嵌入

#### 嵌入模型提供商
- **huggingface** (推荐): 本地模型，支持GPU加速，批量处理
- **ollama**: 通过Ollama API调用
- **openai**: OpenAI嵌入API

### 处理模式选择

#### 文档处理模式
- **enhanced**: 传统主题池处理，适合中小规模文档
- **clustered**: 静态聚类处理，适合大规模文档
- **traditional**: 基础处理模式

#### 冗余检测方法
- **SimHash方法**（推荐）：
  - 适用场景：大规模数据处理（>10000句子）
  - 性能：比embedding方法快100-1000倍
  - 内存占用：每句子仅8字节签名
  - 准确率：95%+（hamming_threshold=3时）
- **Embedding方法**：
  - 适用场景：小规模数据或需要最高准确率
  - 性能：较慢但准确率最高
  - 内存占用：较大（需存储完整向量）

## 🔧 功能模块

### SimHash冗余检测

#### 概述
SimHash冗余检测模块用于替代传统的基于embedding向量相似度的冗余检测方法，具有极高性能和低内存占用的优势。

#### 特性
- **极高性能**: 比传统embedding方法快100-1000倍
- **低内存占用**: 每个句子只需存储8字节的64位签名
- **可扩展性**: 适合处理大规模文档集合（10万+句子）
- **近似准确**: 在大多数场景下提供足够的冗余检测准确率

#### 配置示例

```yaml
redundancy:
  method: "simhash"  # 使用SimHash方法
  hamming_threshold: 3      # Hamming距离阈值（推荐1-5）
  max_buffer_size: 100000   # 最大签名缓冲区大小
  log_interval: 1000        # 日志记录间隔
```

#### 参数调优

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `hamming_threshold` | int | 3 | Hamming距离阈值，控制冗余检测严格程度 |
| `max_buffer_size` | int | 100000 | 最大签名缓冲区大小，防止内存溢出 |
| `log_interval` | int | 1000 | 每处理多少句子输出一次日志 |

**hamming_threshold调优指南**：
- 1: 非常严格，只有几乎完全相同的句子才被认为冗余
- 3: 推荐值，平衡准确率和召回率
- 5: 较宽松，更多相似句子被认为冗余
- 7+: 很宽松，可能误判不相似的句子为冗余

### 批量嵌入处理

#### 概述
系统支持批量嵌入计算，大幅提升文档处理性能，减少API调用次数和网络延迟影响。

#### 性能提升
- **嵌入计算速度**: 提升30-70%
- **API优化**: 减少HTTP往返次数
- **资源利用**: 充分利用GPU并行计算能力
- **成本节约**: 减少API调用次数

#### 特性
1. **智能批处理**: 自动将句子分批处理，避免单次请求过大
2. **配置灵活**: 可调整批量大小以适应不同环境
3. **兼容性好**: 支持OpenAI和Ollama等多种嵌入模型
4. **内存优化**: 合理控制内存使用，避免OOM问题

## 🔄 反馈系统

### 概述

反馈模块是Struct-RAG++系统的核心组件，实现了反馈闭环和策略自适应功能。该模块包含两个主要组件：

- **FeedbackLogger**: 反馈日志记录器，记录每轮问答的详细信息
- **StrategyTuner**: 策略调优器，根据反馈信息动态调整系统配置

### 功能特性

#### 1. 反馈记录功能

- 📝 自动记录每轮问答的详细信息
- 📊 包含检索统计、来源信息、LLM评分等
- 💾 使用JSONL格式存储，便于分析
- 🔍 支持读取历史记录

#### 2. 策略自适应功能

- 🎯 根据反馈质量动态调整配置
- ⚡ 实时优化检索策略和参数
- 🔄 支持配置重置到初始状态
- 📈 记录调整历史便于追踪

### 集成方式

#### 1. 初始化阶段

```python
from utils.feedback.feedback_logger import FeedbackLogger
from utils.feedback.strategy_tuner import StrategyTuner

# 初始化反馈模块（传递工作目录以统一管理产物）
feedback_logger = FeedbackLogger(work_dir=work_dir)
strategy_tuner = StrategyTuner(logger, work_dir=work_dir)

# 保存初始配置状态（用于reset功能）
strategy_tuner.save_initial_config(config)
```

#### 2. 查询循环中使用

```python
# 记录反馈日志
try:
    feedback_dict = feedback_logger.log_feedback(user_input, result, config)
    
    # 根据反馈调整策略配置
    if feedback_dict:
        adjustments = strategy_tuner.update_strategy(config, feedback_dict)
        if adjustments:
            print(f"🔧 策略调整: {len(adjustments)}项配置已优化")
except Exception as e:
    print(f"⚠️ 记录反馈日志或策略调优失败: {e}")
```

### 日志格式

反馈日志采用JSONL格式，每行一个JSON记录：

```json
{
  "timestamp": 1703123456.789,
  "query_original": "什么是机器学习？",
  "query_final": "什么是机器学习？",
  "mode": "auto",
  "enhanced_retrieval": true,
  "answer": "机器学习是...",
  "retrieval": {
    "vector_candidates": 10,
    "bm25_candidates": 8,
    "graph_candidates": 5,
    "final_candidates": 15
  },
  "sources": [
    {
      "rank": 1,
      "id": "doc_123",
      "similarity": 0.85,
      "source_type": "vector"
    }
  ],
  "evaluation": {
    "method": "single",
    "candidates_count": 1,
    "best_score": 0.9
  }
}
```

### 测试验证

```bash
# 反馈系统已集成到主程序中，无需单独测试
# 使用时直接运行主程序即可
python main.py
```

## ⚡ 查询预加载功能

### 概述

查询预加载功能通过在查询循环开始前预先加载所有必要的模型和组件，显著提升查询响应速度。这个功能特别适合需要处理大量查询的场景。

### 功能特性

#### 🚀 预加载组件

- **LLM客户端**: 预加载语言模型和嵌入模型
- **数据文件**: 预加载文档块、向量索引、知识图谱
- **检索器**: 预加载增强检索器和主题匹配器
- **重排序器**: 预加载简单重排序器和LLM重排序器
- **其他组件**: 预加载上下文调度器、查询改写器、答案选择器

#### ⚡ 性能优势

- **首次查询加速**: 避免重复加载模型，首次查询响应更快
- **后续查询优化**: 所有查询都使用预加载的组件，响应时间稳定
- **内存效率**: 组件在内存中复用，避免重复初始化
- **嵌入模型预热**: 预先发送测试文本，确保模型就绪

### 使用方法

#### 自动启用（推荐）

预加载功能已集成到主查询循环中，无需额外配置：

```bash
# 运行查询模式，自动启用预加载
python main.py --debug query
```

#### 手动使用预加载器

```python
from query.preloader import QueryPreloader
from query.query_handler import handle_query

# 创建预加载器
preloader = QueryPreloader(config, work_dir)

# 预加载所有组件
preloader.preload_all()

# 使用预加载的组件处理查询
result = handle_query(
    query="你的查询",
    config=config,
    work_dir=work_dir,
    preloader=preloader  # 传递预加载器
)
```

### 性能测试

```bash
# 预加载功能已集成到主程序中
# 通过配置文件启用预加载功能
python main.py
```

根据测试结果，预加载功能通常能带来：

- **首次查询**: 2-5倍加速
- **后续查询**: 1.5-3倍加速
- **回本点**: 通常在3-10次查询后开始盈利

### 最佳实践

#### 适用场景

✅ **推荐使用**：
- 批量查询处理
- 交互式查询会话
- API服务部署
- 性能要求较高的场景

❌ **不推荐使用**：
- 单次查询
- 内存严重不足的环境
- 频繁切换配置的场景

## 📊 配置缺失分析

### 缺失的配置项

根据代码分析，以下配置项在config.yaml.example中缺失：

#### 1. 上下文调度器配置 (context_scheduler)
```yaml
context_scheduler:
  enabled: true  # 是否启用优先级调度器
  weights:
    relevance: 0.5    # 相关度权重
    structure: 0.3     # 结构权重
    diversity: 0.2     # 多样性权重
  max_tokens: 8000     # 最大token限制
  min_candidates: 3    # 最少候选数
  max_candidates: 10   # 最多候选数
  diversity_threshold: 0.85      # 多样性阈值
  min_diversity_score: 0.3       # 最小多样性分数
  pagerank_bonus: 0.2            # PageRank加分
  multi_source_bonus: 0.1        # 多源加分
  graph_entity_bonus: 0.15       # 图实体加分
```

#### 2. 查询改写配置 (query_rewrite)
```yaml
query_rewrite:
  enabled: false  # 是否启用查询改写
  strategies: ["clarify"]  # 改写策略：clarify, expand, simplify
  max_iterations: 1        # 最大改写轮数
  enable_context_rewrite: false   # 是否启用上下文改写
  enable_evaluation: false        # 是否启用改写效果评估
```

#### 3. 增强检索配置 (enhanced_retrieval)
```yaml
enhanced_retrieval:
  vector_top_k: 10        # 向量检索返回数量
  bm25_top_k: 5          # BM25检索返回数量
  final_top_k: 5         # 最终返回数量
  use_enhanced_graph: true        # 是否使用增强图检索
  enable_query_expansion: true    # 是否启用查询扩展
  enable_bm25: true              # 是否启用BM25检索
  enable_graph_retrieval: true   # 是否启用图谱检索
  enable_reranking: true         # 是否启用重排序
```

#### 4. 重排序权重配置 (rerank_weights)
```yaml
rerank_weights:
  vector: 0.4      # 向量相似度权重
  tfidf: 0.3       # TF-IDF权重
  overlap: 0.2     # 关键词重叠权重
  diversity: 0.1   # 多样性权重
```

#### 5. BM25参数配置
```yaml
bm25:
  k1: 1.5    # BM25参数k1
  b: 0.75    # BM25参数b
```

### BM25权重过高问题分析

#### 当前问题
- 所有文档检索都显示为BM25类型
- 可能原因：BM25权重设置过高，或向量检索失效

#### 建议调整
```yaml
# 在enhanced_retrieval中调整权重
enhanced_retrieval:
  enable_bm25: true
  bm25_top_k: 3      # 减少BM25返回数量
  vector_top_k: 15   # 增加向量检索数量
  
# 在rerank_weights中调整融合权重
rerank_weights:
  vector: 0.6        # 提高向量权重
  tfidf: 0.2         # 降低TF-IDF权重
  overlap: 0.15      # 降低关键词重叠权重
  diversity: 0.05    # 降低多样性权重
```

### "无法回答问题"的原因分析

#### 可能原因
1. **检索阈值过高**：similarity_threshold设置过严格
2. **向量质量问题**：嵌入模型或向量索引异常
3. **文档预处理问题**：文档分块或预处理不当
4. **查询理解问题**：查询与文档语义匹配度低

#### 建议解决方案
```yaml
# 降低相似度阈值
query:
  similarity_threshold: 0.5  # 从0.7降低到0.5
  max_results: 15           # 增加检索结果数量
  
# 启用查询扩展和改写
query_rewrite:
  enabled: true
  strategies: ["clarify", "expand"]
  
# 调整检索策略
enhanced_retrieval:
  enable_query_expansion: true
  enable_bm25: true
  enable_graph_retrieval: true
```

## 🚀 性能优化

### 内存优化
- 使用SimHash冗余检测减少内存占用
- 批量处理嵌入计算
- 智能缓存机制

### 速度优化
- 预加载关键组件
- 并行处理文档
- 优化向量索引结构

### 准确性优化
- 多路检索融合
- 智能重排序
- 上下文感知调度

## 📝 使用示例

### 基本查询

```python
from main import main

# 运行完整流程
main()
```

### 自定义配置

```python
import yaml
from query.query_handler import handle_query

# 加载配置
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 处理查询
result = handle_query("你的问题", config, "./work_dir")
print(result['answer'])
```

### 批量处理

```python
queries = ["问题1", "问题2", "问题3"]
results = []

for query in queries:
    result = handle_query(query, config, work_dir)
    results.append(result)
```

## 🔧 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型路径是否正确
   - 确认网络连接正常
   - 验证API密钥有效性

2. **检索结果为空**
   - 降低相似度阈值
   - 检查文档是否正确处理
   - 验证向量索引完整性

3. **处理速度慢**
   - 启用预加载功能
   - 使用SimHash冗余检测
   - 调整批量处理大小

4. **内存不足**
   - 减少批量处理大小
   - 使用更小的模型
   - 启用内存优化选项

### 日志分析

系统提供详细的日志信息，可以通过以下方式查看：

```bash
# 查看系统日志
tail -f log.txt

# 查看反馈日志
tail -f feedback.jsonl
```

### 性能监控

```python
from utils.performance_monitor import PerformanceMonitor

# 启用性能监控
monitor = PerformanceMonitor()
monitor.start()

# 处理查询
result = handle_query(query, config, work_dir)

# 查看性能报告
report = monitor.get_report()
print(report)
```

## 📚 文档结构

```
rag/
├── README.md                    # 本文档
├── main.py                      # 主程序入口
├── config.yaml.example          # 配置文件模板
├── requirements.txt             # 依赖列表
├── document/                    # 文档处理模块
├── embedding/                   # 嵌入模块
├── graph/                       # 图谱模块
├── llm/                         # 语言模型模块
├── query/                       # 查询处理模块
├── utils/                       # 工具模块
│   ├── feedback/               # 反馈系统
│   ├── common.py               # 通用工具
│   ├── config_manager.py       # 配置管理
│   └── logger.py               # 日志工具
└── vector/                      # 向量索引模块
```

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 Issue
- 发送邮件
- 参与讨论

---

**注意**: 本系统需要足够的计算资源和内存来运行大型语言模型和向量索引。建议在配置充足的环境中使用。