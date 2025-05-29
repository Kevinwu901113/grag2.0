# 增强检索功能使用指南

## 概述

增强检索功能通过以下方式提升RAG系统的检索准确率：

1. **查询重写与扩展**：使用LLM生成查询的多种表达形式
2. **多路检索**：并行使用向量检索和BM25关键词检索
3. **结果合并与重排序**：智能合并不同检索方法的结果并重新排序

## 功能特性

### 1. 查询增强 (Query Enhancement)

- **同义词替换**：生成查询的同义表达
- **关键词提取**：提取核心关键词组合
- **英文翻译**：为中文查询生成英文版本
- **相关概念扩展**：扩展相关概念和术语

### 2. 多路检索 (Multi-Retrieval)

- **向量检索**：基于语义相似度的检索
- **BM25检索**：基于关键词匹配的传统检索
- **并行处理**：同时执行多种检索策略

### 3. 智能合并 (Intelligent Merging)

- **去重合并**：合并相同文档的多个检索结果
- **加权评分**：对多种方法都检索到的文档给予额外加分
- **统一重排序**：使用现有的重排序器进行最终排序

## 配置说明

在 `config.yaml` 中添加以下配置：

```yaml
enhanced_retrieval:                    # 增强检索配置
  enable_query_expansion: true         # 是否启用查询扩展
  enable_bm25: true                    # 是否启用BM25检索
  vector_top_k: 10                     # 向量检索返回数量
  bm25_top_k: 5                        # BM25检索返回数量
  final_top_k: 5                       # 最终返回结果数量
  bm25_params:                         # BM25参数
    k1: 1.5                            # 词频饱和参数
    b: 0.75                            # 长度归一化参数
```

## 使用方法

### 1. 在查询循环中使用

启动查询系统后，可以使用以下命令：

```bash
# 启用增强检索（默认开启）
enhanced on

# 禁用增强检索
enhanced off

# 切换查询模式
mode auto
mode hybrid_precise
mode hybrid_imprecise
mode norag

# 切换重排序器
reranker simple
reranker llm
reranker none
```

### 2. 编程接口使用

```python
from query.enhanced_retriever import EnhancedRetriever
from query.query_enhancer import get_query_enhancer

# 初始化增强检索器
retriever = EnhancedRetriever(config, work_dir)

# 执行检索
results = retriever.retrieve("人工智能的应用", top_k=5)

# 查询增强
query_enhancer = get_query_enhancer(config)
enhanced_queries = query_enhancer.enhance_query("机器学习算法")
```

### 3. 在查询处理中使用

```python
from query.query_handler import handle_query

# 使用增强检索
result = handle_query(
    query="深度学习的原理",
    config=config,
    work_dir=work_dir,
    mode="auto",
    use_enhanced_retrieval=True
)
```

## 测试功能

运行测试脚本验证功能：

```bash
python test_enhanced_retrieval.py
```

测试脚本将验证：
- 查询增强功能
- BM25检索功能
- 完整的增强检索流程

## 性能优化建议

1. **调整检索数量**：根据文档库大小调整 `vector_top_k` 和 `bm25_top_k`
2. **BM25参数调优**：根据文档特性调整 `k1` 和 `b` 参数
3. **查询扩展控制**：对于简单查询可以禁用查询扩展以提升速度
4. **重排序策略**：选择合适的重排序器平衡效果和性能

## 故障排除

### 常见问题

1. **查询扩展失败**
   - 检查LLM配置是否正确
   - 确认网络连接正常
   - 查看错误日志

2. **BM25检索无结果**
   - 确认文档已正确加载
   - 检查中文分词是否正常
   - 调整BM25参数

3. **向量检索失败**
   - 确认向量索引文件存在
   - 检查嵌入模型配置
   - 验证ID映射文件

### 调试模式

在代码中添加调试信息：

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 查看检索统计
print(f"检索统计: 向量{len(vector_candidates)}条, BM25{len(bm25_candidates)}条")
```

## 扩展开发

### 添加新的检索方法

1. 在 `EnhancedRetriever` 类中添加新的检索方法
2. 在 `retrieve` 方法中调用新方法
3. 更新配置文件添加相关参数

### 自定义查询扩展

1. 继承 `QueryEnhancer` 类
2. 重写 `enhance_query` 方法
3. 在配置中指定自定义增强器

## 版本历史

- **v1.0**: 初始版本，支持基本的查询扩展和多路检索
- 后续版本将添加更多检索策略和优化功能