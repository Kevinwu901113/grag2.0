# 图谱索引辅助检索机制使用指南

## 概述

图谱索引辅助检索机制是RAG系统的第五步增强功能，它通过知识图谱中的实体匹配来辅助文档检索，特别适用于涉及人名、机构名等实体的查询场景。

## 功能特点

### 1. 实体匹配检索
- 自动识别查询中的实体（人名、机构名等）
- 基于图谱中的实体-主题关系检索相关文档块
- 支持多实体联合检索

### 2. 智能去重与加权
- 自动去除与向量检索、BM25检索的重复结果
- 对多种检索方法都命中的文档给予额外加分
- 图谱检索结果享有更高的加权优先级

### 3. 实体重要性评估
- 基于度中心性计算实体权重
- 可选的PageRank算法评估实体重要性
- 重要实体连接的文档块获得更高优先级

## 核心组件

### 1. graph_utils.py 新增函数

#### `retrieve_by_entity(graph, matched_entities, chunks)`
基于匹配的实体检索相关文档块的核心函数。

**参数：**
- `graph`: NetworkX图对象，表示知识图谱
- `matched_entities`: 匹配到的实体名称集合
- `chunks`: 所有文档块列表

**返回：**
- 检索结果列表，包含相似度、检索类型等信息

**工作流程：**
1. 遍历匹配的实体，找到其连接的主题节点
2. 收集主题节点对应的topic_id
3. 根据topic_id匹配相应的文档块
4. 计算基于图谱的相似度分数
5. 按相似度排序返回结果

#### `calculate_entity_pagerank(graph, max_iter, alpha)`
计算实体节点的PageRank值，用于评估实体重要性。

**参数：**
- `graph`: 知识图谱
- `max_iter`: 最大迭代次数（默认100）
- `alpha`: 阻尼系数（默认0.85）

**返回：**
- 实体PageRank值字典

### 2. EnhancedRetriever 增强

#### 新增配置项
```yaml
enhanced_retrieval:
  enable_graph_retrieval: true         # 启用图谱检索
  graph_params:
    enable_pagerank: false             # 启用PageRank加权
    pagerank_alpha: 0.85               # PageRank阻尼系数
    entity_weight_factor: 0.1          # 实体权重影响因子
```

#### 检索流程集成
图谱检索已集成到主检索流程中：
1. 查询扩展
2. 向量检索
3. BM25检索
4. **图谱实体检索**（新增）
5. 去重合并
6. 重排序

## 使用方法

### 1. 配置启用

在配置文件中启用图谱检索：

```yaml
enhanced_retrieval:
  enable_graph_retrieval: true
```

### 2. 数据准备

确保工作目录中包含以下文件：
- `graph.json`: 知识图谱文件
- `chunks.json`: 文档块文件
- `vector.index`: 向量索引文件
- `embedding_map.json`: 向量映射文件

### 3. 代码调用

```python
from query.enhanced_retriever import EnhancedRetriever

# 创建检索器
config = {
    'enhanced_retrieval': {
        'enable_graph_retrieval': True,
        'final_top_k': 5
    }
}
retriever = EnhancedRetriever(config, work_dir)

# 执行检索
results = retriever.retrieve("张三在北京大学的研究工作")

# 查看结果
for result in results:
    print(f"文本: {result['text']}")
    print(f"相似度: {result['similarity']}")
    print(f"检索类型: {result.get('retrieval_types', result.get('retrieval_type'))}")
    if 'matched_entities' in result:
        print(f"匹配实体: {result['matched_entities']}")
```

## 检索效果

### 1. 适用场景
- **人名查询**: "张三的研究成果"
- **机构查询**: "北京大学的历史沿革"
- **复合实体查询**: "李四在清华大学的工作"

### 2. 性能优势
- **高召回率**: 通过图谱关系发现相关文档
- **精确匹配**: 实体匹配避免语义漂移
- **互补增强**: 与向量检索、BM25形成互补

### 3. 结果示例

查询："张三在北京大学的研究"

```
📊 检索统计: 向量8条, BM253条, 图谱2条, 去重后10条, 最终5条
🔍 图谱检索匹配到实体: {'张三', '北京大学'}

结果1: 张三教授在北京大学计算机系的研究工作...
  相似度: 1.200 (图谱+向量检索，20%加分)
  检索类型: ['vector', 'graph_entity']
  匹配实体: ['张三', '北京大学']
```

## 测试验证

运行测试文件验证功能：

```bash
python3 test_graph_retrieval.py
```

测试包括：
1. 实体匹配功能测试
2. 基于实体的检索测试
3. PageRank计算测试
4. 增强检索器集成测试

## 性能调优

### 1. 实体权重调整

通过配置调整实体权重的影响程度：

```yaml
graph_params:
  entity_weight_factor: 0.1  # 降低可减少权重影响
```

### 2. PageRank优化

启用PageRank可以更好地评估实体重要性：

```yaml
graph_params:
  enable_pagerank: true
  pagerank_alpha: 0.85      # 调整阻尼系数
```

### 3. 检索数量控制

图谱检索不设置单独的top_k限制，而是返回所有匹配的文档块，由最终的重排序控制输出数量。

## 注意事项

1. **图谱质量**: 检索效果依赖于知识图谱的质量和完整性
2. **实体识别**: 当前使用jieba分词进行实体匹配，可能存在识别误差
3. **性能影响**: 图谱检索增加了计算开销，但通常可以接受
4. **数据一致性**: 确保图谱中的topic_id与文档块中的topic_id一致

## 扩展建议

1. **更精确的实体识别**: 集成NER模型提高实体识别准确性
2. **关系权重**: 根据实体间关系类型设置不同权重
3. **动态图谱**: 支持图谱的动态更新和增量检索
4. **多跳检索**: 支持通过多跳关系发现更远距离的相关文档

## 总结

图谱索引辅助检索机制通过引入知识图谱的结构化信息，显著提升了RAG系统在实体相关查询场景下的检索效果。该机制与现有的向量检索、BM25检索形成有效互补，为用户提供更准确、更全面的检索结果。