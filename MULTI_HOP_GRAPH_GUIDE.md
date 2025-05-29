# 多跳图推理与摘要融合使用指南

## 概述

多跳图推理功能是对原有图谱检索的重要增强，专门用于处理包含多个实体的复杂查询。当查询中匹配到2个或更多实体时，系统会自动启用多跳推理，发现实体间的中介路径和关联关系。

## 核心功能

### 1. 深度为2的子图提取

- **功能**: 提取包含查询实体及其2跳邻居的子图
- **优势**: 发现实体间的间接关联和中介节点
- **应用场景**: 复杂人物关系分析、机构关联发现

```python
from graph.graph_subgraph_extractor import extract_multi_hop_subgraph

# 提取多跳子图
subgraph = extract_multi_hop_subgraph(graph, matched_entities, max_depth=2)
```

### 2. 中介实体路径发现

- **功能**: 自动发现连接查询实体的中介节点
- **路径格式**: 实体A -> 中介实体 -> 实体B
- **权重计算**: 基于节点度中心性评估中介实体重要性

```python
from graph.graph_subgraph_extractor import find_intermediate_entities

# 发现中介实体路径
intermediate_info = find_intermediate_entities(graph, source_entities)
for info in intermediate_info:
    print(f"{info['source']} -> {info['intermediate']} -> {info['target']}")
```

### 3. 简洁摘要生成

- **长度控制**: 默认最大400字符，避免超过LLM输入限制
- **内容优先级**: 直接关系 > 中介关系 > 相关实体
- **智能截断**: 超长时优先保留重要信息

```python
from graph.graph_subgraph_extractor import generate_concise_graph_summary

# 生成简洁摘要
summary = generate_concise_graph_summary(
    subgraph, matched_entities, max_summary_length=400
)
```

## 摘要模板说明

### 摘要结构

1. **直接关系**: "实体A与实体B存在关系类型关系"
2. **中介关系**: "实体A通过中介实体与实体B关联（关系1-关系2）"
3. **相关实体**: "重要实体（连接N个节点）"

### 示例输出

```
直接关系：张三与人工智能存在研究关系；
中介关系：张三通过北京大学与李四关联（任职于-合作）；
相关实体：华为公司（连接5个节点）、机器学习（连接3个节点）。
```

## 集成到检索系统

### 自动触发条件

多跳图推理在以下情况自动启用：
- 查询中匹配到2个或更多实体
- 图谱数据可用
- 实体间存在可达路径

### 检索结果增强

启用多跳推理后，检索结果会包含以下额外信息：

```python
{
    'id': 'doc_001',
    'text': '原始文档内容',
    'similarity': 0.85,
    'retrieval_type': 'graph_entity',
    'matched_entities': ['张三', '李四'],
    'graph_summary': '多跳推理摘要',  # 新增
    'multi_hop_entities': ['张三', '李四'],  # 新增
    'intermediate_paths': [  # 新增
        {
            'source': '张三',
            'intermediate': '北京大学',
            'target': '李四',
            'relations': ['任职于', '合作']
        }
    ]
}
```

## 性能优化策略

### 1. 路径数量限制

- 最短路径搜索限制为前3条
- 中介路径信息限制为前5条
- 避免图遍历过度消耗资源

### 2. 摘要长度控制

- 默认最大长度400字符
- 可根据LLM模型调整
- 智能截断保留核心信息

### 3. 边权重排序

- 基于节点度中心性排序
- 优先选择重要关系
- 提升摘要质量

## 配置参数

### EnhancedRetriever配置

```yaml
enhanced_retrieval:
  enable_graph_retrieval: true  # 启用图谱检索
  vector_top_k: 5
  bm25_top_k: 3
  final_top_k: 5
```

### 多跳推理参数

```python
# 子图提取参数
max_depth = 2  # 最大搜索深度

# 摘要生成参数
max_summary_length = 400  # 最大摘要长度
max_paths = 5  # 最大路径数量
max_edges = 10  # 最大边数量
```

## 使用示例

### 1. 人物关系分析

**查询**: "张三和李四的合作研究"

**多跳推理结果**:
- 直接关系：张三与李四存在合作研究关系
- 中介关系：张三通过北京大学与李四关联（任职于-合作）
- 相关实体：人工智能（连接4个节点）

### 2. 机构关联发现

**查询**: "北京大学与华为公司的产学研合作"

**多跳推理结果**:
- 中介关系：北京大学通过张三与华为公司关联（雇佣-项目合作）
- 中介关系：北京大学通过人工智能与华为公司关联（研究-应用）

### 3. 技术概念关联

**查询**: "人工智能在自然语言处理中的应用"

**多跳推理结果**:
- 直接关系：人工智能与自然语言处理存在包含关系
- 中介关系：人工智能通过深度学习与自然语言处理关联（包含-应用于）

## 最佳实践

### 1. 图谱构建建议

- 确保实体命名一致性
- 添加丰富的关系类型
- 维护适当的图谱密度

### 2. 查询优化

- 使用具体的实体名称
- 避免过于宽泛的概念
- 结合上下文信息

### 3. 摘要长度调优

- 根据下游LLM模型调整
- 平衡信息完整性和简洁性
- 监控摘要质量

## 故障排除

### 常见问题

1. **无多跳推理结果**
   - 检查实体匹配是否正确
   - 确认图谱中存在相关路径
   - 验证图谱数据完整性

2. **摘要过长**
   - 调整max_summary_length参数
   - 减少max_edges限制
   - 优化关系重要性排序

3. **性能问题**
   - 限制搜索深度
   - 减少路径数量
   - 优化图谱结构

### 调试方法

```python
# 启用详细日志
print(f"🔍 图谱检索匹配到实体: {matched_entities}")
print(f"📊 多跳图推理摘要: {multi_hop_summary[:100]}...")
print(f"🔗 发现{len(intermediate_info)}条中介实体路径")
```

## 扩展功能

### 1. 自定义权重算法

可以替换默认的度中心性权重计算：

```python
def custom_weight_function(graph, node):
    # 自定义权重计算逻辑
    return custom_score
```

### 2. 关系类型过滤

可以根据关系类型过滤路径：

```python
allowed_relations = {'合作', '任职于', '研究'}
filtered_paths = [p for p in paths if all(r in allowed_relations for r in p['relations'])]
```

### 3. 动态深度调整

根据查询复杂度动态调整搜索深度：

```python
max_depth = 3 if len(matched_entities) > 3 else 2
```

## 总结

多跳图推理功能显著增强了系统处理复杂关系查询的能力，特别适用于：

- 人物关系网络分析
- 机构关联发现
- 技术概念关联
- 跨领域知识连接

通过合理配置参数和优化图谱结构，可以获得高质量的推理结果，同时保持良好的性能表现。