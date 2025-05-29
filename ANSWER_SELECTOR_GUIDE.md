# 多候选答案生成与选择器使用指南

## 概述

答案选择器（Answer Selector）是RAG系统的第六步增强功能，实现了多候选答案生成、质量判别和最优选择的完整流程。该模块通过生成多个候选答案并使用LLM进行质量评估，最终选择最佳答案返回给用户。

## 功能特性

### 1. 智能复杂度判断
- **查询长度分析**：基于查询文本长度判断问题复杂度
- **实体数量统计**：通过图谱实体匹配数量评估复杂度
- **关键词检测**：识别"比较"、"分析"、"评估"等复杂问题关键词
- **自适应启用**：只对复杂问题启用多候选机制，节省计算资源

### 2. 多样化答案生成
- **参数变化策略**：使用不同的temperature参数生成多样化答案
  - 保守生成（temperature=0.3）：准确、保守的回答
  - 平衡生成（temperature=0.7）：平衡准确性和创新性
  - 创新生成（temperature=0.9）：创新性和多角度的回答
- **提示词优化**：根据生成参数调整提示词内容
- **长度控制**：自动限制答案长度，避免过长输出

### 3. 智能质量评估
- **多维度评分**：从准确性、相关性、完整性、清晰性、有用性五个维度评估
- **LLM评判**：使用同一模型进行答案质量比较
- **结构化输出**：标准化的评分格式和理由说明
- **自动排序**：按评分自动选择最佳答案

### 4. 资源优化
- **Token预算控制**：通过答案长度限制控制Token消耗
- **条件启用**：只对需要的复杂问题启用多候选机制
- **错误处理**：完善的异常处理和降级策略

## 配置说明

在 `config.yaml` 中添加以下配置：

```yaml
answer_selector:
  num_candidates: 3                    # 候选答案数量（建议2-3个）
  max_answer_length: 500               # 单个答案最大长度
  enable_for_complex_queries: true     # 是否启用复杂问题检测
  complexity_threshold:                # 复杂度判断阈值
    min_length: 50                     # 查询最小长度
    min_entities: 3                    # 最小实体数量
```

### 配置参数详解

- **num_candidates**: 生成的候选答案数量，建议2-3个以平衡质量和成本
- **max_answer_length**: 限制单个答案的最大字符数，防止过长输出
- **enable_for_complex_queries**: 是否启用智能复杂度判断
- **min_length**: 触发多候选机制的查询最小长度
- **min_entities**: 触发多候选机制的实体最小数量

## 使用方法

### 1. 基本使用

```python
from llm.llm import LLMClient
from llm.answer_selector import AnswerSelector

# 初始化
llm_client = LLMClient(config)
answer_selector = AnswerSelector(llm_client, config.get('answer_selector', {}))

# 生成最佳答案
answer, metadata = answer_selector.select_best_answer(
    query="请分析人工智能在医疗领域的应用前景",
    context="相关上下文信息...",
    entities=["人工智能", "医疗", "应用"]
)

print(f"答案: {answer}")
print(f"选择信息: {metadata}")
```

### 2. 集成到查询处理

答案选择器已自动集成到 `query_handler.py` 中，无需额外配置即可使用。系统会自动：

1. 判断查询复杂度
2. 选择合适的生成策略
3. 生成和评估候选答案
4. 返回最佳结果

### 3. 复杂度判断测试

```python
# 测试复杂度判断
should_use_multi = answer_selector.should_use_multi_candidate(
    query="什么是AI？",  # 简单问题
    entities=[]
)
print(f"是否使用多候选: {should_use_multi}")  # False

should_use_multi = answer_selector.should_use_multi_candidate(
    query="请详细分析深度学习和传统机器学习的优缺点比较",  # 复杂问题
    entities=["深度学习", "机器学习", "神经网络"]
)
print(f"是否使用多候选: {should_use_multi}")  # True
```

## 输出格式

### 选择元数据

```python
{
    'method': 'multi',                    # 'single' 或 'multi'
    'candidates_count': 3,                # 候选答案数量
    'best_score': 8.5,                   # 最佳答案评分
    'best_reasoning': '答案准确且全面',    # 选择理由
    'generation_params': {                # 生成参数
        'temperature': 0.7,
        'description': '平衡生成'
    },
    'all_scores': [8.5, 7.2, 6.8]       # 所有候选答案评分
}
```

### 查询结果显示

系统会在查询结果中显示答案选择信息：

```
📝 回答:
[生成的最佳答案内容]

🎯 答案选择: 多候选模式 (3个候选答案, 最佳评分: 8.5)
   选择理由: 答案准确且全面，很好地回答了用户问题

📚 参考来源 (5个):
[来源信息]

⏱️ 处理时间: 3.45秒 (增强检索)
```

## 测试和验证

### 运行测试脚本

```bash
# 完整功能测试
python test_answer_selector.py

# 查看评估提示词模板
python test_answer_selector.py template
```

### 测试用例

测试脚本包含三种类型的测试用例：

1. **简单问题**：短查询，少实体，应使用单一答案模式
2. **复杂问题**：长查询，多实体，应使用多候选答案模式
3. **中等复杂度**：测试边界情况

## 性能优化建议

### 1. Token成本控制
- 合理设置 `max_answer_length` 限制答案长度
- 根据实际需求调整 `num_candidates` 数量
- 优化复杂度判断阈值，避免不必要的多候选生成

### 2. 质量优化
- 根据具体领域调整评估标准
- 优化提示词模板提高生成质量
- 收集用户反馈持续改进评估机制

### 3. 性能监控
- 监控多候选模式的使用频率
- 跟踪答案质量评分分布
- 分析处理时间和资源消耗

## 扩展功能

### 1. 自定义评估标准

可以通过修改 `_build_evaluation_prompt` 方法自定义评估标准：

```python
def custom_evaluation_criteria(self, domain="general"):
    if domain == "medical":
        return "医学准确性、安全性、专业性"
    elif domain == "legal":
        return "法律准确性、条文引用、风险提示"
    return "准确性、相关性、完整性、清晰性、有用性"
```

### 2. 答案合并策略

未来可以实现多答案合并功能：

```python
def merge_answers(self, candidates):
    """合并多个候选答案的优点"""
    # 提取各答案的优势部分
    # 生成综合性答案
    pass
```

### 3. 用户偏好学习

可以根据用户反馈学习偏好：

```python
def learn_user_preference(self, user_feedback):
    """根据用户反馈调整评估权重"""
    # 更新评估标准权重
    # 优化候选答案生成策略
    pass
```

## 故障排除

### 常见问题

1. **答案选择器未启用**
   - 检查配置文件中的 `answer_selector` 配置
   - 确认 `enable_for_complex_queries` 设置为 `true`

2. **总是使用单一答案模式**
   - 降低 `complexity_threshold` 中的阈值
   - 检查查询是否包含复杂问题关键词

3. **评估失败**
   - 检查LLM连接状态
   - 确认模型支持长文本处理
   - 查看错误日志获取详细信息

4. **处理时间过长**
   - 减少 `num_candidates` 数量
   - 降低 `max_answer_length` 限制
   - 优化复杂度判断逻辑

### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 测试复杂度判断
print(f"查询长度: {len(query)}")
print(f"实体数量: {len(entities)}")
print(f"包含复杂关键词: {any(kw in query for kw in complex_keywords)}")
```

## 总结

答案选择器模块为RAG系统提供了强大的多候选答案生成和智能选择能力，通过以下特性显著提升了答案质量：

- ✅ **智能复杂度判断**：自动识别需要多候选处理的复杂问题
- ✅ **多样化生成策略**：使用不同参数生成多样化候选答案
- ✅ **LLM质量评估**：基于多维度标准进行智能评分
- ✅ **资源优化控制**：平衡答案质量和计算成本
- ✅ **无缝集成**：与现有查询处理流程完美集成

该模块特别适用于需要高质量答案的复杂查询场景，如分析、比较、评估类问题，能够显著提升用户体验和答案准确性。