# G-RAG 优化版本

基于图增强的检索增强生成系统，经过全面优化重构。

## 🚀 核心特性

- **轻量级查询分类**：基于规则和LLM zero-shot的智能分类，无需训练
- **增强检索机制**：查询扩展 + 多路检索（向量+BM25）+ 智能重排序
- **图增强检索**：知识图谱辅助的实体匹配检索
- **多跳图推理**：支持复杂查询的多实体关联推理
- **智能答案选择**：多候选答案生成与质量评估
- **优化文档分块**：智能分块策略，保持语义完整性
- **统一接口设计**：模块化架构，易于扩展和维护

## 📋 项目结构

```
├── main.py                          # 主入口
├── config.yaml                      # 配置文件
├── requirements.txt                 # 依赖清单
├── result/                          # 运行结果目录
│
├── document/                        # 文档处理模块
│   ├── document_processor.py        # 智能文档分块
│   └── document_loader.py           # 文档加载器
│
├── vector/                          # 向量索引模块
│   ├── vector_index.py              # 向量索引构建
│   └── vector_retriever.py          # 向量检索器
│
├── graph/                           # 图模块
│   ├── graph_builder.py             # 知识图谱构建
│   ├── graph_retriever.py           # 图检索器
│   ├── graph_subgraph_extractor.py  # 子图提取器
│   └── graph_utils.py               # 图工具函数
│
├── query/                           # 查询模块
│   ├── query_handler.py             # 查询处理器
│   ├── query_classifier.py          # 轻量级查询分类器
│   ├── query_enhancer.py            # 查询增强器
│   ├── enhanced_retriever.py        # 增强检索器
│   ├── reranker.py                  # 重排序器
│   └── optimized_theme_matcher.py   # 优化主题匹配器
│
├── llm/                             # LLM模块
│   ├── llm_client.py                # LLM客户端
│   ├── answer_generator.py          # 答案生成器
│   └── answer_selector.py           # 答案选择器
│
├── utils/                           # 工具模块
│   ├── output_manager.py            # 输出管理
│   ├── misc.py                      # 日志工具
│   ├── io.py                        # 统一I/O操作
│   ├── common.py                    # 通用工具函数
│   └── model_cache.py               # 模型缓存
│
└── classifier/                      # 分类器模块（保留评估工具）
    └── evaluate_with_llm.py         # LLM评估工具
```

## 🛠️ 安装与配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置文件

复制并修改配置文件：

```bash
cp config.yaml.example config.yaml
```

主要配置项：
- LLM提供商设置（OpenAI/Ollama等）
- 嵌入模型配置
- 向量索引参数
- 图构建设置

### 3. 准备数据

将文档放入 `./data` 目录，支持格式：
- `.docx` - Word文档
- `.json` - JSON格式数据

## 🚀 使用方法

### 运行完整流程

```bash
python main.py
```

这将执行：
1. 文档加载与智能分块
2. 向量索引构建
3. 知识图谱构建
4. 交互式查询处理

### 单模块调试

可以通过修改各模块的测试代码进行单独调试和验证功能。

## 🧠 查询处理流程

### 1. 轻量级分类

系统使用规则匹配和LLM zero-shot判断对查询进行分类：
- **精确模式**：事实性查询，仅使用RAG检索
- **混合模式**：推理性查询，结合RAG检索和LLM生成
- **无RAG模式**：创意性查询，仅使用LLM生成

### 2. 增强检索

- **查询扩展**：同义词替换、关键词提取、概念扩展
- **多路检索**：向量检索 + BM25检索并行执行
- **图辅助检索**：实体匹配增强检索精度
- **智能重排序**：基于相关性和多样性的结果重排

### 3. 多跳推理

对于复杂查询，系统支持：
- 多实体识别与匹配
- 深度为2的子图提取
- 中介路径发现
- 关联关系推理

### 4. 答案生成与选择

- 生成多个候选答案（不同温度参数）
- LLM质量评估与打分
- 最优答案智能选择

## ⚙️ 高级配置

### LLM提供商

支持多种LLM提供商：
- OpenAI GPT系列
- Ollama本地模型
- 其他兼容OpenAI API的服务

### 向量索引类型

支持多种FAISS索引：
- `Flat`：精确搜索，适合小规模数据
- `IVF`：倒排索引，平衡精度和速度
- `HNSW`：图索引，高效近似搜索

### 性能优化

- 模型缓存：避免重复加载
- 向量缓存：减少重复计算
- 批处理：提高处理效率
- 内存优化：流式处理大文件

## 📈 优化亮点

1. **轻量化**：移除BERT分类器，使用规则+LLM zero-shot
2. **模块化**：统一接口设计，高度解耦
3. **智能化**：多路检索、智能重排、质量评估
4. **高效化**：缓存机制、批处理、内存优化
5. **可扩展**：插件化架构，易于添加新功能

## 📄 许可证

MIT License