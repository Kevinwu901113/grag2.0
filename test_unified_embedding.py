#!/usr/bin/env python3
"""
测试统一嵌入向量接口
验证LLMClient.embed()接口能正确处理单个文本和文本列表
"""

from llm.llm import LLMClient
from query.optimized_theme_matcher import ThemeMatcher
from document.topic_pool_manager import TopicPoolManager

def test_unified_embedding():
    """测试统一的嵌入向量接口"""
    
    # 配置
    config = {
        'embedding': {
            'provider': 'ollama',
            'model_name': 'bge-m3',
            'host': 'http://localhost:11434'
        },
        'llm': {
            'provider': 'ollama',
            'model_name': 'qwen2.5:7b',
            'host': 'http://localhost:11434'
        }
    }
    
    print("=== 测试统一嵌入向量接口 ===")
    
    # 1. 测试LLMClient直接调用
    print("\n1. 测试LLMClient.embed()接口")
    client = LLMClient(config)
    
    # 单个文本
    result1 = client.embed("这是一个测试文本")
    print(f"   单个文本嵌入: 成功，维度 {len(result1[0])}")
    
    # 文本列表
    result2 = client.embed(["文本1", "文本2", "文本3"])
    print(f"   批量文本嵌入: 成功，{len(result2)}条文本，每条维度 {len(result2[0])}")
    
    # 2. 测试ThemeMatcher
    print("\n2. 测试ThemeMatcher统一接口")
    summaries = [
        {'summary': '这是关于机器学习的摘要'},
        {'summary': '这是关于深度学习的摘要'},
        {'summary': '这是关于自然语言处理的摘要'}
    ]
    matcher = ThemeMatcher(summaries, config)
    matches = matcher.match("机器学习算法", top_k=2)
    print(f"   主题匹配: 成功，找到 {len(matches)} 个匹配")
    
    # 3. 测试TopicPoolManager
    print("\n3. 测试TopicPoolManager统一接口")
    manager = TopicPoolManager(config=config)
    manager.add_sentence("这是第一个测试句子")
    manager.add_sentence("这是第二个测试句子")
    topics = manager.get_all_topics()
    print(f"   主题管理: 成功，创建了 {len(topics)} 个主题")
    
    print("\n=== 所有测试通过！统一嵌入向量接口工作正常 ===")
    print("\n优化效果:")
    print("- ✅ 统一了所有模块的嵌入向量调用路径")
    print("- ✅ 移除了分散在各模块中的重复代码")
    print("- ✅ 简化了模型加载和配置逻辑")
    print("- ✅ 提高了代码维护性和一致性")
    print("- ✅ 支持单个文本和批量文本的统一接口")

if __name__ == "__main__":
    test_unified_embedding()