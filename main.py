# G-RAG 项目主入口

import argparse
import yaml
import sys, os
from utils.output_manager import resolve_work_dir
from utils.misc import init_logger

# 主流程模块
from document.document_processor import run_document_processing
from document.enhanced_document_processor import run_enhanced_document_processing
from graph.graph_builder import run_graph_construction
from vector.optimized_vector_indexer import run_vector_indexer
from vector.entity_vector_indexer import run_entity_vector_indexer
from query.query_handler import run_query_loop

# 轻量级分类器模块（无需训练）
# 移除了原有的BERT训练和微调模块

def load_config(config_path="config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# 轻量级分类器无需训练数据生成和微调
# 移除了原有的训练数据检查和微调函数

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", type=str, default=None, help="只运行指定模块，如: doc, enhanced_doc, graph, vector, entity_vector, classifier, query")
    parser.add_argument("--new", action="store_true", help="创建新的运行目录")
    parser.add_argument("--work_dir", type=str, help="指定已有运行目录")
    # 移除高级聚类参数
    args = parser.parse_args()

    print("[DEBUG] 开始加载配置...")
    config = load_config()
    print("[DEBUG] 配置加载完成")
    
    print("[DEBUG] 开始解析工作目录...")
    work_dir = resolve_work_dir(config, args)
    print(f"[DEBUG] 工作目录解析完成: {work_dir}")
    
    print("[DEBUG] 开始初始化日志系统...")
    logger = init_logger(work_dir)
    print("[DEBUG] 日志系统初始化完成")
    
    logger.info(f"当前工作目录: {work_dir}")

    if args.debug == "doc":
        run_document_processing(config, work_dir, logger)
        return
    elif args.debug == "enhanced_doc":
        run_enhanced_document_processing(config, work_dir, logger)
        return
    elif args.debug == "graph":
        run_graph_construction(config, work_dir, logger)
        return
    elif args.debug == "vector":
        run_vector_indexer(config, work_dir, logger)
        return
    elif args.debug == "entity_vector":
        run_entity_vector_indexer(config, work_dir, logger=logger)
        return
    elif args.debug == "classifier":
        # 轻量级分类器无需训练，直接跳过
        logger.info("轻量级分类器无需训练，跳过分类器模块")
        return
    elif args.debug == "query":
        run_query_loop(config, work_dir, logger)
        return

    # 默认完整流程（跳过分类器训练）
    # 统一使用增强文档处理（移除聚类功能后的主题池处理）
    logger.info("开始执行增强文档处理...")
    run_enhanced_document_processing(config, work_dir, logger)
    logger.info("增强文档处理完成")
    
    logger.info("使用轻量级分类器，跳过分类器训练步骤")
    
    run_graph_construction(config, work_dir, logger)
    run_vector_indexer(config, work_dir, logger)
    run_entity_vector_indexer(config, work_dir, logger=logger)
    run_query_loop(config, work_dir, logger)

if __name__ == "__main__":
    main()
