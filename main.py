# G-RAG 项目主入口

import argparse
import yaml
import os
from utils.output_manager import resolve_work_dir
from document.document_processor import run_document_processing
from graph.graph_builder import run_graph_construction
from vector.vector_indexer import run_vector_indexer
from query.query_classifier import run_query_classifier
from query.query_handler import run_query_loop
from generate_training_data import run_generate_training_data
from utils.misc import init_logger

def load_config(config_path="config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def check_and_generate_training_data(config, work_dir, logger):
    buffer_path = os.path.join(work_dir, "continual_buffer.jsonl")
    required_samples = config.get("classifier", {}).get("generate_samples", 30)
    existing = 0
    if os.path.exists(buffer_path):
        with open(buffer_path, 'r', encoding='utf-8') as f:
            existing = sum(1 for _ in f)
    if existing < required_samples:
        logger.info(f"检测到持续学习样本数量不足（当前: {existing}，期望: {required_samples}），开始生成...")
        run_generate_training_data(config, work_dir, logger)
    else:
        logger.info(f"持续学习样本充足，共有 {existing} 条，跳过生成。")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", type=str, default=None, help="只运行指定模块，如: doc, graph, vector, classifier, query")
    parser.add_argument("--new", action="store_true", help="创建新的运行目录")
    parser.add_argument("--work_dir", type=str, help="指定已有运行目录")
    args = parser.parse_args()

    config = load_config()
    work_dir = resolve_work_dir(config, args)
    logger = init_logger(work_dir)
    logger.info(f"当前工作目录: {work_dir}")

    # 调度逻辑
    if args.debug == "doc":
        run_document_processing(config, work_dir, logger)
        check_and_generate_training_data(config, work_dir, logger)
        return
    elif args.debug == "graph":
        run_graph_construction(config, work_dir, logger)
        return
    elif args.debug == "vector":
        run_vector_indexer(config, work_dir, logger)
        return
    elif args.debug == "classifier":
        run_query_classifier(config, work_dir, logger)
        return
    elif args.debug == "query":
        run_query_loop(config, work_dir, logger)
        return

    # 默认完整流程
    run_document_processing(config, work_dir, logger)
    check_and_generate_training_data(config, work_dir, logger)
    run_graph_construction(config, work_dir, logger)
    run_vector_indexer(config, work_dir, logger)
    run_query_classifier(config, work_dir, logger)
    run_query_loop(config, work_dir, logger)

if __name__ == "__main__":
    main()
