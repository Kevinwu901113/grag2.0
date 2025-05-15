import os
import yaml
import argparse
from utils.misc import init_logger
from utils.output_manager import resolve_work_dir
from document.document_processor import run_document_processing
from generate_base_classifier_data import generate_queries
from classifier.finetune_classifier import finetune_model

def main(config_path, base_model_dir, output_model_dir=None, model_name=None):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    args = argparse.Namespace(new=False, work_dir=None)
    work_dir = resolve_work_dir(config, args)
    logger = init_logger(work_dir)
    logger.info(f"当前工作目录: {work_dir}")

    logger.info("=== 步骤1：处理文档 ===")
    run_document_processing(config, work_dir, logger)

    logger.info("=== 步骤2：生成训练数据 ===")
    buffer_path = os.path.join(work_dir, "continual_buffer.jsonl")
    sample_num = config.get("classifier", {}).get("generate_samples", 100)
    generate_queries(config, sample_num, buffer_path, logger)

    logger.info("=== 步骤3：微调分类器 ===")
    finetune_model(
        base_dir=base_model_dir,
        data_path=buffer_path,
        output_path=output_model_dir or work_dir,
        model_name=model_name
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--base", type=str, required=True, help="base模型目录")
    parser.add_argument("--output", type=str, required=False, help="输出微调模型目录（默认与工作目录一致）")
    parser.add_argument("--model", type=str, default="bert-base-chinese", help="BERT模型名称")
    args = parser.parse_args()

    main(args.config, args.base, args.output, args.model)
