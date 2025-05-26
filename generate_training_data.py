import os
import json
import random
import yaml
import argparse
from llm.llm import LLMClient
from utils.misc import init_logger
from utils.output_manager import resolve_work_dir

def load_chunks(work_dir):
    path = os.path.join(work_dir, "chunks.json")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_query_prompt(text):
    return (
        f"请基于以下内容构造一个用户查询：\n{text}\n\n"
        f"并回答：该问题是否涉及精确规定、具体条款、法律规则或时间数字？\n"
        f"是否需要原始文档支持？（如涉及具体措辞）\n\n"
        f"请严格用以下格式返回：\n"
        f"(查询内容, 是否需要精确回答, 是否需要文本块支持)\n"
        f"例如：\n"
        f"(中华人民共和国土地改革法是哪一年颁布的？, 是, 是)"
    )

def run_generate_training_data(config: dict, work_dir: str, logger):
    num_samples = config.get("classifier", {}).get("generate_samples", 30)
    output_path = os.path.join(work_dir, "continual_buffer.jsonl")

    llm = LLMClient(config)
    chunks = load_chunks(work_dir)
    # selected = random.sample(chunks, min(num_samples, len(chunks)))
    selected = random.choices(chunks, k=num_samples)  # ✅ 有放回采样
    logger.info(f"准备从 {len(selected)} 个 chunk 中生成训练样本…")

    count = 0
    with open(output_path, 'a', encoding='utf-8') as f:
        for chunk in selected:
            prompt = generate_query_prompt(chunk['text'])
            response = llm.generate(prompt)
            try:
                line = response.strip().strip('()').replace('，', ',')
                query, is_precise, is_hybrid = [x.strip() for x in line.rsplit(',', 2)]
                f.write(json.dumps({
                    "query": query,
                    "precise": is_precise in ["是", "yes", "true", "True"],
                    "label": "hybrid" if is_hybrid in ["是", "yes", "true", "True"] else "norag"
                }, ensure_ascii=False) + "\n")
                count += 1
            except (ValueError, IndexError, json.JSONDecodeError) as e:
                logger.warning(f"解析生成响应失败: {response} -> {str(e)}")

    logger.info(f"训练数据生成完成，共生成 {count} 条记录，已追加写入 {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", action="store_true", help="创建新工作目录")
    parser.add_argument("--samples", type=int, help="生成样本数量（覆盖 config）")
    args = parser.parse_args()

    with open("config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if args.samples:
        config.setdefault("classifier", {})["generate_samples"] = args.samples

    work_dir = resolve_work_dir(config, args)
    logger = init_logger(work_dir)

    run_generate_training_data(config, work_dir, logger)
