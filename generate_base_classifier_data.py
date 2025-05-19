import os
import json
import yaml
import argparse
import torch
from llm.llm import LLMClient
from utils.output_manager import resolve_work_dir
from utils.misc import init_logger
from sentence_transformers import SentenceTransformer, util

PROMPT_TEMPLATE = """你是一名专业的AI助手，负责为训练一个多领域查询分类器构造高质量的样本数据。

请确保每条查询都不重复，内容和语义也应尽可能多样化。

请严格按照以下标准生成 {num} 条查询数据：

数据需覆盖以下五个领域，每个领域均应提供多条样本：

法律：法律条文引用、责任判定、规定解释。

金融：贷款咨询、股票投资建议、保险条款解读、金融政策信息。

医疗：疾病诊断咨询、药品使用建议、医疗政策解读、就诊流程指引。

地方政策：区域政策说明、政府规定解读、社区服务介绍、惠民措施详情。

快递客服：物流信息查询、赔偿标准说明、客户投诉处理、包裹异常问题解决。

每条查询应展示多样化语言风格，包括口语表达、正式表述和书面用语，体现语言的丰富性。

对每个查询明确以下判断标准：

是否需要精确回答：涉及明确法律条文、具体日期、金额数量、责任主体或机构名称、具体地点地址等精确具体信息。

是否必须依赖原始文本块支持才能回答，或可以仅依靠结构化知识或常识作答。

请特别注意：如果问题被标记为“需要精确回答”，则它必须同时标记为“需要文本块支持”。

查询的输出格式严格遵循以下要求：
(问题文本, 是否需要精确回答, 是否需要文本块支持)

所有查询问题统一使用简体中文，每条单独一行，标点符号必须为中文格式。

以下为参考示例：
(根据《劳动法》第四十四条，法定假日加班工资标准是多少？, 是, 是)
(企业申请上市的主要条件有哪些？, 否, 否)
(感冒时服用阿莫西林合适吗？, 否, 否)
(深圳市公明区低保政策的申请条件和流程是什么？, 否, 是)
(快递三天未更新物流信息，应如何处理？, 否, 否)

请开始生成：
"""

def parse_response(text):
    lines = text.strip().splitlines()
    results = []
    for line in lines:
        line = line.strip().strip('()').replace('，', ',')
        if not line or line.count(',') < 2:
            continue
        try:
            q, p, l = [x.strip() for x in line.rsplit(',', 2)]
            need_precise = p in ["是", "true", "yes", "True"]
            need_doc = l in ["是", "true", "yes", "True"]

            if not need_doc and need_precise:
                continue

            if not need_doc:
                label = "norag"
            elif need_precise:
                label = "hybrid_precise"
            else:
                label = "hybrid_imprecise"

            results.append({
                "query": q,
                "label": label
            })
        except:
            continue
    return results

def generate_queries(config: dict, num: int, output_path: str, logger):
    llm = LLMClient(config)
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    batch_size = 50
    existing_queries = set()
    existing_list = []
    existing_emb = None

    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())
                    q = sample["query"]
                    existing_queries.add(q)
                    existing_list.append(q)
                except:
                    continue
        if existing_list:
            existing_emb = model.encode(existing_list, convert_to_tensor=True)
        logger.info(f"已检测到已有样本 {len(existing_queries)} 条，将跳过这些样本。")

    remaining = num - len(existing_queries)
    if remaining <= 0:
        logger.info(f"目标数量已达成（{num}条），不再生成。")
        return

    logger.info(f"开始增量生成剩余 {remaining} 条数据...")

    with open(output_path, 'a', encoding='utf-8') as f:
        generated = 0
        while generated < remaining:
            current_batch_size = min(batch_size, remaining - generated)
            prompt = PROMPT_TEMPLATE.format(num=current_batch_size)
            response = llm.generate(prompt)
            parsed = parse_response(response)

            for item in parsed:
                query = item["query"]
                if query in existing_queries:
                    continue

                query_emb = model.encode([query], convert_to_tensor=True)
                if existing_emb is not None:
                    similarity = util.cos_sim(query_emb, existing_emb)
                    if similarity.max().item() >= 0.9:
                        continue

                json.dump(item, f, ensure_ascii=False)
                f.write("\n")
                f.flush()
                existing_queries.add(query)
                existing_list.append(query)
                existing_emb = query_emb if existing_emb is None else torch.cat((existing_emb, query_emb), dim=0)
                generated += 1

            logger.info(f"当前已生成并写入总数：{generated}/{remaining}")

    logger.info(f"✅ 高质量分类器训练数据生成完成，总计生成并写入 {generated} 条，保存在 {output_path}")

def generate_samples(config, num, output_path, logger):
    generate_queries(config, num, output_path, logger)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=50, help="生成的总样本数量")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--output", type=str, default="base_classifier_samples.jsonl")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    work_dir = resolve_work_dir(config, args)
    logger = init_logger(work_dir)
    generate_queries(config, args.num, os.path.join(work_dir, args.output), logger)
