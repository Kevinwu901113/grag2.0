
import os
import json
import yaml
import argparse
from tqdm import tqdm
from llm.llm import LLMClient

EVAL_PROMPT_TEMPLATE = """
你是一名专业的AI评估助手。以下是一个用户查询及其分类标签预测结果：

- 查询内容：{query}
- 模型预测标签：{label}
- 是否需要精确回答：{precise}

请根据以上内容判断：该标签是否合理？如果不合理，请指出合理标签，并简单说明原因。
请用以下 JSON 格式回答：
{{
  "agree": true/false,
  "suggestion": "合理标签（如 hybrid 或 norag）",
  "reason": "简要解释"
}}
"""

def evaluate_sample(sample, llm_client):
    query = sample["query"]
    label = sample["label"]
    precise = "是" if sample["precise"] else "否"
    prompt = EVAL_PROMPT_TEMPLATE.format(query=query, label=label, precise=precise)

    try:
        response = llm_client.generate(prompt)
        parsed = json.loads(response)
        sample["llm_agree"] = parsed.get("agree", None)
        sample["llm_suggestion"] = parsed.get("suggestion", None)
        sample["llm_reason"] = parsed.get("reason", None)
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        sample["llm_agree"] = None
        sample["llm_error"] = str(e)
        sample["llm_raw"] = response

    return sample

def process_file(input_path, output_path, config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    llm = LLMClient(config)

    results = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Evaluating"):
            try:
                sample = json.loads(line.strip())
                evaluated = evaluate_sample(sample, llm)
                results.append(evaluated)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print("跳过异常样本:", e)

    with open(output_path, "w", encoding="utf-8") as out:
        for item in results:
            json.dump(item, out, ensure_ascii=False)
            out.write("\n")

    print(f"✅ 评估完成，共保存 {len(results)} 条结果到 {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="输入样本文件路径 (.jsonl)")
    parser.add_argument("--output", type=str, required=True, help="评估结果保存路径 (.jsonl)")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    args = parser.parse_args()

    process_file(args.input, args.output, args.config)
