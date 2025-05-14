
import os
import json
import torch
import argparse
import joblib
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertModel

class HFEncoder:
    def __init__(self, model_name_or_path="bert-base-chinese", cache_dir="./hf_models"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        self.encoder = BertModel.from_pretrained(model_name_or_path, cache_dir=cache_dir).to(self.device)
        self.encoder.eval()

    def encode_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding.squeeze(0).cpu().numpy()

    def encode_batch(self, texts):
        return [self.encode_text(text) for text in tqdm(texts, desc="Embedding")]

def load_data(jsonl_path):
    queries, labels = [], []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                sample = json.loads(line.strip())
                queries.append(sample["query"])
                labels.append(sample["label"])
            except:
                continue
    return queries, labels

def finetune_model(data_path, base_model_dir, output_path, model_name="bert-base-chinese"):
    print("🚀 加载样本数据...")
    queries, labels = load_data(data_path)
    encoder = HFEncoder(model_name_or_path=model_name)

    print("🧠 生成嵌入向量...")
    X = encoder.encode_batch(queries)

    print("📦 加载原模型和标签编码器...")
    clf = joblib.load(os.path.join(base_model_dir, "query_classifier.pkl"))
    label_encoder = joblib.load(os.path.join(base_model_dir, "label_encoder.pkl"))

    print("📊 编码标签...")
    y = label_encoder.transform(labels)

    print("🔁 重新训练模型（拟微调）...")
    clf.fit(X, y)

    os.makedirs(output_path, exist_ok=True)
    joblib.dump(clf, os.path.join(output_path, "query_classifier_finetuned.pkl"))
    joblib.dump(label_encoder, os.path.join(output_path, "label_encoder.pkl"))

    print("✅ 微调完成，保存至：", output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="微调使用的样本文件")
    parser.add_argument("--base_model", type=str, required=True, help="base模型目录")
    parser.add_argument("--output", type=str, required=True, help="保存微调后模型的目录")
    parser.add_argument("--model", type=str, default="bert-base-chinese", help="HuggingFace模型路径")
    args = parser.parse_args()

    finetune_model(args.data, args.base_model, args.output, args.model)
