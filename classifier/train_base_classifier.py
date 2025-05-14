
import os
import json
import torch
import argparse
import joblib
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel

class HFClassifierTrainer:
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

def train_and_save_model(data_path, output_path, model_path, model_name="bert-base-chinese"):
    print("🚀 加载数据...")
    queries, labels = load_data(data_path)
    trainer = HFClassifierTrainer(model_name_or_path=model_name)

    print("🧠 开始编码...")
    X = trainer.encode_batch(queries)

    print("🔖 标签编码...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    print("🎯 模型训练...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)

    print("💾 保存模型...")
    os.makedirs(output_path, exist_ok=True)
    joblib.dump(clf, os.path.join(output_path, "query_classifier.pkl"))
    joblib.dump(label_encoder, os.path.join(output_path, "label_encoder.pkl"))

    print("✅ 模型训练完成，保存在", output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="base_classifier_samples.jsonl", help="输入的样本数据路径")
    parser.add_argument("--output", type=str, default="./result/query_classifier", help="模型保存路径")
    parser.add_argument("--model", type=str, default="bert-base-chinese", help="HuggingFace模型名称或路径")
    args = parser.parse_args()

    train_and_save_model(args.data, args.output, args.output, args.model)
