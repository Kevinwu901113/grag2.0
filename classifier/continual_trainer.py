
import os
import json
import torch
import argparse
import joblib
from tqdm import tqdm
from sklearn.linear_model import SGDClassifier
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

def continual_update(buffer_path, base_model_dir, output_path, model_name="bert-base-chinese"):
    print("📥 加载持续学习数据...")
    queries, labels = load_data(buffer_path)
    encoder = HFEncoder(model_name_or_path=model_name)

    print("🔍 加载旧模型与标签器...")
    clf_path = os.path.join(base_model_dir, "query_classifier.pkl")
    label_path = os.path.join(base_model_dir, "label_encoder.pkl")

    if not os.path.exists(clf_path) or not os.path.exists(label_path):
        print("❌ 找不到基础模型，请先训练 base model")
        return

    clf = joblib.load(clf_path)
    label_encoder = joblib.load(label_path)

    print("🧠 编码增量样本...")
    X_new = encoder.encode_batch(queries)
    y_new = label_encoder.transform(labels)

    print("📈 执行 partial_fit 增量更新...")
    if not hasattr(clf, "partial_fit"):
        print("⚠️ 当前分类器不支持 partial_fit，将使用全量 fit 替代")
        clf.fit(X_new, y_new)
    else:
        classes = list(label_encoder.transform(label_encoder.classes_))
        clf.partial_fit(X_new, y_new, classes=classes)

    print("💾 保存增量更新后的模型...")
    os.makedirs(output_path, exist_ok=True)
    joblib.dump(clf, os.path.join(output_path, "query_classifier.pkl"))
    joblib.dump(label_encoder, os.path.join(output_path, "label_encoder.pkl"))

    print("✅ 持续学习完成，模型已保存至：", output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer", type=str, required=True, help="持续学习用的数据文件")
    parser.add_argument("--base_model", type=str, required=True, help="base模型目录")
    parser.add_argument("--output", type=str, required=True, help="保存模型输出目录")
    parser.add_argument("--model", type=str, default="bert-base-chinese", help="HuggingFace模型路径")
    args = parser.parse_args()

    continual_update(args.buffer, args.base_model, args.output, args.model)
