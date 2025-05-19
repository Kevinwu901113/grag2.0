# train_deep_classifier.py
import os
import json
import torch
import argparse
import joblib
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from classifier.model import BERTClassifier
from collections import Counter

class QueryDataset(Dataset):
    def __init__(self, queries, labels, tokenizer, max_length=128):
        self.queries = queries
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.queries[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

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

def get_or_download_tokenizer(model_name, cache_dir="./hf_models"):
    try:
        return BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    except Exception as e:
        print(f"⚠️ 加载 tokenizer 失败: {e}")
        print("⏬ 尝试重新下载 tokenizer...")
        return BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir, force_download=True)

def train_model(data_path, output_dir, model_name, num_epochs=30, batch_size=16, lr=2e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    queries, labels = load_data(data_path)
    print(f"共加载 {len(queries)} 条训练样本。")
    print("标签分布：", Counter(labels))

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_labels = len(label_encoder.classes_)

    tokenizer = get_or_download_tokenizer(model_name, cache_dir="./hf_models")
    dataset = QueryDataset(queries, encoded_labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = BERTClassifier(model_name, num_labels, cache_dir="./hf_models")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs} - Total Loss: {total_loss:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "query_classifier.pt"))
    joblib.dump(label_encoder, os.path.join(output_dir, "label_encoder.pkl"))
    print("✅ 模型训练完成，保存在", output_dir)
    print("类别标签映射：")
    for i, cls in enumerate(label_encoder.classes_):
        print(f"  [{i}] => {cls}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="base_classifier_samples.jsonl", help="输入的样本数据路径")
    parser.add_argument("--output", type=str, default="./result/query_classifier", help="模型保存路径")
    parser.add_argument("--model", type=str, default="bert-base-chinese", help="BERT模型名称")
    args = parser.parse_args()

    train_model(args.data, args.output, args.model)
