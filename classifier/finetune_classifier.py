import os
import json
import argparse
import torch
import joblib
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW
from classifier.train_base_classifier import BERTClassifier

class QueryDataset(Dataset):
    def __init__(self, data_path, tokenizer, label_encoder, max_length=128):
        self.samples = []
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                self.samples.append((obj["query"], obj["label"]))
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        query, label = self.samples[idx]
        encoding = self.tokenizer(query, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.label_encoder.transform([label])[0], dtype=torch.long)
        }

def finetune_model(base_dir, data_path, output_path, model_name="bert-base-chinese", epochs=300, batch_size=16, lr=2e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = os.getenv('HF_CACHE_DIR', './hf_models')
    os.makedirs(cache_dir, exist_ok=True)
    tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    label_encoder = joblib.load(os.path.join(base_dir, "label_encoder.pkl"))

    dataset = QueryDataset(data_path, tokenizer, label_encoder)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = BERTClassifier(model_name, cache_dir=cache_dir, num_labels=len(label_encoder.classes_))
    model.load_state_dict(torch.load(os.path.join(base_dir, "query_classifier.pt")))
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} loss: {total_loss:.4f}")

    os.makedirs(output_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_path, "query_classifier.pt"))
    joblib.dump(label_encoder, os.path.join(output_path, "label_encoder.pkl"))
    print(f"✅ 微调完成，模型已保存至: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, required=True, help="base模型目录")
    parser.add_argument("--data", type=str, required=True, help="微调用的数据路径")
    parser.add_argument("--output", type=str, required=True, help="输出模型保存路径")
    parser.add_argument("--model", type=str, default="bert-base-chinese", help="模型名称或路径")
    args = parser.parse_args()

    finetune_model(args.base, args.data, args.output, args.model)
