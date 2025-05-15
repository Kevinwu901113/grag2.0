# model.py
import os
import torch
import joblib
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class BERTClassifier(nn.Module):
    def __init__(self, model_name, num_labels, cache_dir="./hf_models"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)

def load_model(model_dir, model_name="bert-base-chinese"):
    label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
    num_labels = len(label_encoder.classes_)

    model = BERTClassifier(model_name, num_labels)
    model.load_state_dict(torch.load(os.path.join(model_dir, "query_classifier.pt"), map_location="cpu"))
    model.eval()

    tokenizer = BertTokenizer.from_pretrained(model_name)

    return model, tokenizer, label_encoder

def save_model(model, label_encoder, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "query_classifier.pt"))
    joblib.dump(label_encoder, os.path.join(output_dir, "label_encoder.pkl"))

def predict(model, tokenizer, label_encoder, query, device="cpu"):
    encoding = tokenizer(
        query,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    model.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        pred_class_id = torch.argmax(logits, dim=1).item()
        label = label_encoder.inverse_transform([pred_class_id])[0]
    return label
