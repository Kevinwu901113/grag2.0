import argparse
import os
import json
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(jsonl_path):
    queries, labels, precise_flags = [], [], []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            queries.append(data['query'])
            labels.append(data['label'])  # 'hybrid' or 'norag'
            precise_flags.append(1 if data.get('precise', False) else 0)
    return queries, labels, precise_flags

def train_classifier(queries, labels, precise_flags, output_path):
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(queries)
    y_mode = np.array([0 if l == 'norag' else 1 for l in labels])
    y_precise = np.array(precise_flags)

    clf_mode = LogisticRegression()
    clf_mode.fit(X, y_mode)

    clf_precise = LogisticRegression()
    clf_precise.fit(X, y_precise)

    joblib.dump((clf_mode, clf_precise, vectorizer), output_path)
    print(f"✅ 通用查询分类器训练完成，已保存至: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="训练数据的 .jsonl 文件路径")
    parser.add_argument("--output", type=str, default="base_classifier.pkl", help="输出模型文件路径")
    args = parser.parse_args()

    queries, labels, precise_flags = load_data(args.data)
    train_classifier(queries, labels, precise_flags, args.output)

if __name__ == "__main__":
    main()



#python prepare_base_classifier_data.py --data data/base.jsonl --output result/base_classifier.pkl
