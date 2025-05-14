import os
import json
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List

model_filename = "query_classifier.pkl"

def run_query_classifier(config: dict, work_dir: str, logger):
    model_path = os.path.join(work_dir, model_filename)
    buffer_path = os.path.join(work_dir, "continual_buffer.jsonl")
    base_model_path = config.get("classifier", {}).get("base_model_path")

    if os.path.exists(model_path):
        logger.info(f"加载已有查询分类器模型: {model_path}")
        return

    if not os.path.exists(buffer_path):
        logger.warning("未找到持续学习缓存文件，无法训练初始模型。")
        return

    queries, labels, precise_flags = [], [], []
    with open(buffer_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            queries.append(data['query'])
            labels.append(data['label'])  # 'hybrid' or 'norag'
            precise_flags.append(1 if data.get('precise', False) else 0)

    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(queries)
    y_mode = np.array([0 if l == 'norag' else 1 for l in labels])
    y_precise = np.array(precise_flags)

    # 判断是否存在 base model 初始化
    clf_mode = LogisticRegression()
    clf_precise = LogisticRegression()
    base_model_full_path = os.path.join(work_dir, base_model_path) if base_model_path else None
    if base_model_full_path and os.path.exists(base_model_full_path):
        try:
            clf_mode_pre, clf_precise_pre, vec_pre = joblib.load(base_model_full_path)
            logger.info(f"加载通用查询分类器作为初始模型: {base_model_full_path}")
            clf_mode = clf_mode_pre
            clf_precise = clf_precise_pre
        except Exception as e:
            logger.warning(f"加载 base model 失败，使用新模型训练。错误: {e}")

    clf_mode.fit(X, y_mode)
    clf_precise.fit(X, y_precise)

    joblib.dump((clf_mode, clf_precise, vectorizer), model_path)
    logger.info(f"查询分类器（模式+精确度）训练完成，保存至 {model_path}")

def classify_query(query: str, model_path: str) -> str:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型文件: {model_path}")

    clf_mode, _, vectorizer = joblib.load(model_path)
    X = vectorizer.transform([query])
    pred = clf_mode.predict(X)[0]
    return "norag" if pred == 0 else "hybrid"

def predict_precise_need(query: str, model_path: str) -> bool:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型文件: {model_path}")

    _, clf_precise, vectorizer = joblib.load(model_path)
    X = vectorizer.transform([query])
    pred = clf_precise.predict(X)[0]
    return bool(pred)
