#!/usr/bin/env python3
"""
通用分类器训练脚本
整合训练相关的通用逻辑，支持基础训练和微调
"""

import os
import json
import argparse
import torch
from typing import Dict, Any, Optional, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

from utils.io import load_json, save_json, ensure_dir_exists
from utils.logger import setup_logger
from utils.common import validate_config


class ClassifierTrainer:
    """
    通用分类器训练器
    支持基础训练和微调模式
    """
    
    def __init__(self, config: Dict[str, Any], work_dir: str, logger=None):
        self.config = config
        self.work_dir = work_dir
        self.logger = logger or setup_logger(work_dir)
        
        # 验证配置
        required_keys = ['model_name', 'num_labels']
        if not validate_config(config, required_keys):
            raise ValueError("配置文件不完整")
        
        self.model_name = config['model_name']
        self.num_labels = config['num_labels']
        self.max_length = config.get('max_length', 512)
        self.batch_size = config.get('batch_size', 16)
        self.learning_rate = config.get('learning_rate', 2e-5)
        self.num_epochs = config.get('num_epochs', 3)
        
        # 初始化模型和分词器
        self.tokenizer = None
        self.model = None
        self.label_encoder = LabelEncoder()
        
    def load_pretrained_model(self, model_path: Optional[str] = None):
        """
        加载预训练模型
        
        Args:
            model_path: 预训练模型路径，如果为None则使用配置中的model_name
        """
        try:
            model_name_or_path = model_path or self.model_name
            self.logger.info(f"加载模型: {model_name_or_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path,
                num_labels=self.num_labels
            )
            
            # 如果是微调模式，加载标签编码器
            if model_path:
                label_encoder_path = os.path.join(model_path, "label_encoder.pkl")
                if os.path.exists(label_encoder_path):
                    with open(label_encoder_path, 'rb') as f:
                        self.label_encoder = pickle.load(f)
                    self.logger.info("已加载标签编码器")
                    
        except Exception as e:
            self.logger.error(f"加载模型失败: {e}")
            raise
    
    def prepare_data(self, data_path: str, is_finetune: bool = False):
        """
        准备训练数据
        
        Args:
            data_path: 数据文件路径
            is_finetune: 是否为微调模式
            
        Returns:
            (train_dataset, eval_dataset)
        """
        self.logger.info(f"加载训练数据: {data_path}")
        
        # 加载数据
        if data_path.endswith('.jsonl'):
            data = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
        else:
            data = load_json(data_path)
        
        if not data:
            raise ValueError("训练数据为空")
        
        # 提取文本和标签
        texts = [item['text'] if 'text' in item else item['query'] for item in data]
        labels = [item['label'] for item in data]
        
        # 编码标签
        if not is_finetune:
            # 基础训练模式：拟合标签编码器
            encoded_labels = self.label_encoder.fit_transform(labels)
        else:
            # 微调模式：使用已有的标签编码器
            encoded_labels = self.label_encoder.transform(labels)
        
        # 分割数据
        train_texts, eval_texts, train_labels, eval_labels = train_test_split(
            texts, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
        )
        
        # 创建数据集
        train_dataset = self._create_dataset(train_texts, train_labels)
        eval_dataset = self._create_dataset(eval_texts, eval_labels)
        
        self.logger.info(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(eval_dataset)}")
        return train_dataset, eval_dataset
    
    def _create_dataset(self, texts: List[str], labels: List[int]):
        """
        创建PyTorch数据集
        """
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        class Dataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels
            
            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
                return item
            
            def __len__(self):
                return len(self.labels)
        
        return Dataset(encodings, labels)
    
    def train(self, train_dataset, eval_dataset, output_dir: str):
        """
        训练模型
        
        Args:
            train_dataset: 训练数据集
            eval_dataset: 验证数据集
            output_dir: 输出目录
        """
        ensure_dir_exists(output_dir)
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=os.path.join(output_dir, 'logs'),
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        
        # 创建训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        # 开始训练
        self.logger.info("开始训练...")
        trainer.train()
        
        # 保存模型
        self.logger.info(f"保存模型到: {output_dir}")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # 保存标签编码器
        label_encoder_path = os.path.join(output_dir, "label_encoder.pkl")
        with open(label_encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        self.logger.info("训练完成")


def main():
    """
    主函数，支持命令行参数
    """
    parser = argparse.ArgumentParser(description="通用分类器训练脚本")
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--data", required=True, help="训练数据路径")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--pretrained", help="预训练模型路径（微调模式）")
    parser.add_argument("--work_dir", default="./", help="工作目录")
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_json(args.config)
    if not config:
        print("❌ 无法加载配置文件")
        return
    
    # 创建训练器
    trainer = ClassifierTrainer(config, args.work_dir)
    
    # 加载模型
    if args.pretrained:
        # 微调模式
        trainer.load_pretrained_model(args.pretrained)
        is_finetune = True
    else:
        # 基础训练模式
        trainer.load_pretrained_model()
        is_finetune = False
    
    # 准备数据
    train_dataset, eval_dataset = trainer.prepare_data(args.data, is_finetune)
    
    # 训练
    trainer.train(train_dataset, eval_dataset, args.output)


if __name__ == "__main__":
    main()