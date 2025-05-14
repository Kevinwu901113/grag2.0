
from grag.classifier.base_trainer import train_and_save_model
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="训练样本路径")
    parser.add_argument("--output", type=str, required=True, help="输出模型目录")
    parser.add_argument("--model", type=str, default="bert-base-chinese", help="HuggingFace模型名或本地路径")
    args = parser.parse_args()

    train_and_save_model(args.data, args.output, args.output, args.model)
