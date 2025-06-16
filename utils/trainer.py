# utils/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim


def get_default_loss(loss_type="cross_entropy"):
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif loss_type == "mse":
        return nn.MSELoss()
    else:
        raise ValueError(f"不支持的 loss 类型: {loss_type}")


def get_optimizer(model, optimizer_type="adam", lr=1e-3):
    if optimizer_type == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == "sgd":
        return optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")
