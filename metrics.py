# metrics.py
import torch
from monai.metrics import compute_meandice
from losses.ph_loss import PHLoss

# 实例化 PH 损失，用作 TER 指标
ph_metric = PHLoss(reduction='mean', p=2, scaling=False)

@torch.no_grad()
def dice(pred, gt):
    """
    pred: 模型输出概率，shape (B, C, H, W)
    gt:   ground truth，shape (B, 1, H, W)，整数类标
    """
    # MONAI 的 compute_meandice 支持多类；去掉背景后平均
    return compute_meandice(pred, gt, include_background=False).mean().item()

@torch.no_grad()
def ter(pred, gt):
    """
    直接调用 PH 损失，不参与梯度，作为 TER 指标
    """
    return ph_metric(pred, gt).item()