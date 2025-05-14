# metrics.py
import torch
from monai.metrics import compute_meandice
from losses.ph_loss import PHLoss
import torch.nn.functional as F

# 实例化 PH 损失，用作 TER 指标
ph_metric = PHLoss()

@torch.no_grad()
def dice(pred: torch.Tensor, gt: torch.Tensor):
    """
    既接受 (B,1,H,W) 也接受 (B,H,W) / (B,C,H,W) 的标签
    自动做 one-hot / unsqueeze，保证 pred & gt 最终同形状
    """
    if gt.ndim == pred.ndim - 1:           # (B,H,W) → (B,1,H,W)
        gt = gt.unsqueeze(1)

    if gt.shape[1] == 1 and pred.shape[1] > 1:
        # pred 多通道 + gt 单通道: 把 gt one-hot 到同通道数
        num_c = pred.shape[1]
        gt = F.one_hot(gt.squeeze(1).long(), num_c).permute(0, 3, 1, 2)

    assert pred.shape == gt.shape, f"{pred.shape=} {gt.shape=}"
    # 先拿到每个通道的 Dice
    scores = compute_meandice(pred, gt, include_background=False)
    # 只保留非 nan 的通道
    # 只保留非 nan 的通道
    valid = ~torch.isnan(scores)
    if valid.sum() == 0:
        # 如果一张图所有前景类别都没出现，直接当作 0
        return 0.0
    # 按有效通道做平均
    return scores[valid].mean().item()

@torch.no_grad()
def ter(pred, gt):
    """
    直接调用 PH 损失，不参与梯度，作为 TER 指标
    """
    
    return ph_metric(pred, gt).item()