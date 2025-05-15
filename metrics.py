# metrics.py
import torch
import torch.nn.functional as F
from losses.ph_loss import PHLoss

_ph = PHLoss(reduction="mean")
EPS = 1e-6

def _to_one_hot(x: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    把整数标签 x (shape: B×D1×D2×…×Dn) 转成 one-hot (shape: B×C×D1×D2×…×Dn)
    """
    # one_hot: shape = (*x.shape, num_classes)
    oh = F.one_hot(x.long(), num_classes)
    # oh.dim() = x.dim() + 1
    # 我们要把最后一个 dim (num_classes) 换到 dim=1：
    dims = oh.dim()  # e.g. 4 for 2D, 5 for 3D
    # permute order: [0, last, 1, 2, ..., last-1]
    perm = [0, dims - 1] + list(range(1, dims - 1))
    return oh.permute(*perm).float()

@torch.no_grad()
def dice(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Manual Multi-class Dice (exclude background) averaged over batch & classes.
    """
    # 1) ensure 2-class splits become C=2
    if logits.shape[1] == 1:
        logits = torch.cat([1 - logits, logits], dim=1)
    # 2) squeeze away a singleton channel dim on labels
    if labels.ndim == logits.ndim:
        labels = labels.squeeze(1)

    B, C = logits.shape[0], logits.shape[1]
    # 3) get hard predictions per voxel
    preds = torch.argmax(logits, dim=1)             # (B, D1, D2, …)
    p1 = _to_one_hot(preds,  C)                     # (B, C, ...)
    t1 = _to_one_hot(labels, C)                     # (B, C, ...)

    # 4) flatten spatial dims
    p_flat = p1.reshape(B, C, -1)
    t_flat = t1.reshape(B, C, -1)

    # 5) compute intersection & union
    inter = (p_flat * t_flat).sum(-1)               # (B, C)
    union = p_flat.sum(-1) + t_flat.sum(-1)         # (B, C)
    dice_score = (2 * inter + EPS) / (union + EPS)  # (B, C)

    # 6) drop background class 0, then mean over B×(C-1)
    if C > 1:
        dice_score = dice_score[:, 1:]
    return float(dice_score.mean())

@torch.no_grad()
def ter(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float(_ph(logits, labels))