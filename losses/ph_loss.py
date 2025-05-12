# TPSDG/losses/ph_loss.py
import torch, torch.nn as nn
from torchph.core import cubical        # ⬅️ GPU cubical persistence
from torchph.utils.wasserstein import wasserstein_distance

class PHLoss(nn.Module):
    """
    Persistent-Homology loss for binary/多类分割
    pred : (B,C,H,W)  —— logits or probs
    target : (B,C,H,W) —— one-hot tensor 0/1
    dims : tuple of homology dimensions to penalise, e.g. (0,1) for 连通性+空洞
    """
    def __init__(self, dims=(0, ), thresh=0.5, reduction="mean"):
        super().__init__()
        self.dims, self.th, self.reduction = dims, thresh, reduction

    @torch.no_grad()
    def _diagram(self, mask, dims):
        """mask: (H,W) float32  ∈[0,1]"""
        # cubical persistence expects **smaller value = earlier birth**
        filt = 1. - mask      # invert s.t. object pixels=0
        dgms = cubical.cubical_persistence(filt, dims=dims)  # list of diagrams
        return dgms

    def forward(self, pred, target):
        prob = torch.sigmoid(pred) if pred.dtype == torch.float32 else pred
        B, C, _, _ = prob.shape
        losses = []

        for b in range(B):
            for c in range(C):
                p, g = prob[b, c], target[b, c].float()
                d_pred = self._diagram(p, self.dims)
                d_gt   = self._diagram(g, self.dims)

                loss_c = 0.
                # 按每个维度计算 W₂ 距离
                for idx, dim in enumerate(self.dims):
                    loss_c += wasserstein_distance(d_pred[idx], d_gt[idx], p=2)
                losses.append(loss_c)

        loss = torch.stack(losses)
        return loss.mean() if self.reduction == "mean" else loss.sum()