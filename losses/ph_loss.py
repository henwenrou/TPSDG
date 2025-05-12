# TPSDG/losses/ph_loss.py
"""
Topology-aware loss compatible with torchph==0.1.1
-------------------------------------------------
* 支持二值 / 多类分割
* 默认用 W₂ 距离度量同调差异（可自行换 p=1 / p=∞）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# ----- 试优先用 torchph（真编好了扩展）-----
try:
    from torchph.cubical import cubical_persistence
except (ImportError, ModuleNotFoundError):
    # ----- 否则回退到 GUDHI -----
    from gudhi.sklearn.cubical_persistence import CubicalPersistence
    def cubical_persistence(img, dims=(0,)):
        """
        img : torch.Tensor (1,H,W) or (H,W)
        返回 list[np.ndarray]  ⟨按 dims 顺序排列的条形码⟩
        """
        import numpy as np
        cp = CubicalPersistence(homology_dimensions=list(dims))
        dgms = cp.fit_transform(img.squeeze().cpu().numpy()[None])[0]   # ↩︎ dgms 可能是 list **或** dict
    
        # -------- 兼容两种返回类型 --------
        if isinstance(dgms, dict):              # 官方 sklearn 包装：dict{dim:array}
            pull = lambda d: dgms.get(d, np.zeros((0, 2), np.float32))
        else:                                   # 早期实现：list  [array_dim0, array_dim1, ...]
            pull = lambda d: dgms[d] if d < len(dgms) else np.zeros((0, 2), np.float32)
    
        return [np.asarray(pull(dim), dtype=np.float32) for dim in dims]
        
# --- Wasserstein 距离 --------------------------------------------------------
try:                                # 如果将来你真装上带 utils 的 torchph
    from torchph.utils.wasserstein import wasserstein_distance
except ImportError:                 # 否则回退到 GUDHI 实现
    from gudhi.wasserstein import wasserstein_distance as _gudhi_wd

    # 包一层，以保持原来 p 参数的调用方式
    def wasserstein_distance(diag_p, diag_g, p=2):
        # GUDHI 的签名是 order=、internal_p=
        return _gudhi_wd(diag_p, diag_g, order=p, internal_p=2.0)

class PHLoss(nn.Module):
    """
    Args
    ----
    dims        : tuple(int)  需要惩罚的同调维，如 (0,1) 表示连通分量 + 空洞
    threshold   : float       若 pred 是概率，可以二值化做硬过滤；设为 None 保持连续
    reduction   : "mean"|"sum"
    """

    def __init__(self, dims=(0,), threshold=0.5, reduction="mean"):
        super().__init__()
        self.dims = dims
        self.th = threshold
        self.reduction = reduction

    # ------------------------------------------------------------------ utils
    @torch.no_grad()
    def _diagram(self, img: torch.Tensor):
        """
        img : (H,W) float32, 已经在 [0,1] or {0,1}
        返回 list[diag_dim_i]，每个 diag 形如 (N_i, 2)
        """
        # cubical 按“越小越早”原则；前景像素设为 0 可以让其先出生
        filtration = 1.0 - img
        # torchph 要求 batch 维；(1,H,W) 即可
        return cubical_persistence(filtration.unsqueeze(0), dims=self.dims)

    # ---------------------------------------------------------------- forward
    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        pred   : (B,C,H,W) logits **或** prob
        target : (B,C,H,W) one-hot 0/1  (若传入 (B,1,H,W) 则视为二值)
        """
        # ------ 1) 概率化 ------
        if pred.dtype.is_floating_point and pred.max() > 1.5:   # 粗糙判定 logits
            prob = torch.sigmoid(pred) if pred.shape[1] == 1 else F.softmax(pred, dim=1)
        else:
            prob = pred                                           # 已是概率

        # ------ 2) one-hot 校正 ------
        if target.shape[1] != prob.shape[1]:                      # 例如 (B,1,H,W)
            target = F.one_hot(target.long().squeeze(1),
                               num_classes=prob.shape[1]).permute(0, 3, 1, 2).float()

        B, C, *_ = prob.shape
        losses = []

        # ------ 3) 逐通道/逐样本比较拓扑 ------
        for b in range(B):
            for c in range(C):
                # 可选硬阈值化：使图像更像 0/1
                p_img = (prob[b, c] > self.th).float() if self.th is not None else prob[b, c]
                g_img = target[b, c]

                dg_pred = self._diagram(p_img)
                dg_true = self._diagram(g_img)

                # 同调维循环求 W₂
                l = 0.0
                for idx, _ in enumerate(self.dims):
                    if dg_pred[idx].size == 0 and dg_true[idx].size == 0:
                        continue
                    l += wasserstein_distance(dg_pred[idx], dg_true[idx], p=2)
                
                # 转成tensor 放在与logits同设备
                if np.isnan(l) or np.isinf(l):
                    l = 0.0
                losses.append(torch.tensor(l, device=pred.device, dtype=pred.dtype))


        out = torch.stack(losses)
        return out.mean() if self.reduction == "mean" else out.sum()