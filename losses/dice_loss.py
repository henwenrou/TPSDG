# losses/dice_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, C, H, W]，网络输出的原始分数（logits）
            targets: [B, H, W]，类别索引（值在 0 到 C-1 之间）
        返回:
            dice loss 标量
        """
        num_classes = logits.shape[1]
        # 将 targets 转换为 one-hot 格式，形状 [B, C, H, W]
        targets_onehot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        # 使用 softmax 得到每个类别的概率
        probs = F.softmax(logits, dim=1)
        # 计算交集和并集
        intersection = (probs * targets_onehot).sum(dim=(2, 3))
        cardinality = (probs + targets_onehot).sum(dim=(2, 3))
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        # dice loss = 1 - 平均 dice_score
        dice_loss = 1 - dice_score.mean()
        return dice_loss

if __name__ == "__main__":
    # 测试代码
    B, C, H, W = 2, 5, 128, 128
    logits = torch.randn(B, C, H, W)
    # 随机生成类别索引作为 ground truth，范围 0 ~ 4
    targets = torch.randint(0, C, (B, H, W))
    loss = DiceLoss()(logits, targets)
    print("Dice Loss:", loss.item())