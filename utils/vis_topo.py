#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick topology overlay visualizer
python tools/vis_topo.py \
  --cfg configs/sc.yaml \
  --ckpt_base logs/baseline/checkpoints/best.pth \
  --ckpt_topo logs/topo/checkpoints/best.pth \
  --n 8 --out vis_topo
"""
import os, sys
# Ensure TPSDG root is in Python path
ROOT = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
if ROOT not in sys.path:
    sys.path.append(ROOT)

import argparse, os, random, torch, numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, find_contours
from torchvision.utils import make_grid

# ---------- 解析参数 ----------
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, required=True)
parser.add_argument('--ckpt_base', type=str, required=True)
parser.add_argument('--ckpt_topo', type=str, required=True)
parser.add_argument('--n', type=int, default=8)
parser.add_argument('--out', type=str, default='vis_topo')
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------- 数据 & 模型 ----------
from omegaconf import OmegaConf
from main import instantiate_from_config
from torch.utils.data import DataLoader

# Load configuration
config = OmegaConf.load(args.cfg)

# Build data module
data_module = instantiate_from_config(config.data)
data_module.setup()              # 初始化 train / val / test_all
print("DataModule setup complete: train, val, test_all loaded")
val_loader = DataLoader(data_module.datasets["validation"], batch_size=1, num_workers=1, shuffle=False)

# Build two models (baseline and topo) from the same model config
model = instantiate_from_config(config.model).to(device)
model_base = model
model_topo = instantiate_from_config(config.model).to(device)

 # ---------- 加载权重（支持多层 unwrap） ----------
def unwrap_state_dict(ckpt):
    """
    自动从 {'model': {...}} 或 {'state_dict': {...}} 或 嵌套多层中提取最内层实际权重 dict
    """
    state = ckpt
    # if checkpoint is a dict, try unwrapping common keys
    if isinstance(state, dict):
        for key in ('state_dict', 'model'):
            if key in state and isinstance(state[key], dict):
                state = state[key]
    # handle repeated nesting: e.g., {'model': {'model': {...}}}
    while isinstance(state, dict) and list(state.keys()) == ['model']:
        state = state['model']
    return state

# Load base checkpoint
ckpt_b = torch.load(args.ckpt_base, map_location=device)
state_b = unwrap_state_dict(ckpt_b)
model_base.load_state_dict(state_b)

# Load topo checkpoint
ckpt_t = torch.load(args.ckpt_topo, map_location=device)
state_t = unwrap_state_dict(ckpt_t)
model_topo.load_state_dict(state_t)
model_base.eval()
model_topo.eval()

# ---------- 辅助函数 ----------
cmap = plt.get_cmap('tab20')

def overlay_cc(image, mask_a, mask_b):
    """返回三通道 overlay (H,W,3)"""
    h, w = image.shape
    overlay = np.stack([image]*3, -1)  # 灰底
    # A: baseline-only  B: topo-only
    diff_a = np.logical_and(mask_a, np.logical_not(mask_b))
    diff_b = np.logical_and(mask_b, np.logical_not(mask_a))
    # 填充闭合区域内部
    overlay[diff_a, 0] = 1.  # 红色通道填充 topo-only 区域
    overlay[diff_b, 1] = 1.  # 绿色通道填充 baseline-only 区域
    for idx, msk in enumerate([diff_a, diff_b]):
        lbs = label(msk)
        for cc in np.unique(lbs)[1:]:
            yx = lbs == cc
            cnt = find_contours(yx.astype(float), .5)
            for c in cnt:
                c = np.flip(c,1).astype(int)
                overlay[c[:,1], c[:,0], idx] = 1.      # R or G 通道
    return overlay.clip(0,1)

# ---------- 主循环 ----------
chosen = random.sample(range(len(val_loader.dataset)), args.n)
for i, idx in enumerate(chosen):
    sample = val_loader.dataset[idx]
    # Handle datasets that return dicts vs. tuples
    if isinstance(sample, dict):
        # Extract image tensor (support dict of modalities or single tensor)
        images_field = sample.get('images', sample.get('aug_images', None))
        if images_field is None:
            raise KeyError(
                f"Cannot locate image key in sample. "
                f"Available keys: {list(sample.keys())}. "
                f"Expected one of 'images','aug_images'."
            )
        # If multiple modalities, take the first modality
        if isinstance(images_field, dict):
            img = list(images_field.values())[0]
        else:
            img = images_field

        # Extract label tensor
        labels_field = sample.get('labels', sample.get('label', sample.get('lb', None)))
        if labels_field is None:
            raise KeyError(
                f"Cannot locate label key in sample. "
                f"Available keys: {list(sample.keys())}. "
                f"Expected one of 'labels','label','lb'."
            )
        if isinstance(labels_field, dict):
            gt = list(labels_field.values())[0]
        else:
            gt = labels_field
    else:
        # Dataset returned a tuple of (image, label)
        img, gt = sample
    with torch.no_grad():
        inp = img.unsqueeze(0).to(device)
        pb = torch.sigmoid(model_base(inp)).cpu()[0,0].numpy()>0.5
        pt = torch.sigmoid(model_topo(inp)).cpu()[0,0].numpy()>0.5
    # Normalize the first channel of img to [0,1] using numpy
    img_arr = img.detach().cpu().numpy()  # shape: (C, H, W)
    img_np = (img_arr[0] - img_arr[0].min()) / (img_arr[0].max() - img_arr[0].min())

    fig, ax = plt.subplots(2,2, figsize=(6,6))
    ax[0,0].imshow(img_np, cmap='gray');       ax[0,0].set_title('Image')
    ax[0,1].imshow(pb,   cmap='gray');         ax[0,1].set_title('Baseline')
    ax[1,0].imshow(pt,   cmap='gray');         ax[1,0].set_title('TopoPred')
    ax[1,1].imshow(overlay_cc(img_np, pb, pt));ax[1,1].set_title('Δ Connected Components')


    for a in ax.ravel(): a.axis('off')
    plt.tight_layout()
    plt.savefig(f"{args.out}/vis_{i:02d}.png", dpi=300)
    plt.close()
    # ---------- 单独叠加图: Baseline(Green) vs Topo(Red) ----------
    fig2, ax2 = plt.subplots(1, 1, figsize=(6,6))
    ax2.imshow(img_np, cmap='gray', origin='lower')
    # 半透明覆盖 baseline
    ax2.imshow(np.ma.masked_where(~pb, pb), cmap='Greens', alpha=0.8, origin='lower')
    # 半透明覆盖 topo
    ax2.imshow(np.ma.masked_where(~pt, pt), cmap='Reds', alpha=0.8, origin='lower')
    ax2.set_title('Overlay: Baseline (Green) vs Topo (Red)')
    ax2.axis('off')
    plt.tight_layout()
    plt.close(fig2)
print(f"Saved {args.n} figures to {args.out}/")