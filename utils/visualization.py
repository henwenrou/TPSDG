# utils/visualization.py

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_nifti(image_path, baseline_path=None, pred_path=None, slice_idx=None):
    """
    可视化单个nii.gz图像以及（可选的）对应标签。

    参数:
    --------
    image_path : str
        图像文件的路径 (.nii 或 .nii.gz)
    baseline_path : str, optional
        基线（GT）标签文件的路径 (.nii 或 .nii.gz)，如果不提供，则只显示图像
    pred_path : str, optional
        预测标签文件的路径 (.nii 或 .nii.gz)，如果不提供，则只显示基线
    slice_idx : int, optional
        指定想要显示的切片，如果为 None，则默认显示图像的中间切片

    使用举例:
    --------
        visualize_nifti(
            image_path='data/processed/case001_image.nii.gz',
            baseline_path='data/processed/case001_label.nii.gz',
            pred_path='data/processed/case001_pred.nii.gz',
            slice_idx=60
        )
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")

    # 加载图像数据
    image_nii = nib.load(image_path)
    image_data = image_nii.get_fdata()  # 转为浮点数的 numpy 数组 (H, W, D)

    # 如果没有指定 slice_idx，就默认选取中间切片
    if slice_idx is None:
        slice_idx = image_data.shape[2] // 2  # 取最后一维的中间位置

    if baseline_path is not None:
        if not os.path.exists(baseline_path):
            raise FileNotFoundError(f"基线文件不存在: {baseline_path}")

        # 加载基线标签数据
        baseline_nii = nib.load(baseline_path)
        baseline_data = baseline_nii.get_fdata()  # (H, W, D)，通常是 0/1 或若干整数标签
    else:
        baseline_data = None

    if pred_path is not None:
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"预测文件不存在: {pred_path}")
        pred_nii = nib.load(pred_path)
        pred_data = pred_nii.get_fdata()
    else:
        pred_data = None

    # 取指定 slice 的图像
    # 注: 这里假设图像是 (H, W, D)，第三维为切片
    image_slice = image_data[:, :, slice_idx]

    # 建立绘图窗口
    plt.figure(figsize=(12, 5))

    # 子图1：显示原始图像
    plt.subplot(1, 2, 1)
    plt.imshow(image_slice.T, cmap='gray', origin='lower')
    plt.title(f'Image (slice {slice_idx})')

    # 子图2：显示原始图像和分割对比
    plt.subplot(1, 2, 2)
    plt.imshow(image_slice.T, cmap='gray', origin='lower')
    if baseline_data is not None:
        baseline_slice = (baseline_data[:, :, slice_idx] > 0)
        # 填充基线全部区域（浅红色）
        plt.imshow(np.ma.masked_where(~baseline_slice, baseline_slice).T, cmap='Reds', alpha=0.3, origin='lower')
        # 绘制基线（GT）轮廓，颜色加深
        plt.contour(baseline_slice.T, levels=[0.5], colors='blue', linewidths=2, origin='lower')
        if pred_data is not None:
            pred_slice = (pred_data[:, :, slice_idx] > 0)
            # 填充预测全部区域（浅绿色）
            plt.imshow(np.ma.masked_where(~pred_slice, pred_slice).T, cmap='Greens', alpha=0.3, origin='lower')
            # 绘制预测轮廓
            plt.contour(pred_slice.T, levels=[0.5], colors='green', linewidths=2, origin='lower')
            # 计算差异
            only_baseline = baseline_slice & (~pred_slice)
            only_pred = pred_slice & (~baseline_slice)
            # 标注仅GT区域（红）
            plt.imshow(np.ma.masked_where(~only_baseline, only_baseline).T, cmap='Reds', alpha=0.6, origin='lower')
            # 标注仅预测区域（绿）
            plt.imshow(np.ma.masked_where(~only_pred, only_pred).T, cmap='Greens', alpha=0.6, origin='lower')
    plt.title(f'Comparison (slice {slice_idx})')

    plt.tight_layout()
    plt.show()

# visualize_nifti(
#     image_path='/Users/RexRyder/PycharmProjects/D2SDG/data/processed/abdominal/SABSCT/processed/image_28.nii.gz',
#     baseline_path='/Users/RexRyder/PycharmProjects/D2SDG/data/processed/abdominal/SABSCT/processed/label_28.nii.gz',
#     pred_path=None,
#     slice_idx=60
# )