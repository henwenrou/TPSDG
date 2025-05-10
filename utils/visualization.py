# utils/visualization.py

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_nifti(image_path, label_path=None, slice_idx=None):
    """
    可视化单个nii.gz图像以及（可选的）对应标签。

    参数:
    --------
    image_path : str
        图像文件的路径 (.nii 或 .nii.gz)
    label_path : str, optional
        标签文件的路径 (.nii 或 .nii.gz)，如果不提供，则只显示图像
    slice_idx : int, optional
        指定想要显示的切片，如果为 None，则默认显示图像的中间切片

    使用举例:
    --------
        visualize_nifti(
            image_path='data/processed/case001_image.nii.gz',
            label_path='data/processed/case001_label.nii.gz',
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

    if label_path is not None:
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"标签文件不存在: {label_path}")

        # 加载标签数据
        label_nii = nib.load(label_path)
        label_data = label_nii.get_fdata()  # (H, W, D)，通常是 0/1 或若干整数标签
    else:
        label_data = None

    # 取指定 slice 的图像
    # 注: 这里假设图像是 (H, W, D)，第三维为切片
    image_slice = image_data[:, :, slice_idx]

    # 建立绘图窗口
    plt.figure(figsize=(12, 5))

    # 子图1：显示原始图像
    plt.subplot(1, 2, 1)
    plt.imshow(image_slice.T, cmap='gray', origin='lower')
    plt.title(f'Image (slice {slice_idx})')

    # 子图2：如果提供 label，则显示 label
    if label_data is not None:
        label_slice = label_data[:, :, slice_idx]
        plt.subplot(1, 2, 2)
        plt.imshow(image_slice.T, cmap='gray', origin='lower')
        # 为了让标签覆盖在图像上，可以指定 alpha 半透明度或使用不同的 cmap
        plt.imshow(label_slice.T, cmap='Reds', alpha=0.4, origin='lower')
        plt.title(f'Label (slice {slice_idx})')
    else:
        plt.subplot(1, 2, 2)
        plt.imshow(image_slice.T, cmap='gray', origin='lower')
        plt.title(f'Image (slice {slice_idx}) - no label')

    plt.tight_layout()
    plt.show()