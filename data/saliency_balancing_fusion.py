import torch
import numpy as np
import torch.nn.functional as F


def bspline_kernel_2d(sigma=[1, 1], order=2, asTensor=False, dtype=torch.float32, device='cuda'):
    '''
    生成 2D B样条（B-spline）核矩阵。
    参考：https://en.wikipedia.org/wiki/B-spline
    B样条插值的快速计算可以通过迭代均值滤波实现。

    :param sigma: 控制平滑度的整数元组
    :param order: 插值的阶数
    :param asTensor: 是否返回 PyTorch 张量
    :param dtype: 数据类型
    :param device: 设备（默认为 'cuda'）
    :return: 2D B样条核矩阵（numpy 数组或 PyTorch 张量）
    '''
    # 生成全 1 核矩阵，尺寸由 sigma 控制
    kernel_ones = torch.ones(1, 1, *sigma)
    kernel = kernel_ones
    padding = np.array(sigma)

    # 通过卷积递归地生成 B 样条核
    for i in range(1, order + 1):
        kernel = F.conv2d(kernel, kernel_ones, padding=(i * padding).tolist()) / (sigma[0] * sigma[1])

    # 根据 asTensor 选择返回数据格式
    if asTensor:
        return kernel.to(dtype=dtype, device=device)
    else:
        return kernel.numpy()


def get_bspline_kernel(spacing=[32, 32], order=3):
    '''
    计算 B 样条核和相应的 padding 值

    :param spacing: 控制点间距，定义核的尺寸
    :param order: B 样条插值的阶数，默认 3
    :return: 生成的 B 样条核（PyTorch 张量）及其 padding
    '''
    _kernel = bspline_kernel_2d(spacing, order=order, asTensor=True)  # 生成 B 样条核
    _padding = (np.array(_kernel.size()[2:]) - 1) / 2  # 计算 padding 以保证正确对齐
    _padding = _padding.astype(dtype=int).tolist()  # 转换为整数列表
    return _kernel, _padding


def rescale_intensity(data, new_min=0, new_max=1, group=4, eps=1e-20):
    '''
    归一化 PyTorch 批量数据的强度，使其范围限定在 [new_min, new_max] 之间。

    :param data: 形状为 (N,1,H,W) 的张量，表示批量数据
    :param new_min: 归一化后的最小值
    :param new_max: 归一化后的最大值
    :param group: 未使用参数（可能用于分组计算）
    :param eps: 避免除零错误的极小值
    :return: 归一化后的数据
    '''
    bs, c, h, w = data.size(0), data.size(1), data.size(2), data.size(3)
    data = data.view(bs * c, -1)  # 将数据展平，每个通道视为一个独立数组

    # 计算数据的最大值和最小值
    old_max = torch.max(data, dim=1, keepdim=True).values
    old_min = torch.min(data, dim=1, keepdim=True).values

    # 归一化数据到 [new_min, new_max] 范围
    new_data = (data - old_min + eps) / (old_max - old_min + eps) * (new_max - new_min) + new_min
    new_data = new_data.view(bs, c, h, w)  # 重新调整回原始形状
    return new_data


def get_SBF_map(gradient, grid_size):
    '''
    计算平滑的 B 样条特征（SBF）映射，用于梯度数据的平滑处理。

    :param gradient: 输入梯度图，形状为 (B, C, H, W)
    :param grid_size: 网格尺寸，用于控制下采样尺度
    :return: 归一化的平滑显著性图
    '''
    b, c, h, w = gradient.size()

    # 获取 B 样条核及其 padding
    bs_kernel, bs_pad = get_bspline_kernel(spacing=[h // grid_size, h // grid_size], order=2)

    # 计算梯度的低分辨率显著性图
    saliency = F.adaptive_avg_pool2d(gradient, grid_size)

    # 使用 B 样条核进行转置卷积（上采样平滑）
    saliency = F.conv_transpose2d(saliency, bs_kernel, padding=bs_pad, stride=h // grid_size)

    # 使用双线性插值恢复到原始尺寸
    saliency = F.interpolate(saliency, size=(h, w), mode='bilinear', align_corners=True)

    # 归一化显著性图
    saliency = rescale_intensity(saliency)

    return saliency