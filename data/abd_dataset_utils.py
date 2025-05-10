# abd_dataset_utils.py
"""
Utils for datasets
"""
import numpy as np
import os
import sys
import numpy as np
import pdb
import SimpleITK as sitk # 用于读取和处理医学图像
from niftiio import read_nii_bysitk

class mean_std_norm(object):
    """
    归一化操作类：使用均值和标准差对数据进行标准化
    """
    def __init__(self,mean=None,std=None):
        self.mean=mean
        self.std=std

    def __call__(self,x_in):
        if self.mean is None:
            return (x_in-x_in.mean())/x_in.std()
        else:
            return (x_in-self.mean)/self.std

def get_normalize_op(modality, fids):
    """
    获取归一化操作
    Args:
        modality (str): 医学影像的模态（如 CT 或 MR）
        fids (list): 文件路径列表，用于计算 CT 图像的全局统计量
    Returns:
        mean_std_norm: 返回归一化操作的实例
    """

    def get_CT_statistics(scan_fids):
        """
        计算 CT 图像的全局均值和标准差（CT 图像是定量的，适合计算全局统计量）
        在真实应用场景中，可能无法一次性加载所有数据，因此应将统计计算与数据加载分离
        但在未知数据集的情况下，无法获得数据的统计信息，因此只能对每个 3D 图像进行标准化，使其均值为 0，标准差为 1
        Args:
            scan_fids (list): CT 图像文件路径列表
        Returns:
            meanval (float): 全局均值
            global_std (float): 全局标准差
        """
        total_val = 0  # 记录所有像素值的总和
        n_pix = 0  # 记录像素点的数量

        # 计算全局均值
        for fid in scan_fids:
            in_img = read_nii_bysitk(fid)  # 读取 NIfTI 图像
            total_val += in_img.sum()  # 计算所有像素的总和
            n_pix += np.prod(in_img.shape)  # 计算像素点总数
            del in_img  # 释放内存

        meanval = total_val / n_pix  # 计算均值

        total_var = 0  # 记录方差总和

        # 计算全局方差
        for fid in scan_fids:
            in_img = read_nii_bysitk(fid)
            total_var += np.sum((in_img - meanval) ** 2)  # 计算所有像素点的方差总和
            del in_img  # 释放内存

        var_all = total_var / n_pix  # 计算总体方差
        global_std = var_all ** 0.5  # 计算标准差

        return meanval, global_std  # 返回均值和标准差

    # 如果影像模态是 MR（如 CHAOST2 或 LGE）
    if modality == 'CHAOST2' or modality == 'LGE':
        def MR_normalize(x_in):
            """
            MR 图像的标准化方法：
            由于 MR 图像的信号强度不具有绝对物理单位，通常采用对每个样本单独进行零均值单位方差标准化的方法
            """
            return (x_in - x_in.mean()) / x_in.std()

        return mean_std_norm()  # 由于 MR 图像不需要全局统计量，直接返回默认的归一化类实例

    # 如果影像模态是 CT（如 SABSCT 或 bSSFP）
    elif modality == 'SABSCT' or modality == 'bSSFP':
        # 计算 CT 图像的全局均值和标准差
        ct_mean, ct_std = get_CT_statistics(fids)

        def CT_normalize(x_in):
            """
            CT 图像的标准化方法：
            由于 CT 图像的像素值代表定量的物理测量值（如 HU 值），可以使用全局统计量进行标准化
            """
            return (x_in - ct_mean) / ct_std

        return mean_std_norm(ct_mean, ct_std)  # 返回带有全局均值和标准差的归一化类实例