# niftiio.py
"""
Utils for datasets
"""

import numpy as np
import SimpleITK as sitk

def read_nii_bysitk(input_fid, peel_info=False):
    """
    通过 SimpleITK 读取 NIfTI 格式文件，并将其转换为 numpy 数组。

    Args:
        input_fid (str): 输入 NIfTI 文件的路径。
        peel_info (bool): 是否提取图像的额外元数据信息（如 spacing、origin、direction 等）。

    Returns:
        如果 peel_info 为 True：
            返回一个二元组 (img_np, info_obj)，其中 img_np 为 numpy 数组，info_obj 为包含图像元数据的字典。
        否则：
            仅返回 numpy 数组 img_np。
    """
    # 使用 SimpleITK 读取图像文件
    img_obj = sitk.ReadImage(input_fid)
    # 将 SimpleITK 图像对象转换为 numpy 数组
    img_np = sitk.GetArrayFromImage(img_obj)

    if peel_info:
        # 如果需要提取元数据信息，则构造包含 spacing、origin、direction 以及数组尺寸的字典
        info_obj = {
            "spacing": img_obj.GetSpacing(),       # 图像像素间距
            "origin": img_obj.GetOrigin(),           # 图像原点
            "direction": img_obj.GetDirection(),     # 图像方向矩阵
            "array_size": img_np.shape               # numpy 数组的形状信息
        }
        return img_np, info_obj
    else:
        # 否则仅返回图像的 numpy 数组
        return img_np

def convert_to_sitk(input_mat, peeled_info):
    """
    将 numpy 数组转换为 SimpleITK 图像对象，并设置基本的元数据信息。

    Args:
        input_mat (numpy array): 输入的图像数组。
        peeled_info (dict): 包含 spacing、origin 和 direction 的字典信息。

    Returns:
        nii_obj: 转换后的 SimpleITK 图像对象，已设置相应的元数据。
    """
    # 通过 numpy 数组创建一个 SimpleITK 图像对象
    nii_obj = sitk.GetImageFromArray(input_mat)
    if peeled_info:
        # 设置图像的 spacing、origin 和 direction 信息
        nii_obj.SetSpacing(peeled_info["spacing"])
        nii_obj.SetOrigin(peeled_info["origin"])
        nii_obj.SetDirection(peeled_info["direction"])
    return nii_obj

def np2itk(img, ref_obj):
    """
    根据参考 SimpleITK 对象，将 numpy 数组转换为具有相同元数据信息的 SimpleITK 图像对象。

    Args:
        img (numpy array): 输入的图像数组。
        ref_obj (SimpleITK.Image): 参考图像对象，用于复制 spacing、origin 和 direction 信息。

    Returns:
        itk_obj: 转换后的 SimpleITK 图像对象，其元数据信息与 ref_obj 相同。
    """
    # 将 numpy 数组转换为 SimpleITK 图像对象
    itk_obj = sitk.GetImageFromArray(img)
    # 从参考图像中复制 spacing、origin 和 direction 信息
    itk_obj.SetSpacing(ref_obj.GetSpacing())
    itk_obj.SetOrigin(ref_obj.GetOrigin())
    itk_obj.SetDirection(ref_obj.GetDirection())
    return itk_obj