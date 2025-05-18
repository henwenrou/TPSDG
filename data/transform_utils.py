# transform_utils.py
"""
Utilities for image transforms, part of the code base credits to Dr. Jo Schlemper
"""
# 该文件提供图像几何和强度变换的工具，部分代码归功于 Jo Schlemper 博士

from os.path import join
import torch
import numpy as np
import torchvision.transforms as deftfx   # torchvision 的 transform 模块，用于构造组合变换
import data.image_transforms as myit  # 自定义的图像变换模块
import copy                                # 用于深拷贝数据，避免原数据被修改
import math

# 定义用于数据增强的参数字典
my_augv = {
    'flip': {
        'v': False,   # 垂直翻转开关
        'h': False,   # 水平翻转开关
        't': False,   # 翻转（可能指沿第三个维度翻转）的开关
        'p': 0.25     # 翻转概率
    },
    'affine': {
        'rotate': 20,         # 随机旋转角度范围（单位：度）
        'shift': (15, 15),     # 随机平移范围（横向、纵向）
        'shear': 20,          # 随机错切角度范围
        'scale': (0.5, 1.5),    # 随机缩放范围
    },
    'elastic': {
        'alpha': 20,  # 弹性变换的 alpha 参数，控制形变幅度
        'sigma': 5    # 弹性变换的 sigma 参数，控制平滑程度
    },
    'reduce_2d': True,  # 是否将数据简化为2D（通常在2D标签处理时使用）
    'gamma_range': (0.2, 1.8),  # gamma 变换范围
    'noise': {
        'noise_std': 0.15,   # 噪声标准差，用于添加高斯噪声
        'clip_pm1': False    # 是否将添加噪声后的结果裁剪到[-1,1]
    },
    'bright_contrast': {
        'contrast': (0.60, 1.5),  # 对比度调整范围
        'bright': (-10, 10)       # 亮度调整范围
    }
}

# 将上述数据增强参数封装为一个字典
tr_aug = {
    'aug': my_augv
}

def get_geometric_transformer(aug, order=3):
    """
    根据传入的增强参数构造几何变换组合器。

    Args:
        aug: 增强参数字典，包含 'flip'、'affine'、'elastic' 等选项
        order: 插值顺序，默认为3（通常表示三次插值）

    Returns:
        input_transform: 一个组合的几何变换对象（由 torchvision.transforms.Compose 构成）
    """
    # 从增强参数中获取各项参数，如果不存在则使用默认值
    affine = aug['aug'].get('affine', 0)
    alpha = aug['aug'].get('elastic', {'alpha': 0})['alpha']
    sigma = aug['aug'].get('elastic', {'sigma': 0})['sigma']
    flip = aug['aug'].get('flip', {'v': True, 'h': True, 't': True, 'p': 0.125})

    tfx = []  # 用于存放各个几何变换操作

    # 如果有翻转操作，则添加随机翻转变换（3D翻转）
    if 'flip' in aug['aug']:
        tfx.append(myit.RandomFlip3D(**flip))

    # 如果有仿射变换，则添加随机仿射变换（旋转、平移、剪切、缩放等）
    if 'affine' in aug['aug']:
        tfx.append(myit.RandomAffine(
            affine.get('rotate'),
            affine.get('shift'),
            affine.get('shear'),
            affine.get('scale'),
            affine.get('scale_iso', True),  # 是否保持各轴比例一致
            order=order
        ))

    # 如果有弹性变换，则添加弹性变换操作
    if 'elastic' in aug['aug']:
        tfx.append(myit.ElasticTransform(alpha, sigma))
    # ★ 在所有几何变换前先做 Z-score 归一化
    all_tfx = [ZScoreNorm()] + tfx
    # 使用 torchvision 的 Compose 将所有几何变换组合起来
    input_transform = deftfx.Compose(all_tfx)
    return input_transform

def get_intensity_transformer(aug):
    """
    构造图像强度变换函数，主要包括 gamma 变换、亮度对比度调整以及添加噪声。

    Returns:
        compile_transform: 一个接受图像作为输入，并依次执行各强度变换的函数
    """
    def gamma_tansform(img):
        # 从增强参数中获取 gamma 变换范围
        gamma_range = aug['aug']['gamma_range']
        if isinstance(gamma_range, tuple):
            # 随机生成一个 gamma 值
            gamma = np.random.rand() * (gamma_range[1] - gamma_range[0]) + gamma_range[0]
            cmin = img.min()
            irange = (img.max() - cmin + 1e-5)
            # 调整图像，使其值在一个较好的范围内进行 gamma 变换
            img = img - cmin + 1e-5
            img = irange * np.power(img / irange, gamma)
            img = img + cmin
        elif gamma_range == False:
            pass  # 若 gamma_range 为 False，则不进行 gamma 变换
        else:
            raise ValueError("Cannot identify gamma transform range {}".format(gamma_range))
        return img

    def brightness_contrast(img):
        """
        根据随机生成的对比度和亮度参数调整图像。
        参考文献：Chaitanya,K. et al. Semi-Supervised and Task-Driven data augmentation ...
        """
        cmin, cmax = aug['aug']['bright_contrast']['contrast']
        bmin, bmax = aug['aug']['bright_contrast']['bright']
        c = np.random.rand() * (cmax - cmin) + cmin  # 随机对比度系数
        b = np.random.rand() * (bmax - bmin) + bmin  # 随机亮度偏移量
        img_mean = img.mean()  # 图像均值
        img = (img - img_mean) * c + img_mean + b  # 对比度调整后再加上亮度偏移
        return img

    def zm_gaussian_noise(img):
        """
        为图像添加零均值高斯噪声。
        """
        noise_sigma = aug['aug']['noise']['noise_std']
        noise_vol = np.random.randn(*img.shape) * noise_sigma
        img = img + noise_vol
        # 如果设置了裁剪，将结果裁剪到 [-1, 1] 区间
        if aug['aug']['noise']['clip_pm1']:
            img = np.clip(img, -1.0, 1.0)
        return img

    def compile_transform(img):
        # 依次应用亮度对比度调整、gamma 变换、添加噪声
        if 'bright_contrast' in aug['aug'].keys():
            img = brightness_contrast(img)
        if 'gamma_range' in aug['aug'].keys():
            img = gamma_tansform(img)
        if 'noise' in aug['aug'].keys():
            img = zm_gaussian_noise(img)
        return img

    return compile_transform

class transform_with_label(object):
    def __init__(self, aug, add_pseudolabel=False):
        """
        用于同时对图像和标签进行几何变换和强度变换的操作类。

        假定输入图像为形状 [H x W x C + CL]，其中 CL 表示标签的通道数（非 one-hot 编码）。

        Args:
            aug (dict): 包含数据增强参数的字典。
            add_pseudolabel (bool): 是否添加伪标签（当前未使用）。
        """
        self.aug = aug
        # 构造几何变换操作（如翻转、仿射、弹性变换）
        self.geometric_tfx = get_geometric_transformer(aug)

    def intensity_tfx(self, image):
        """
        对单张图像执行强度变换，包括 gamma 调整、亮度对比度调整以及添加高斯噪声。

        Args:
            image (numpy array): 输入图像。

        Returns:
            image: 强度变换后的图像。
        """
        aug = self.aug
        # 内部定义与 get_intensity_transformer 类似的函数

        def gamma_tansform(img):
            gamma_range = aug['aug']['gamma_range']
            if isinstance(gamma_range, tuple):
                gamma = np.random.rand() * (gamma_range[1] - gamma_range[0]) + gamma_range[0]
                cmin = img.min()
                irange = (img.max() - cmin + 1e-5)
                img = img - cmin + 1e-5
                img = irange * np.power(img / irange, gamma)
                img = img + cmin
            elif gamma_range == False:
                pass
            else:
                raise ValueError("Cannot identify gamma transform range {}".format(gamma_range))
            return img

        def brightness_contrast(img):
            cmin, cmax = aug['aug']['bright_contrast']['contrast']
            bmin, bmax = aug['aug']['bright_contrast']['bright']
            c = np.random.rand() * (cmax - cmin) + cmin
            b = np.random.rand() * (bmax - bmin) + bmin
            img_mean = img.mean()
            img = (img - img_mean) * c + img_mean + b
            return img

        def zm_gaussian_noise(img):
            noise_sigma = aug['aug']['noise']['noise_std']
            noise_vol = np.random.randn(*img.shape) * noise_sigma
            img = img + noise_vol
            if aug['aug']['noise']['clip_pm1']:
                img = np.clip(img, -1.0, 1.0)
            return img

        # 按照配置依次对图像应用各种强度变换
        if 'bright_contrast' in aug['aug'].keys():
            image = brightness_contrast(image)
        if 'gamma_range' in aug['aug'].keys():
            image = gamma_tansform(image)
        if 'noise' in aug['aug'].keys():
            image = zm_gaussian_noise(image)
        return image

    def geometric_aug(self, comp, c_label, c_img, nclass, is_train, use_onehot=False):
        """
        对图像和标签进行几何变换。

        Args:
            comp (numpy array): 输入图像与标签拼接后的数组，形状为 [H x W x C + c_label]。
            c_label (int): 标签通道数（当前只支持单通道 2D 标签）。
            c_img (int): 图像通道数。
            nclass (int): 类别数量，用于将标签转换为 one-hot 编码。
            is_train (bool): 是否处于训练阶段，只有训练阶段才进行几何变换。
            use_onehot (bool): 是否使用 one-hot 编码的标签。

        Returns:
            t_img: 经过几何变换后的图像部分。
            t_label: 经过几何变换后的标签部分（如果 use_onehot 为 False，则返回 compact 标签）。
        """
        comp = copy.deepcopy(comp)  # 深拷贝，防止原始数据被修改

        if (use_onehot is True) and (c_label != 1):
            raise NotImplementedError("Only allow compact label, also the label can only be 2d")
        # 保证图像部分与标签部分在最后一个维度拼接后，其总通道数为 c_img + 1
        assert c_img + 1 == comp.shape[-1], "only allow single slice 2D label"

        if is_train is True:
            _label = comp[..., c_img]  # 提取 compact 标签（假定标签在最后一个通道）
            # 将标签转换为 one-hot 编码，生成 shape 为 [H, W, nclass] 的 one-hot 标签
            _h_label = np.float32(np.arange(nclass) == (_label[..., None]))
            # 将图像和 one-hot 标签拼接
            comp = np.concatenate([comp[..., :c_img], _h_label], -1)
            # 对拼接后的数据进行几何变换
            comp = self.geometric_tfx(comp)
            # 对 one-hot 标签部分进行四舍五入，使其变为 0 或 1
            t_label_h = comp[..., c_img:]
            t_label_h = np.rint(t_label_h)
            # 分离出图像部分
            t_img = comp[..., 0:c_img]
        if use_onehot is True:
            t_label = t_label_h
        else:
            # 若不使用 one-hot，则取 one-hot 中概率最高的通道作为标签
            t_label = np.expand_dims(np.argmax(t_label_h, axis=-1), -1)
        return t_img, t_label

    def __call__(self, comp, c_label, c_img, nclass, is_train, use_onehot=False):
        """
        同时对图像和标签进行几何和强度变换。

        Args:
            comp (numpy array): 拼接后的图像和标签数组，形状为 [H x W x C + c_label]。
            c_label (int): 标签通道数。
            c_img (int): 图像通道数。
            nclass (int): 类别数，用于 one-hot 编码。
            is_train (bool): 是否为训练阶段（非训练阶段仅进行强度变换）。
            use_onehot (bool): 是否使用 one-hot 编码标签。

        Returns:
            t_img: 最终变换后的图像部分（经过几何和强度变换）。
            t_label: 最终变换后的标签部分。
        """
        comp = copy.deepcopy(comp)  # 深拷贝避免对原始数据进行修改

        if (use_onehot is True) and (c_label != 1):
            raise NotImplementedError("Only allow compact label, also the label can only be 2d")
        assert c_img + 1 == comp.shape[-1], "only allow single slice 2D label"

        if is_train is True:
            _label = comp[..., c_img]  # 获取 compact 标签
            # 将标签转换为 one-hot 编码
            _h_label = np.float32(np.arange(nclass) == (_label[..., None]))
            comp = np.concatenate([comp[..., :c_img], _h_label], -1)
            # 进行几何变换
            comp = self.geometric_tfx(comp)
            t_label_h = comp[..., c_img:]
            t_label_h = np.rint(t_label_h)
            t_img = comp[..., 0:c_img]

        # 对图像部分进行强度变换（无论训练与否都执行）
        t_img = self.intensity_tfx(t_img)

        if use_onehot is True:
            t_label = t_label_h
        else:
            t_label = np.expand_dims(np.argmax(t_label_h, axis=-1), -1)
        return t_img, t_label

class ZScoreNorm:
    """对每个 3D/2D volume 做 z-score 归一化"""
    def __call__(self, img: np.ndarray):
        img = img.astype(np.float32)
        mu, sigma = img.mean(), img.std()
        if sigma < 1e-6:               # 防止除零
            return np.zeros_like(img)
        return (img - mu) / sigma