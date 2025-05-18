# AbdominalDataset.py
# Dataloader for abdominal images,
import glob  # 用于文件路径匹配，获取数据集内所有符合条件的文件名
import numpy as np  # 数组和数值计算库
import data.niftiio as nio  # 自定义的 nifti 文件读取模块，用于读取 NIfTI 格式医学影像
import data.transform_utils as trans  # 自定义的数据增强/变换工具包
from data.transform_utils import ZScoreNorm
import torch  # PyTorch 库，用于深度学习模型及 tensor 相关操作
import os  # 文件和目录操作库
import platform  # 获取操作系统和主机信息
import torch.utils.data as torch_data  # PyTorch 数据加载工具，提供 Dataset 和 DataLoader 类
import math  # 数学计算库，主要用于取整、四舍五入等操作
import itertools  # 提供迭代器工具，便于对列表进行拼接等操作

# 导入位置尺度增强类，用于对图像进行额外的增强操作（例如局部对比度调整等）
from data.location_scale_augmentation import LocationScaleAugmentation

# 数据集根目录，这里存放腹部图像数据
BASEDIR = 'data/processed/abdominal'

hostname = platform.node()
# 打印当前运行机器和使用的数据集目录
print(f'Running on machine {hostname}, using dataset from {BASEDIR}')

# 定义标签名称列表，分别代表背景、肝、右肾、左肾、脾脏
LABEL_NAME = ["bg", "liver", "rk", "lk", "spleen"]

# 从 dataloaders.niftiio 模块导入读取 NIfTI 文件的函数
from data.niftiio import read_nii_bysitk


# --------------------- 均值标准差归一化类 ---------------------
class mean_std_norm(object):
    """
    用于数据归一化的类，根据给定的均值和标准差对输入数据进行标准化。
    如果未指定均值和标准差，则直接对输入数据计算均值和标准差。
    """

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, x_in):
        """
        对输入数据进行标准化
        Args:
            x_in (numpy.array): 输入数据
        Returns:
            标准化后的数据
        """
        if self.mean is None:
            # 当未提供全局均值和标准差时，直接计算输入数据的均值和标准差进行归一化
            return (x_in - x_in.mean()) / x_in.std()
        else:
            # 使用提供的均值和标准差进行归一化
            return (x_in - self.mean) / self.std


# --------------------- 获取归一化操作函数 ---------------------
def get_normalize_op(fids, domain=False):
    """
    根据文件列表 fids，返回归一化操作对象。
    当 domain 为 True 时，会先计算全局统计信息（均值、标准差），用于 CT 数据的归一化；
    当 domain 为 False 时，直接返回一个动态归一化操作（不依赖于全局统计量）。

    Args:
        fids (list): 文件路径列表，用于计算统计量
        domain (bool): 是否使用跨域（CT-MR）统计，True 表示计算全局统计量

    Returns:
        mean_std_norm 实例：归一化操作对象
    """

    def get_statistics(scan_fids):
        """
        内部函数：计算所有图像的全局均值和标准差
        Args:
            scan_fids (list): 图像文件路径列表
        Returns:
            meanval: 全局均值
            global_std: 全局标准差
        """
        total_val = 0  # 累计所有图像像素值总和
        n_pix = 0  # 累计所有像素总数
        for fid in scan_fids:
            in_img = read_nii_bysitk(fid)  # 读取图像文件
            total_val += in_img.sum()  # 累计像素和
            n_pix += np.prod(in_img.shape)  # 累计像素总数
            del in_img  # 删除变量，释放内存
        meanval = total_val / n_pix  # 计算全局均值

        total_var = 0  # 累计方差的总和
        for fid in scan_fids:
            in_img = read_nii_bysitk(fid)
            total_var += np.sum((in_img - meanval) ** 2)  # 计算每个像素与均值差平方的和
            del in_img
        var_all = total_var / n_pix  # 计算总体方差
        global_std = var_all ** 0.5  # 标准差为方差的平方根

        return meanval, global_std

    if not domain:
        # 当不需要跨域统计时，直接返回动态归一化对象（在调用时会计算当前数据的均值和标准差）
        return mean_std_norm()
    else:
        # 当需要全局统计时，计算全局均值和标准差，再返回归一化对象
        _mean, _std = get_statistics(fids)
        return mean_std_norm(_mean, _std)


# --------------------- 定义腹部图像数据集类 ---------------------
class AbdominalDataset(torch_data.Dataset):
    """
    自定义 PyTorch 数据集类，用于加载腹部医学图像数据。
    支持训练、验证、测试以及全测试（target domain testing）的模式。
    """

    def __init__(self, mode, transforms, base_dir, domains: list, idx_pct=[0.7, 0.1, 0.2],
                 tile_z_dim=3, extern_norm_fn=None, location_scale=False):
        """
        Args:
            mode (str): 数据集模式，可选值为 'train', 'val', 'test', 'test_all'
            transforms: 数据增强函数，用于对图像和标签进行变换
            base_dir (str): 数据集根目录
            domains (list): 数据域列表，表示不同来源的图像数据
            idx_pct (list): 数据划分比例，如 [0.7, 0.1, 0.2] 分别对应训练、验证、测试
            tile_z_dim (int): z 方向上切片的重复数，用于扩充图像通道
            extern_norm_fn: 外部传入的归一化函数，用于跨域归一化
            location_scale (bool): 是否应用位置尺度增强
        """
        self._base_dir = base_dir
        # 如果路径存在问题，可以用 os.path.abspath() 转换为绝对路径
        import os
        abs_base_dir = os.path.abspath(self._base_dir)
        super(AbdominalDataset, self).__init__()
        self.transforms = transforms
        self.is_train = True if mode == 'train' else False  # 根据模式确定是否为训练阶段
        self.phase = mode
        self.domains = domains
        self.all_label_names = LABEL_NAME  # 数据标签列表
        self.nclass = len(LABEL_NAME)  # 类别数
        self.tile_z_dim = tile_z_dim
        self._base_dir = base_dir
        self.idx_pct = idx_pct

        # 遍历每个域，加载该域内所有图像文件的 id（文件名中最后部分的数字），并按数字大小排序
        self.img_pids = {}
        for _domain in self.domains:  # 加载各个域的文件名
            self.img_pids[_domain] = sorted(
                [fid.split("_")[-1].split(".nii.gz")[0] for fid in
                 glob.glob(self._base_dir + "/" + _domain + "/processed/image_*.nii.gz")],
                key=lambda x: int(x)
            )

        # 根据模式及划分比例生成扫描（patient）id的字典，用于划分训练、验证、测试数据
        self.scan_ids = self.__get_scanids(mode, idx_pct)

        self.info_by_scan = None  # 存储每个扫描的元数据信息
        # 根据 scan_ids 搜索图像和标签文件，返回一个字典结构
        self.sample_list = self.__search_samples(self.scan_ids)

        # 根据数据集模式设置当前加载的 scan id
        if self.is_train:
            self.pid_curr_load = self.scan_ids
        elif mode == 'val':
            self.pid_curr_load = self.scan_ids
        elif mode == 'test':  # 源域测试
            self.pid_curr_load = self.scan_ids
        elif mode == 'test_all':  # 目标域测试，包含所有扫描数据
            self.pid_curr_load = self.scan_ids

        # 使用外部传入的归一化函数计算归一化统计量，针对第一个域的数据
        self.normalize_op = extern_norm_fn([itm['img_fid'] for _, itm in self.sample_list[self.domains[0]].items()])

        print(f'For {self.phase} on {[_dm for _dm in self.domains]} using scan ids {self.pid_curr_load}')

        # 预先将数据加载到内存中，形成切片级别的数据集（2D）
        self.actual_dataset = self.__read_dataset()
        self.size = len(self.actual_dataset)  # 数据集中切片数量

        # 如果启用了位置尺度增强，则初始化对应增强对象
        if location_scale:
            print(f'Applying Location Scale Augmentation on {mode} split')
            self.location_scale = LocationScaleAugmentation(vrange=(0., 1.), background_threshold=0.01)
        else:
            self.location_scale = None

    # --------------------- 内部方法：根据划分比例生成 scan ids ---------------------
    def __get_scanids(self, mode, idx_pct):
        """
        根据每个域内图像的数量，按照给定比例划分训练、验证和测试数据。
        Args:
            mode (str): 数据集模式
            idx_pct (list): 划分比例，顺序为 [训练, 验证, 测试]
        Returns:
            分域后的 scan ids 字典
        """
        tr_ids = {}  # 训练集 scan ids
        val_ids = {}  # 验证集 scan ids
        te_ids = {}  # 测试集 scan ids
        te_all_ids = {}  # 包含所有 scan ids（训练+验证+测试）

        for _domain in self.domains:
            dset_size = len(self.img_pids[_domain])  # 当前域的图像总数
            tr_size = round(dset_size * idx_pct[0])  # 训练集数量
            val_size = math.floor(dset_size * idx_pct[1])  # 验证集数量
            te_size = dset_size - tr_size - val_size  # 测试集数量

            # 划分当前域内的测试、验证和训练集
            te_ids[_domain] = self.img_pids[_domain][:te_size]
            val_ids[_domain] = self.img_pids[_domain][te_size:te_size + val_size]
            tr_ids[_domain] = self.img_pids[_domain][te_size + val_size:]
            # 合并所有 scan ids
            te_all_ids[_domain] = list(itertools.chain(tr_ids[_domain], te_ids[_domain], val_ids[_domain]))

        # 根据当前阶段返回对应的 scan ids 字典
        if self.phase == 'train':
            return tr_ids
        elif self.phase == 'val':
            return val_ids
        elif self.phase == 'test':
            return te_ids
        elif self.phase == 'test_all':
            return te_all_ids

    # --------------------- 内部方法：搜索图像和标签文件 ---------------------
    def __search_samples(self, scan_ids):
        """
        在每个域中，根据 scan ids 查找对应的图像和标签文件路径
        Returns:
            字典，结构为 {域: {scan_id: {"img_fid": image_file, "lbs_fid": label_file}}}
        """
        out_list = {}
        for _domain, id_list in scan_ids.items():
            out_list[_domain] = {}
            for curr_id in id_list:
                curr_dict = {}
                # 构造图像文件路径
                _img_fid = os.path.join(self._base_dir, _domain, 'processed', f'image_{curr_id}.nii.gz')
                # 构造标签文件路径
                _lb_fid = os.path.join(self._base_dir, _domain, 'processed', f'label_{curr_id}.nii.gz')
                curr_dict["img_fid"] = _img_fid
                curr_dict["lbs_fid"] = _lb_fid
                out_list[_domain][str(curr_id)] = curr_dict
        return out_list

    # --------------------- 内部方法：将数据读取到内存 ---------------------
    def __read_dataset(self):
        """
        读取数据集，将图像和标签文件加载到内存中，并按切片（2D）组织。
        同时保存每个扫描的元数据信息。
        Returns:
            out_list: 包含所有切片样本的列表，每个元素为字典形式包含图像、标签、扫描信息、切片索引等
        """
        out_list = []
        self.info_by_scan = {}  # 用于保存每个扫描的元数据信息
        glb_idx = 0  # 全局切片索引

        # 遍历每个域及其对应的样本
        for _domain, _sample_list in self.sample_list.items():
            for scan_id, itm in _sample_list.items():
                # 如果当前 scan_id 不在当前加载列表中，则跳过
                if scan_id not in self.pid_curr_load[_domain]:
                    continue

                # 读取图像，同时提取元数据信息（如 spacing, origin, direction 等）
                img, _info = nio.read_nii_bysitk(itm["img_fid"], peel_info=True)
                # 存储该扫描的元数据
                self.info_by_scan[_domain + '_' + scan_id] = _info

                img = np.float32(img)  # 转换图像数据为 float32 类型
                if self.normalize_op.mean is not None:
                    # 如果归一化操作有预设均值，则构造体积信息字典
                    vol_info = {
                        'vol_vmin': img.min(),
                        'vol_vmax': img.max(),
                        'vol_mean': self.normalize_op.mean,
                        'vol_std': self.normalize_op.mean  # NOTE: 此处存在 bug，应该为 self.normalize_op.std
                    }
                else:
                    # 否则直接计算图像的统计信息
                    vol_info = {
                        'vol_vmin': img.min(),
                        'vol_vmax': img.max(),
                        'vol_mean': img.mean(),
                        'vol_std': img.std()
                    }
                # 对图像进行归一化处理
                img = self.normalize_op(img)

                # 读取标签数据，并转换为 float32 类型
                lb = nio.read_nii_bysitk(itm["lbs_fid"])
                lb = np.float32(lb)

                # 将图像和标签转换为 [height, width, slices] 形式
                img = np.transpose(img, (1, 2, 0))
                lb = np.transpose(lb, (1, 2, 0))

                # 确保图像和标签在切片数（z 轴）上匹配
                assert img.shape[-1] == lb.shape[-1]

                # 将第一帧单独处理为起始帧
                out_list.append({
                    "img": img[..., 0:1],
                    "lb": lb[..., 0:1],
                    "is_start": True,  # 标记为起始帧
                    "is_end": False,
                    "domain": _domain,
                    "nframe": img.shape[-1],  # 扫描的总切片数
                    "scan_id": _domain + "_" + scan_id,
                    "z_id": 0,  # 当前切片索引
                    "vol_info": vol_info
                })
                glb_idx += 1

                # 处理中间帧，每一帧均标记为非起始也非结束帧
                for ii in range(1, img.shape[-1] - 1):
                    out_list.append({
                        "img": img[..., ii:ii + 1],
                        "lb": lb[..., ii:ii + 1],
                        "is_start": False,
                        "is_end": False,
                        "nframe": -1,
                        "domain": _domain,
                        "scan_id": _domain + "_" + scan_id,
                        "z_id": ii,
                        "vol_info": vol_info
                    })
                    glb_idx += 1

                # 处理最后一帧，标记为结束帧
                ii += 1  # 最后一帧的索引
                out_list.append({
                    "img": img[..., ii:ii + 1],
                    "lb": lb[..., ii:ii + 1],
                    "is_start": False,
                    "is_end": True,
                    "nframe": -1,
                    "domain": _domain,
                    "scan_id": _domain + "_" + scan_id,
                    "z_id": ii,
                    "vol_info": vol_info
                })
                glb_idx += 1

        return out_list

    # --------------------- __getitem__方法：返回单个样本 ---------------------
    def __getitem__(self, index):
        """
        根据索引返回一个样本，包含图像、标签以及辅助信息
        如果是训练阶段，可能会进行额外的数据增强（如位置尺度增强）
        """
        index = index % len(self.actual_dataset)
        curr_dict = self.actual_dataset[index]

        if self.is_train:
            if self.location_scale is not None:
                # 若启用了位置尺度增强，则对图像进行增强处理
                img = curr_dict["img"].copy()
                lb = curr_dict["lb"].copy()
                # 先进行反归一化，将图像还原到原始范围
                img = self.denorm_(img, curr_dict['vol_info'])

                # 应用全局位置尺度增强
                GLA = self.location_scale.Global_Location_Scale_Augmentation(img.copy())
                GLA = self.renorm_(GLA, curr_dict['vol_info'])

                # 应用局部位置尺度增强（利用标签信息）
                LLA = self.location_scale.Local_Location_Scale_Augmentation(img.copy(), lb.astype(np.int32))
                LLA = self.renorm_(LLA, curr_dict['vol_info'])

                # 将全局和局部增强图像及标签拼接为一个整体
                comp = np.concatenate([GLA, LLA, curr_dict["lb"]], -1)
                if self.transforms:
                    # 对拼接后的数据进行额外变换处理
                    timg, lb = self.transforms(comp, c_img=2, c_label=1, nclass=self.nclass, is_train=self.is_train,
                                               use_onehot=False)
                    # 分离变换后的全局和局部增强图像
                    GLA, LLA = np.split(timg, 2, -1)
                img = GLA
                aug_img = LLA
                # 将增强后的图像转换为 tensor 格式
                aug_img = np.float32(aug_img)
                aug_img = np.transpose(aug_img, (2, 0, 1))
                aug_img = torch.from_numpy(aug_img)
            else:
                # 若未启用位置尺度增强，则仅将图像与标签拼接后进行数据增强（若有定义 transforms）
                comp = np.concatenate([curr_dict["img"], curr_dict["lb"]], axis=-1)
                if self.transforms:
                    img, lb = self.transforms(comp, c_img=1, c_label=1, nclass=self.nclass, is_train=self.is_train,
                                              use_onehot=False)
                aug_img = 1
        else:
            # 非训练阶段直接使用原始图像和标签
            img = curr_dict['img']
            img = ZScoreNorm()(img)
            lb = curr_dict['lb']
            aug_img = 1

        # 将图像和标签转换为 float32 类型，并调整通道顺序为 (C, H, W)
        img = np.float32(img)
        lb = np.float32(lb)
        img = np.transpose(img, (2, 0, 1))
        lb = np.transpose(lb, (2, 0, 1))

        # 转换为 PyTorch 的 tensor
        img = torch.from_numpy(img)
        lb = torch.from_numpy(lb)

        # 如果 tile_z_dim 大于 1，则在通道上对图像进行重复扩充
        if self.tile_z_dim > 1:
            img = img.repeat([self.tile_z_dim, 1, 1])
            assert img.ndimension() == 3

        # 获取样本的辅助信息：起始、结束帧标识、切片数量、扫描 id、切片索引
        is_start = curr_dict["is_start"]
        is_end = curr_dict["is_end"]
        nframe = np.int32(curr_dict["nframe"])
        scan_id = curr_dict["scan_id"]
        z_id = curr_dict["z_id"]

        # 组装最终返回的样本字典
        sample = {
            "images": img,  # 图像 tensor
            "labels": lb[0].long(),  # 标签 tensor（取第一通道，并转换为 long 类型）
            "is_start": is_start,  # 是否为起始帧
            "is_end": is_end,  # 是否为结束帧
            "nframe": nframe,  # 该扫描的切片总数
            "scan_id": scan_id,  # 扫描 id
            "z_id": z_id,  # 当前切片索引
            "aug_images": aug_img,  # 额外的增强图像（如位置尺度增强结果）
        }
        return sample

    # --------------------- 反归一化方法 ---------------------
    def denorm_(self, img, vol_info):
        """
        将归一化后的图像还原到原始范围（0-1）
        Args:
            img (numpy.array): 归一化后的图像
            vol_info (dict): 包含原始图像统计信息（最小值、最大值、均值、标准差）
        Returns:
            反归一化后的图像
        """
        vmin, vmax, vmean, vstd = vol_info['vol_vmin'], vol_info['vol_vmax'], vol_info['vol_mean'], vol_info['vol_std']
        return ((img * vstd + vmean) - vmin) / (vmax - vmin)

    # --------------------- 再归一化方法 ---------------------
    def renorm_(self, img, vol_info):
        """
        将经过增强的图像重新归一化，确保与原始归一化范围一致
        Args:
            img (numpy.array): 增强后的图像
            vol_info (dict): 包含原始图像统计信息
        Returns:
            再归一化后的图像
        """
        vmin, vmax, vmean, vstd = vol_info['vol_vmin'], vol_info['vol_vmax'], vol_info['vol_mean'], vol_info['vol_std']
        return ((img * (vmax - vmin) + vmin) - vmean) / vstd

    # --------------------- 获取数据集大小 ---------------------
    def __len__(self):
        """
        返回数据集中样本的总数量
        """
        return len(self.actual_dataset)


# --------------------- 构造训练时使用的 transform 函数 ---------------------
tr_func = trans.transform_with_label(trans.tr_aug)

from functools import partial


def get_training(modality, location_scale, idx_pct=[0.7, 0.1, 0.2], tile_z_dim=3):
    """
    构造训练数据集
    Args:
        modality: 数据域列表，如不同的采集协议（CT、MR等）
        location_scale: 是否应用位置尺度增强
        idx_pct: 数据划分比例（训练、验证、测试）
        tile_z_dim: z 方向切片重复数
    Returns:
        AbdominalDataset 的训练数据集实例
    """
    return AbdominalDataset(
        idx_pct=idx_pct,
        mode='train',
        domains=modality,
        transforms=tr_func,
        base_dir=BASEDIR,
        extern_norm_fn=partial(get_normalize_op, domain=True),  # 针对跨域场景使用全局统计归一化
        tile_z_dim=tile_z_dim,
        location_scale=location_scale
    )


def get_validation(modality, idx_pct=[0.7, 0.1, 0.2], tile_z_dim=3):
    """
    构造验证数据集，验证时不使用数据增强
    """
    return AbdominalDataset(
        idx_pct=idx_pct,
        mode='val',
        transforms=None,
        domains=modality,
        base_dir=BASEDIR,
        extern_norm_fn=partial(get_normalize_op, domain=False),
        tile_z_dim=tile_z_dim
    )


def get_test(modality, tile_z_dim=3, idx_pct=[0.7, 0.1, 0.2]):
    """
    构造测试数据集（源域测试）
    """
    return AbdominalDataset(
        idx_pct=idx_pct,
        mode='test',
        transforms=None,
        domains=modality,
        extern_norm_fn=partial(get_normalize_op, domain=False),
        base_dir=BASEDIR,
        tile_z_dim=tile_z_dim
    )


def get_test_all(modality, tile_z_dim=3, idx_pct=[0.7, 0.1, 0.2]):
    """
    构造全测试数据集（目标域测试），包含所有扫描数据
    """
    return AbdominalDataset(
        idx_pct=idx_pct,
        mode='test_all',
        transforms=None,
        domains=modality,
        extern_norm_fn=partial(get_normalize_op, domain=False),
        base_dir=BASEDIR,
        tile_z_dim=tile_z_dim
    )