# main.py
# 导入所需的标准库模块
import argparse      # 用于解析命令行参数
import os            # 操作系统接口（如文件、目录操作）
import sys           # 系统相关操作，如修改模块搜索路径
import datetime      # 日期时间处理
import importlib     # 动态导入模块



# 解决 Intel MKL 库重复加载时的冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'true'

import torch.optim  # PyTorch 优化器相关模块
from omegaconf import OmegaConf  # OmegaConf 用于加载和合并配置文件
from torch.utils.data import DataLoader  # 用于构造数据加载器
from engine import train_warm_up, evaluate, train_one_epoch_SBF, train_one_epoch
# 从 engine 模块导入训练、评估和单个 epoch 训练的函数

from ref.SLAug.losses import SetCriterion  # 导入自定义的损失函数
import numpy as np  # 数值计算库
import random      # 随机数生成
from torch.optim import lr_scheduler  # 学习率调度器

# 用于多进程数据加载时设置每个 worker 的随机种子
def worker_init_fn(worker_id):
    # 通过当前 numpy 随机数状态和 worker_id 来设置种子，保证每个 worker 的随机性不同
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# 用于设置全局随机种子，确保实验可重复
def seed_everything(seed=None):
    # 获取 uint32 的最大最小值，作为种子范围
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min
    try:
        # 如果未提供种子，则从环境变量中获取，若没有则随机生成一个种子
        if seed is None:
            seed = os.environ.get("PL_GLOBAL_SEED", random.randint(min_seed_value, max_seed_value))
        seed = int(seed)
    except (TypeError, ValueError):
        # 如果转换出错，则随机生成种子
        seed = random.randint(min_seed_value, max_seed_value)
    # 设置 Python、numpy、torch（CPU和GPU）的随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f'training seed is {seed}')
    return seed

# 构造命令行参数解析器，并添加相关参数
def get_parser(**parser_kwargs):
    # 内部函数，用于将字符串转换为布尔类型
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    # resume 参数：用于指定从日志目录或 checkpoint 恢复训练
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoints in logdir",
    )
    # base 参数：基础配置文件路径列表，可以有多个配置文件，后续参数可覆盖配置中的参数
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    # seed 参数：设置全局随机种子，默认 42
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="seed for seed_everything",
    )
    # postfix 参数：用于给实验名称添加后缀
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    return parser

# 从字符串描述中获取对象（例如从配置文件中指定的模块路径）
def get_obj_from_str(string, reload=False):
    # 将字符串按最后一个点分割成模块名和类名
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    # 返回该模块中的对应对象（如类或函数）
    return getattr(importlib.import_module(module, package=None), cls)

# 根据配置实例化对象
def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    # 根据配置中的 "target" 字段调用相应的对象构造函数，传入 "params" 参数（如果有的话）
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

# 用于根据配置构造数据模块的封装类，继承自 torch.nn.Module
class DataModuleFromConfig(torch.nn.Module):
    def __init__(self, batch_size, train=None, validation=None, test=None, num_workers=None):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        # 如果未指定 num_workers，则默认为 batch_size 的两倍
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        # 如果提供了 train、validation、test 数据集配置，则存储并绑定相应的 dataloader 方法
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader

    # 调用 instantiate_from_config 对所有数据集配置进行数据准备
    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    # 根据配置实例化数据集对象，并存储到 self.datasets 中
    def setup(self):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs
        )

    # 定义训练数据的 DataLoader
    def _train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True)

    # 定义验证数据的 DataLoader
    def _val_dataloader(self):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    # 定义测试数据的 DataLoader
    def _test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers)

# 设置 CuDNN 的 benchmark 模式，加速卷积计算（适用于固定输入尺寸）
torch.backends.cudnn.benchmark = True

# 程序主入口
if __name__ == "__main__":
    # 获取当前时间，用于生成实验名称
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    # 再次将当前工作目录添加到模块搜索路径中，确保可以导入本目录下的模块
    sys.path.append(os.getcwd())
    # 构造参数解析器
    parser = get_parser()
    # 解析已知的命令行参数，未识别的参数存储在 unknown 中
    opt, unknown = parser.parse_known_args()
    # 设置全局随机种子
    seed = seed_everything(opt.seed)
    # 如果指定了 resume 参数，则检查对应路径是否存在
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
    # 处理基础配置文件，若 opt.base 非空则生成配置文件名称后缀
    if opt.base:
        cfg_fname = os.path.split(opt.base[0])[-1]
        cfg_name = os.path.splitext(cfg_fname)[0]
        name = "_" + cfg_name
    else:
        name = None
        raise ValueError('no config')

    # 构造实验名称，包含时间、种子、配置文件名称后缀以及命令行后缀
    nowname = now + f'_seed{seed}' + name + opt.postfix
    # 定义日志目录、checkpoint 目录、配置目录以及可视化目录
    logdir = os.path.join("logs", nowname)
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    visdir = os.path.join(logdir, "visuals")
    # 创建上述目录，如果不存在则自动创建
    for d in [logdir, cfgdir, ckptdir, visdir]:
        os.makedirs(d, exist_ok=True)

    # 加载所有基础配置文件，并合并命令行中未识别的参数
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    # 将合并后的配置保存到配置目录中，文件名包含当前时间
    OmegaConf.save(config, os.path.join(cfgdir, "{}-project.yaml".format(now)))

    # 从配置中弹出模型、优化器和 SBF（saliency balancing fusion）部分
    model_config = config.pop("model", OmegaConf.create())
    optimizer_config = config.pop('optimizer', OmegaConf.create())
    SBF_config = config.pop('saliency_balancing_fusion', OmegaConf.create())

    # 根据模型配置实例化模型
    model = instantiate_from_config(model_config)
    if torch.cuda.is_available():
        model = model.cuda()

    # 根据模型配置判断使用哪种学习率设定：若存在 base_learning_rate，则根据 batch size 计算 lr
    if getattr(model_config.params, 'base_learning_rate'):
        bs, base_lr = config.data.params.batch_size, optimizer_config.base_learning_rate
        lr = bs * base_lr
    else:
        bs, lr = config.data.params.batch_size, optimizer_config.learning_rate

    # 根据模型是否为预训练，获取模型参数列表（包含不同 lr_scale 参数）
    if getattr(model_config.params, 'pretrain'):
        param_dicts = model.optim_parameters()
    else:
        param_dicts = [{"params": [p for n, p in model.named_parameters() if p.requires_grad], "lr_scale": 1}]

    # 构造优化器参数字典，并添加 momentum、weight_decay（如果在配置中有设置）
    opt_params = {'lr': lr}
    for k in ['momentum', 'weight_decay']:
        if k in optimizer_config:
            opt_params[k] = optimizer_config[k]

    # 实例化损失函数（SetCriterion）
    criterion = SetCriterion()

    print('optimization parameters: ', opt_params)
    # 根据优化器配置中的 target 字符串（例如 'torch.optim.SGD'）实例化优化器
    opt = eval(optimizer_config['target'])(param_dicts, **opt_params)

    # 如果使用 lambda 学习率调度器，则定义 lambda 函数计算 lr 衰减因子
    if optimizer_config.lr_scheduler == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + 0 - 50) / float(optimizer_config.max_epoch - 50 + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(opt, lr_lambda=lambda_rule)
    else:
        scheduler = None
        print('We follow the SSDG learning rate schedule by default, you can add your own schedule by yourself')
        raise NotImplementedError

    # 根据配置确定训练的最大迭代次数或最大 epoch 数
    assert optimizer_config.max_epoch > 0 or optimizer_config.max_iter > 0
    if optimizer_config.max_iter > 0:
        max_epoch = 999
        print('detect identified max iteration, set max_epoch to 999')
    else:
        max_epoch = optimizer_config.max_epoch

    # 实例化数据模块并准备数据
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    print(len(data.datasets["train"]))
    # 构造训练集 DataLoader，设置 persistent_workers、drop_last 和 pin_memory 以加速数据加载
    train_loader = DataLoader(data.datasets["train"], batch_size=data.batch_size,
                              num_workers=data.num_workers, shuffle=True, persistent_workers=True, drop_last=True, pin_memory=True)

    # 构造验证集 DataLoader，batch_size 与训练一致，num_workers 设为 1
    val_loader = DataLoader(data.datasets["validation"], batch_size=data.batch_size, num_workers=1)

    # 如果存在测试集，则构造测试集 DataLoader，并设置相关标志
    if data.datasets.get('test') is not None:
        test_loader = DataLoader(data.datasets["test"], batch_size=1, num_workers=1)
        best_test_dice = 0
        test_phase = True
    else:
        test_phase = False

    # 如果配置中设置了 warmup_iter 且大于 0，则执行预热训练
    if getattr(optimizer_config, 'warmup_iter'):
        if optimizer_config.warmup_iter > 0:
            train_warm_up(model, criterion, train_loader, opt, torch.device('cuda'), lr, optimizer_config.warmup_iter)
    cur_iter = 0
    best_dice = 0
    # 获取训练集中的标签名称（用于后续评估指标计算）
    label_name = data.datasets["train"].all_label_names
    # 开始训练循环
    for cur_epoch in range(max_epoch):
        # 根据 SBF 配置决定使用哪种单 epoch 训练方法
        if SBF_config.usage:
            cur_iter = train_one_epoch_SBF(model, criterion, train_loader, opt, torch.device('cuda'), cur_epoch, cur_iter, optimizer_config.max_iter, SBF_config, visdir)
        else:
            cur_iter = train_one_epoch(model, criterion, train_loader, opt, torch.device('cuda'), cur_epoch, cur_iter, optimizer_config.max_iter)
        # 更新学习率调度器
        if scheduler is not None:
            scheduler.step()

        # 每 100 个 epoch 进行验证，保存验证集上最佳模型
        if (cur_epoch + 1) % 100 == 0:
            cur_dice = evaluate(model, val_loader, torch.device('cuda'))
            if np.mean(cur_dice) > best_dice:
                best_dice = np.mean(cur_dice)
                # 删除之前保存的 val 模型 checkpoint
                for f in os.listdir(ckptdir):
                    if 'val' in f:
                        os.remove(os.path.join(ckptdir, f))
                # 保存当前 epoch 的模型 checkpoint
                torch.save({'model': model.state_dict()}, os.path.join(ckptdir, f'val_best_epoch_{cur_epoch}.pth'))

            # 输出当前 epoch 的各类别 DICE 指标及平均值
            str_log = f'Epoch [{cur_epoch}]   '
            for i, d in enumerate(cur_dice):
                str_log += f'Class {i}: {d}, '
            str_log += f'Validation DICE {np.mean(cur_dice)}/{best_dice}'
            print(str_log)

        # 每 50 个 epoch 保存最新的模型 checkpoint
        if (cur_epoch + 1) % 50 == 0:
            torch.save({'model': model.state_dict()}, os.path.join(ckptdir, 'latest.pth'))

        # 如果达到最大迭代次数，则保存最新模型并结束训练
        if cur_iter >= optimizer_config.max_iter and optimizer_config.max_iter > 0:
            torch.save({'model': model.state_dict()}, os.path.join(ckptdir, 'latest.pth'))
            print(f'End training with iteration {cur_iter}')
            break