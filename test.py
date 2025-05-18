#test.py test
import os
import sys

import distutils.version
# 将当前工作目录添加到 Python 模块搜索路径中，确保可以导入当前目录下的模块
sys.path.append(os.getcwd())

import argparse  # 用于解析命令行参数
from torch.utils.data import DataLoader  # PyTorch 中用于数据加载的 DataLoader 类
from torch.utils.tensorboard import SummaryWriter
import torchvision
import numpy as np  # 数组和数值计算库
import glob  # 用于文件路径匹配，查找符合特定模式的文件
from main import instantiate_from_config  # 从 main 模块中导入根据配置实例化模型或其他对象的函数
from main import Tee
import torch  # PyTorch 库，用于深度学习相关操作
from omegaconf import OmegaConf

def get_parser():
    """
    构造命令行参数解析器，并添加相应的参数选项
    """
    parser = argparse.ArgumentParser()
    # 参数 -r / --resume 用于指定恢复训练时加载的 checkpoint 或日志目录
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoints in logdir",
    )
    # 参数 -b / --base 用于指定基础配置文件的路径列表（可以多个），
    # 后续可通过命令行传入参数来覆盖配置中的部分参数
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    # 参数 -c / --config 用于指定单个配置文件路径，如果指定了该参数，则忽略基础配置文件（除非未指定）
    parser.add_argument(
        "-c",
        "--config",
        nargs="?",
        metavar="single_config.yaml",
        help="path to single config. If specified, base configs will be ignored "
             "(except for the last one if left unspecified).",
        const=True,
        default="",
    )
    # 参数 --ignore_base_data 用于在命令行中指定忽略基础配置中的数据设定，
    # 以便自定义数据集配置
    parser.add_argument(
        "--ignore_base_data",
        action="store_true",
        help="Ignore data specification from base configs. Useful if you want "
             "to specify a custom datasets on the command line.",
    )
    return parser

if __name__ == "__main__":
    # 再次将当前工作目录添加到模块搜索路径中，确保可以导入本目录下的模块
    sys.path.append(os.getcwd())
    # 构造参数解析器
    parser = get_parser()
    # 解析已知的命令行参数，其余未识别的参数存放在 unknown 中
    opt, unknown = parser.parse_known_args()

    ckpt = None  # 初始化 checkpoint 变量
    if opt.resume:
        # 如果指定了 resume 参数，则检查路径是否存在
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        # 如果 resume 指向一个文件（checkpoint 文件）
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            try:
                # 从路径中提取日志目录部分，此处寻找目录中包含 "logs" 的部分
                idx = len(paths) - paths[::-1].index("logs") + 1
            except ValueError:
                # 若未找到 "logs"，猜测使用倒数第二个目录
                idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
            # 组合成日志目录路径
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume
        else:
            # 如果 resume 指向一个目录，则必须为日志目录
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            # 在日志目录下的 checkpoints 文件夹中查找包含 "latest" 的文件，作为最新的 checkpoint
            for f in os.listdir(os.path.join(logdir, "checkpoints")):
                if 'latest' in f:
                    ckpt = os.path.join(logdir, "checkpoints", f)
        print(f"logdir:{logdir}")
        # 在日志目录下搜索所有基础配置文件，排序后添加到 opt.base 列表中
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*-project.yaml")))
        opt.base = base_configs + opt.base

    # 如果通过 -c 指定了单个配置文件
    if opt.config:
        if type(opt.config) == str:
            opt.base = [opt.config]
        else:
            # 如果 opt.config 不是字符串，则取 opt.base 列表中最后一个元素作为配置文件
            opt.base = [opt.base[-1]]

    # 使用 OmegaConf 加载所有基础配置文件（需先导入 OmegaConf，此处假设已经导入）
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    # 将命令行未识别的参数转为 OmegaConf 对象（点表示法）
    cli = OmegaConf.from_dotlist(unknown)
    # 合并基础配置和命令行参数，命令行参数具有更高优先级
    config = OmegaConf.merge(*configs, cli)
    # 从合并后的配置中弹出 "model" 部分，如果不存在则创建一个空配置
    model_config = config.pop("model", OmegaConf.create())
    print(model_config)

    gpu = True
    eval_mode = True
    show_config = False
    # 根据 model 配置实例化模型
    model = instantiate_from_config(model_config)
    # 加载 checkpoint 文件，映射到 CPU 上（后续再转到 GPU）
    pl_sd = torch.load(ckpt, map_location="cpu")
    # 加载 checkpoint 中的模型参数，允许部分不匹配（strict=False）
    model.load_state_dict(pl_sd['model'], strict=False)
    # 将模型转移到 GPU，并设置为评估模式
    model.cuda().eval()

    # 根据配置实例化数据模块（包含数据集和数据加载逻辑）
    data = instantiate_from_config(config.data)
    data.prepare_data()  # 数据准备工作，例如下载或预处理
    data.setup()  # 根据当前阶段（训练/验证/测试）进行数据集分配和初始化
    # 构造验证集 DataLoader，batch_size=1, num_workers=1
    val_loader = DataLoader(data.datasets["validation"], batch_size=1, num_workers=1)
    # 构造测试集 DataLoader，batch_size=1, num_workers=1
    test_loader = DataLoader(data.datasets["test"], batch_size=1, num_workers=1)

    from engine import prediction_wrapper  # 导入预测封装函数，用于模型预测

    # 获取训练集中的标签名称
    label_name = data.datasets["train"].all_label_names
    # 调用 prediction_wrapper 进行预测，传入模型、测试集加载器、以及标签名称，
    # 同时保存预测结果。函数返回预测列表、DSC 评估表、错误信息字典及域名称列表
    out_prediction_list, dsc_table, error_dict, domain_names = prediction_wrapper(
        model, test_loader, 0, label_name, save_prediction=True
    )




def test_and_log(models_info,       # dict: { "model_name": (model_obj, ckpt_path), ... }
                 test_dataset,
                 device="cuda",
                 base_logdir="runs/seg_vis"):

    for model_name, (model, ckpt_path) in models_info.items():
        # 1. 加载模型权重
        model.to(device).eval()
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        
        # 2. 为每个模型创建单独的 writer
        logdir = f"{base_logdir}/{model_name}"
        writer = SummaryWriter(log_dir=logdir)
        
        # 3. 随便选几张 test 样本来可视化
        for step, (img, gt) in enumerate(test_dataset):
            if step >= 5:          # 最多可视化前5个样本
                break

            img = img.to(device).unsqueeze(0)    # [1, C, H, W]
            with torch.no_grad():
                logits = model(img)               # [1, C, H, W]
                pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)  # [1, H, W]

            # 把原图、GT、Pred 都转到 CPU、numpy 格式
            img0  = img[0].cpu()                   # [C, H, W]
            gt0   = gt.unsqueeze(0).cpu()          # [1, H, W]
            pred0 = pred.unsqueeze(0).cpu()        # [1, H, W]

            # 拼成一行三列的 grid（也可以自定义 nrow）
            grid = torchvision.utils.make_grid(
                [img0, 
                 torch.cat([img0]*3)[0:1],  # 如果想overlay，可以先把原图多份放着
                 torch.cat([img0]*3)[0:1]],
                nrow=3,
                normalize=True,
                scale_each=True
            )
            # 写入原图 Grid
            writer.add_image(f"{model_name}/image_grid", grid, global_step=step)
            # 单独写入 GT 和 Pred
            writer.add_image(f"{model_name}/GT_mask", gt0,  step)
            writer.add_image(f"{model_name}/Pred_mask", pred0, step)

        writer.close()
