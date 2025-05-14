# engine.py
# 导入所需模块
from typing import Iterable  # 用于类型提示，表示可迭代对象
import os  # 操作系统接口，用于路径和文件操作
import matplotlib.pyplot as plt  # 绘图库，用于可视化输出
import numpy as np  # 数值计算库
import torch  # PyTorch，用于深度学习模型和张量操作
import utils.misc as utils  # 自定义的辅助工具模块（例如日志记录、进度条等）
import functools  # 函数工具库，用于函数包装、部分函数应用等
from tqdm import tqdm  # 进度条工具，用于显示迭代进度
import torch.nn.functional as F  # PyTorch中常用的函数接口，例如one_hot等
from monai.metrics import compute_meandice  # MONAI中计算平均Dice系数的函数
from torch.autograd import Variable  # 用于包装变量，使其支持自动求导
from data.saliency_balancing_fusion import get_SBF_map  # 导入获取SBF（Saliency Balancing Fusion）图的函数
from losses.ph_loss import PHLoss
from metrics import dice as dice_metric, ter as ter_metric

# 重定义print函数，自动flush标准输出，确保输出不被缓冲
print = functools.partial(print, flush=True)


# ------------------------- 训练预热阶段 -------------------------
def train_warm_up(model: torch.nn.Module, criterion: torch.nn.Module,
                  data_loader: Iterable, optimizer: torch.optim.Optimizer,
                  device: torch.device, learning_rate: float, warmup_iteration: int = 1500):
    """
    训练预热函数，用于在正式训练前对模型进行预热，以逐步增加学习率。

    参数：
        model: 训练的模型（torch.nn.Module）
        criterion: 损失函数模块
        data_loader: 训练数据的迭代器
        optimizer: 优化器
        device: 设备（如GPU）
        learning_rate: 基础学习率
        warmup_iteration: 预热总迭代次数，默认1500

    过程：
        1. 将模型和损失函数设置为训练模式。
        2. 使用MetricLogger记录指标（如损失和当前学习率）。
        3. 循环遍历数据，在每个iteration中更新学习率（按当前迭代数比例上升），计算损失并反向传播更新模型。
        4. 当迭代次数达到预热设定值时退出。
    """
    model.train()
    criterion.train()

    # 使用自定义MetricLogger记录指标信息
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    print_freq = 10  # 设置打印频率，每10个iteration打印一次
    cur_iteration = 0
    while True:
        # 遍历数据加载器，并使用tqdm显示进度
        for i, samples in enumerate(metric_logger.log_every(data_loader, print_freq,                                                         'WarmUp with max iteration: {}'.format(warmup_iteration))):
            visual_dict = None
            # 将样本中所有张量移动到指定设备
            for k, v in samples.items():
                if isinstance(v, torch.Tensor):
                    samples[k] = v.to(device)
            cur_iteration += 1

            # 根据当前迭代数更新每个参数组的学习率：线性增加
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = cur_iteration / warmup_iteration * learning_rate * param_group["lr_scale"]

            img = samples['images']
            lbl = samples['labels']
            # 前向传播计算预测结果
            pred = model(img)
            # 计算损失字典，通常包含多个损失项
            loss_dict = criterion.get_loss(pred, lbl)
            # 将所有损失按权重求和得到总损失
            losses = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys())
            # 多类分割 — softmax；二值分割 — sigmoid（二选一）


            optimizer.zero_grad()  # 清除梯度
            losses.backward()  # 反向传播计算梯度
            optimizer.step()  # 优化器更新参数

            # 更新指标记录器
            metric_logger.update(**loss_dict)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            # 达到预热迭代次数后退出
            if cur_iteration >= warmup_iteration:
                print(f'WarnUp End with Iteration {cur_iteration} and current lr is {optimizer.param_groups[0]["lr"]}.')
                return cur_iteration
        # 同步各进程之间的指标信息（适用于分布式训练）
        metric_logger.synchronize_between_processes()


# ------------------------- 单个Epoch训练（常规版） -------------------------
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, cur_iteration: int, 
                    max_iteration: int = -1, loss_config=None, grad_scaler=None):
    """
    单个Epoch的训练过程（常规版）。

    参数：
        model: 训练模型
        criterion: 损失函数
        data_loader: 训练数据加载器
        optimizer: 优化器
        device: 设备（如GPU）
        epoch: 当前Epoch编号
        cur_iteration: 当前已训练的迭代次数
        max_iteration: 最大迭代次数（若设置 > 0，则训练到该次数后停止）
        grad_scaler: 混合精度训练下的梯度缩放器（可选）

    过程：
        遍历数据集，执行前向传播、计算损失、反向传播和参数更新，
        同时更新并记录损失和学习率等指标，若达到最大迭代次数则中断训练。
    """
    model.train()
    criterion.train()
    
   # 如果启用了拓扑损失，则创建实例
    if loss_config and loss_config.topo.enabled:
        ph_criterion = PHLoss(threshold=None).to(device)
        topo_weight = loss_config.topo.weight
    else:
        ph_criterion = None
        
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('topo', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10



    # 遍历数据加载器，并记录进度
    for i, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        img = samples['images'].to(device)
        lbl = samples['labels'].to(device)
    
        # 最好在拿到初始 batch、算出 pred 和 prob/gt 之后做一次检查
        pred = model(img)
        prob_fg, gt_fg = get_ph_inputs(pred, lbl)
        if i == 0 and epoch == 0:
            print("DEBUG → gt_fg.sum =", gt_fg.sum().item(),
                  "prob_fg.min/max =", prob_fg.min().item(), prob_fg.max().item())
            
        # 再把 samples 里的所有 tensor 都 .to(device)
        for k, v in samples.items():
            if isinstance(v, torch.Tensor):
                samples[k] = v.to(device)
        loss_dict = criterion.get_loss(pred, lbl)
        losses = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict)
        # 判断是否使用混合精度训练
        if grad_scaler is None:            
            if ph_criterion is not None and gt_fg.sum() > 0:     # 有前景才算拓扑
                topo_loss = ph_criterion(prob_fg, gt_fg)
                losses = losses + topo_weight * topo_loss
                loss_dict['topo'] = topo_loss.item()
            else:
                loss_dict['topo'] = 0.0
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        else:
            # 混合精度训练环境下
            with torch.cuda.amp.autocast():                
                if ph_criterion is not None and gt_fg.sum() > 0:     # 有前景才算拓扑
                    topo_loss = ph_criterion(prob_fg, gt_fg)
                    losses = losses + topo_weight * topo_loss
                    loss_dict['topo'] = topo_loss.item()
                else:
                    loss_dict['topo'] = 0.0
            optimizer.zero_grad()
            grad_scaler.scale(losses).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

        # 更新日志记录器
        metric_logger.update(**loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        cur_iteration += 1
        if cur_iteration >= max_iteration and max_iteration > 0:
            break

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return cur_iteration


# ------------------------- 单个Epoch训练（SBF版） -------------------------
def train_one_epoch_SBF(model: torch.nn.Module, criterion: torch.nn.Module,
                        data_loader: Iterable, optimizer: torch.optim.Optimizer,
                        device: torch.device, epoch: int, cur_iteration: int, max_iteration: int = -1, config=None,
                        visdir=None, loss_config=None):
    """
    单个Epoch训练过程（针对Saliency Balancing Fusion, SBF版）。

    参数：
        model: 训练模型
        criterion: 损失函数
        data_loader: 数据加载器
        optimizer: 优化器
        device: 设备（如GPU）
        epoch: 当前Epoch编号
        cur_iteration: 当前已训练的迭代次数
        max_iteration: 最大迭代次数
        config: SBF相关配置参数
        visdir: 可视化结果保存目录（可选）

    过程：
        1. 训练模式下，遍历数据加载器，获取图像及增强图像（GLA, LLA）和标签；
        2. 每隔一定迭代次数保存可视化结果；
        3. 计算标准前向传播损失并反向传播，获得梯度信息计算 saliency map；
        4. 根据 saliency map 混合 GLA 与 LLA，重新计算增强后的损失，再次反向传播并更新参数；
        5. 更新各项指标记录，并定期保存可视化图像；
        6. 当达到最大迭代次数时退出训练。
    """
    # 构造
    if loss_config and loss_config.topo.enabled:
        ph_criterion = PHLoss(threshold=None).to(device)
        topo_weight  = loss_config.topo.weight
    else:
        ph_criterion = None
        topo_weight  = 0.0
        
    model.train()
    criterion.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('topo', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    visual_freq = 500  # 可视化频率：每500次迭代保存一次图像
    for i, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        visual_dict = None
        # 1) 先把所有 tensor 都送上 GPU
        GLA_img = samples['images'].to(device)
        LLA_img = samples['aug_images'].to(device)
        lbl     = samples['labels'].to(device)
    
        # 2) 清空梯度
        optimizer.zero_grad()
    
        # 3) 正向、算 topo loss
        input_var = Variable(GLA_img, requires_grad=True)
        logits    = model(input_var)               # (B,1,H,W) or (B,C,H,W)
        prob_fg, gt_fg = get_ph_inputs(logits, lbl)
    
        # debug
        if i == 0 and epoch == 0:
            print("DEBUG → gt_fg.sum=", gt_fg.sum().item(),
                  "prob_fg.min/max=", prob_fg.min().item(), prob_fg.max().item())
    
        # 标准 loss
        loss_dict = criterion.get_loss(logits, lbl)
        losses    = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict)
    
        # topo loss
        if ph_criterion is not None:
            topo_loss = ph_criterion(prob_fg, gt_fg)
            losses   += topo_weight * topo_loss
            loss_dict['topo'] = topo_loss.item()
        else:
            loss_dict['topo'] = 0.0
    
        # 第一次 backward，只保留计算 saliency 的 graph
        losses.backward(retain_graph=True)
    
        # 4) 构造 saliency map
        gradient = torch.sqrt(torch.mean(input_var.grad**2, dim=1, keepdim=True)).detach()
        saliency = get_SBF_map(gradient, config.grid_size)
    
        # 5) 混合图像，再 forward/backward 增强 loss
        mixed_img = GLA_img * saliency + LLA_img * (1 - saliency)
        aug_var   = Variable(mixed_img, requires_grad=True)
        aug_logits= model(aug_var)
        aug_loss_dict = criterion.get_loss(aug_logits, lbl)
        aug_losses    = sum(
            aug_loss_dict[k] * criterion.weight_dict[k]
            for k in aug_loss_dict if k in criterion.weight_dict
        )
        aug_losses.backward()
    
        # 6) 最后一步，更新参数
        optimizer.step()
    
        # 7) 把所有 loss（原始 + topo + _aug）打给 logger
        all_loss_dict = {}
        for k in loss_dict:
            if k in criterion.weight_dict:
                all_loss_dict[k] = loss_dict[k]
                all_loss_dict[k + '_aug'] = aug_loss_dict.get(k, 0.0)
        all_loss_dict['topo'] = loss_dict['topo']
    
        metric_logger.update(**all_loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])
    
        # … cur_iteration / 可视化 / break 等照旧
        if cur_iteration >= max_iteration and max_iteration > 0:
            break

        # 保存可视化图像
        if visdir is not None and visual_dict is not None:
            fs = int(len(visual_dict)**0.5) + 1
            for idx, k in enumerate(visual_dict.keys()):
                plt.subplot(fs, fs, idx + 1)
                plt.title(k)
                plt.axis('off')
                if k not in ['GT', 'GLA_pred', 'SBF_pred']:
                    plt.imshow(visual_dict[k], cmap='gray')
                else:
                    plt.imshow(visual_dict[k], vmin=0, vmax=4)
            plt.tight_layout()
            plt.savefig(f'{visdir}/{cur_iteration}.png')
            plt.close()
        cur_iteration += 1

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return cur_iteration


# ------------------------- 模型评估函数 -------------------------
@torch.no_grad()
def _evaluate_perclass(model: torch.nn.Module, data_loader: Iterable, device: torch.device):    """
    评估模型在验证/测试集上的分割性能，计算每个类别的平均Dice系数。

    参数：
        model: 待评估模型
        data_loader: 数据加载器
        device: 设备（如GPU）

    过程：
        遍历所有样本，模型预测后将预测结果转换为 one-hot 编码，
        与 ground truth 进行对比，利用 MONAI 的 compute_meandice 计算 Dice 系数，
        最后求各类别Dice的均值作为评估结果。
    """
    model.eval()

    def convert_to_one_hot(tensor, num_c):
        return F.one_hot(tensor, num_c).permute((0, 3, 1, 2))

    dices = []
    for samples in data_loader:
        for k, v in samples.items():
            if isinstance(v, torch.Tensor):
                samples[k] = v.to(device)
        img = samples['images']
        lbl = samples['labels']
        logits = model(img)
        num_classes = logits.size(1)
        pred = torch.argmax(logits, dim=1)
        one_hot_pred = convert_to_one_hot(pred, num_classes)
        one_hot_gt = convert_to_one_hot(lbl, num_classes)
        dice = compute_meandice(one_hot_pred, one_hot_gt, include_background=False)
        dices.append(dice.cpu().numpy())
    dices = np.concatenate(dices, 0)
    dices = np.nanmean(dices, 0)
    return dices

@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    """
    包装函数 —— 同时返回:
        dices_array (旧接口)、
        mean_dice   (标量)、
        mean_ter    (标量, λ 不参与)
    """
    # 先拿到按类别 Dice（旧逻辑）
    dices = _evaluate_perclass(model, data_loader, device)

    # 重新跑一次，算 mean-Dice + TER
    model.eval()
    dice_list, ter_list = [], []
    for samples in data_loader:
        for k, v in samples.items():
            if isinstance(v, torch.Tensor):
                samples[k] = v.to(device)
        img = samples['images']
        lbl = samples['labels']
        logits = model(img)
        prob = torch.sigmoid(logits) if logits.shape[1] == 1 else torch.softmax(logits, dim=1)
        dice_list.append(dice_metric(prob, lbl))
        ter_list .append(ter_metric (prob, lbl))

    mean_dice = sum(dice_list) / len(dice_list)
    mean_ter  = sum(ter_list ) / len(ter_list)

    print(f"[EVAL] mean-Dice={mean_dice:.4f}  mean-TER={mean_ter:.4f}")
    return dices, mean_dice, mean_ter      # ← 关键：三个值

# ------------------------- 预测封装函数 -------------------------
def prediction_wrapper(model, test_loader, epoch, label_name, mode='base', save_prediction=False):
    """
    用于评估和整理预测结果的封装函数。

    参数：
        model: 待评估的网络模型
        test_loader: 测试数据加载器
        epoch: 当前测试的 epoch 编号
        label_name: 各类别标签名称列表
        mode: 标记当前测试模式，用于保存结果时备注
        save_prediction: 是否保存预测结果（默认为False，节省内存）

    过程：
        遍历测试集，每个扫描（batch中 is_start 为True时开始）收集所有切片的预测结果与ground truth，
        最后调用 eval_list_wrapper 进行评估，计算Dice系数，并返回预测结果、Dice表、错误信息字典和域名称列表。
    """
    model.eval()
    with torch.no_grad():
        out_prediction_list = {}  # 用于存储每个扫描的预测结果
        for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            if batch['is_start']:
                slice_idx = 0
                scan_id_full = str(batch['scan_id'][0])
                out_prediction_list[scan_id_full] = {}
                nframe = batch['nframe']
                nb, nc, nx, ny = batch['images'].shape
                # 初始化当前扫描的预测和 ground truth 张量（存储每个切片的分割结果）
                curr_pred = torch.Tensor(np.zeros([nframe, nx, ny])).cuda()
                curr_gth = torch.Tensor(np.zeros([nframe, nx, ny])).cuda()
                curr_img = np.zeros([nx, ny, nframe])
            # 强制 batch size 为1
            assert batch['labels'].shape[0] == 1

            img = batch['images'].cuda()
            gth = batch['labels'].cuda()

            pred = model(img)
            pred = torch.argmax(pred, 1)
            curr_pred[slice_idx, ...] = pred[0, ...]  # 存储预测结果
            curr_gth[slice_idx, ...] = gth[0, ...]  # 存储 ground truth
            curr_img[:, :, slice_idx] = batch['images'][0, 0, ...].numpy()
            slice_idx += 1

            if batch['is_end']:
                out_prediction_list[scan_id_full]['pred'] = curr_pred
                out_prediction_list[scan_id_full]['gth'] = curr_gth

        print("Epoch {} test result on mode {} segmentation are shown as follows:".format(epoch, mode))
        error_dict, dsc_table, domain_names = eval_list_wrapper(out_prediction_list, len(label_name), label_name)
        error_dict["mode"] = mode
        if not save_prediction:  # 如果不需要保存预测结果，则释放内存
            del out_prediction_list
            out_prediction_list = []
        torch.cuda.empty_cache()

    return out_prediction_list, dsc_table, error_dict, domain_names


# ------------------------- 评估整理函数 -------------------------
def eval_list_wrapper(vol_list, nclass, label_name):
    """
    对预测结果进行评估和整理，计算各类别的Dice系数，并按域进行统计。

    参数：
        vol_list: 字典，存储每个扫描的预测结果和ground truth
        nclass: 类别数
        label_name: 各类别的名称列表
    过程：
        1. 将预测和ground truth转换为 one-hot 编码；
        2. 调用 MONAI 的 compute_meandice 计算每个扫描的Dice系数；
        3. 按域统计各扫描的Dice系数，并计算均值和标准差；
        4. 输出各类别及整体的Dice系数，并返回错误信息字典、Dice表和域名称列表。
    """

    def convert_to_one_hot2(tensor, num_c):
        return F.one_hot(tensor.long(), num_c).permute((3, 0, 1, 2)).unsqueeze(0)

    out_count = len(vol_list)
    tables_by_domain = {}  # 按域存储结果的字典
    dsc_table = np.ones([out_count, nclass])  # 每一行对应一个扫描，每一列对应一个类别
    idx = 0
    for scan_id, comp in vol_list.items():
        domain, pid = scan_id.split("_")
        if domain not in tables_by_domain.keys():
            tables_by_domain[domain] = {'scores': [], 'scan_ids': []}
        pred_ = comp['pred']
        gth_ = comp['gth']
        dices = compute_meandice(y_pred=convert_to_one_hot2(pred_, nclass),
                                 y=convert_to_one_hot2(gth_, nclass),
                                 include_background=True).cpu().numpy()[0].tolist()
        tables_by_domain[domain]['scores'].append(dices)
        tables_by_domain[domain]['scan_ids'].append(scan_id)
        dsc_table[idx, ...] = np.reshape(dices, (-1))
        del pred_
        del gth_
        idx += 1
        torch.cuda.empty_cache()

    # 输出每个器官的Dice均值和标准差
    error_dict = {}
    for organ in range(nclass):
        mean_dc = np.mean(dsc_table[:, organ])
        std_dc = np.std(dsc_table[:, organ])
        print("Organ {} with dice: mean: {:06.5f}, std: {:06.5f}".format(label_name[organ], mean_dc, std_dc))
        error_dict[label_name[organ]] = mean_dc
    print("Overall std dice by sample {:06.5f}".format(dsc_table[:, 1:].std()))
    print("Overall mean dice by sample {:06.5f}".format(dsc_table[:, 1:].mean()))
    error_dict['overall'] = dsc_table[:, 1:].mean()

    # 按域统计平均Dice
    overall_by_domain = []
    domain_names = []
    for domain_name, domain_dict in tables_by_domain.items():
        domain_scores = np.array(domain_dict['scores'])
        domain_mean_score = np.mean(domain_scores[:, 1:])  # 忽略背景
        error_dict[f'domain_{domain_name}_overall'] = domain_mean_score
        error_dict[f'domain_{domain_name}_table'] = domain_scores
        overall_by_domain.append(domain_mean_score)
        domain_names.append(domain_name)
    print('per domain resutls:', overall_by_domain)
    error_dict['overall_by_domain'] = np.mean(overall_by_domain)
    print("Overall mean dice by domain {:06.5f}".format(error_dict['overall_by_domain']))
    return error_dict, dsc_table, domain_names


# ------------------------------------TOOLS
EPS = 1e-6                    # 防 0

def safe_prob(t: torch.Tensor):
    """把概率图限制在 (0,1) 并保证 sum>0；保持原 4D 形状"""
    t = t.clamp(min=EPS)      # 全 0 时变为极小正数
    s = t.sum(dim=(2, 3), keepdim=True) + EPS
    return t / s
    

def get_ph_inputs(logits, lbl):
    if logits.shape[1] == 1:
        prob_fg = torch.sigmoid(logits)           # (B,1,H,W), 0~1 连续
        gt_fg   = lbl.float().unsqueeze(1)        # 0/1 二值
    else:
        prob_all = torch.softmax(logits, dim=1)
        prob_fg  = 1.0 - prob_all[:, :1]
        gt_fg    = (lbl > 0).float().unsqueeze(1) 

    return prob_fg, gt_fg