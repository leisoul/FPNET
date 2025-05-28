import numpy as np
import importlib
import torch

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

_metrics = [
    importlib.import_module(f'utils.metrics')
]


def create_metrics(opt):
    """Create metrics based on configuration.

    Args:
        opt (dict): Configuration. It contains:
            metric_types (list): List of metric types.
        _metrics (list): List of metric classes.
    """
    metric_types = opt['val']['metrics']
    created_metrics = []
    for metric_type in metric_types:

        for metric in _metrics:
            metric_cls = getattr(metric, metric_type, None)
            break

        if metric_cls is None:
            raise ValueError(f'Metric {metric_type} is not found.')

        metric = metric_cls()
        created_metrics.append(metric)


    return created_metrics


class PSNR:
    def __init__(self, average=True, data_range=1):
        self.average = average
        self.data_range = data_range
    def __call__(self, img1_list, img2_list):

        psnr_values = [
            peak_signal_noise_ratio(
                im1.detach().cpu().numpy().astype(np.float32),  # 将 img1 转换为 numpy float32
                im2.detach().cpu().numpy().astype(np.float32),  # 将 img2 转换为 numpy float32
                data_range=self.data_range
            ) 
            for im1, im2 in zip(img1_list, img2_list)
        ]
        return sum(psnr_values) / len(psnr_values) if len(psnr_values) > 0 else 0 


class SSIM:
    def __init__(self, data_range=1, win_size=3):
        self.data_range = data_range
        self.win_size = win_size

    def __call__(self, img1_list, img2_list):
        ssim_values = []
        # 遍历每对图
        if len(img1_list) == 3:
            im1_np = img1_list.permute(1, 2, 0).cpu().numpy().astype(np.float32)
            im2_np = img2_list.permute(1, 2, 0).cpu().numpy().astype(np.float32)

            # 计算 SSIM
            ssim_value = structural_similarity(
                im1_np,
                im2_np,
                data_range=self.data_range,
                win_size=self.win_size,
                multichannel=True  # 处理多通道图像
            )
            ssim_values.append(ssim_value)

        else:
            for id in range(len(img1_list)):
                im1_np = img1_list[id].permute(1, 2, 0).cpu().numpy().astype(np.float32)
                im2_np = img2_list[id].permute(1, 2, 0).cpu().numpy().astype(np.float32)

                # 计算 SSIM
                ssim_value = structural_similarity(
                    im1_np,
                    im2_np,
                    data_range=self.data_range,
                    win_size=self.win_size,
                    multichannel=True  # 处理多通道图像
                )
                ssim_values.append(ssim_value)
        # 返回平均 SSIM 值
        return sum(ssim_values) / len(ssim_values) if ssim_values else 0



def mIOU(predictions, targets, num_classes):
    if predictions.shape[1] != 1:  # (N, C, H, W)
        predictions = torch.argmax(predictions, dim=1)  # 取最大概率的类别作为预测标签

    else:
        predictions = torch.sigmoid(predictions)
        predictions = (predictions > 0.5).float()

    # 初始化 IoU 统计量
    ious = []

    for cls in range(1, num_classes + 1):
        # 获取预测和真实标签中的当前类别的区域
        pred_cls = (predictions == cls)
        target_cls = (targets == cls)

        # 计算交集和并集
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()

        # 计算 IoU
        if union == 0:
            iou = 0.0  # 如果并集为零，则 IoU 设为 NaN
        else:
            iou = intersection / union

        ious.append(iou)

    # 计算 mIoU
    mIoU = np.nanmean(ious)  # 忽略 NaN 值，计算平均 IoU

    return mIoU


def f1_score(predictions, targets):
    """
    计算F1 score.

    参数:
        predictions (np.ndarray): 二进制预测值。
        targets (np.ndarray): 二进制真实标签。

    返回:
        float: F1 score.
    """
    predictions = predictions.astype(np.bool)
    targets = targets.astype(np.bool)

    TP = np.logical_and(predictions, targets).sum()
    FP = np.logical_and(predictions, np.logical_not(targets)).sum()
    FN = np.logical_and(np.logical_not(predictions), targets).sum()

    if TP + FP == 0 or TP + FN == 0:
        return 0.0

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1
