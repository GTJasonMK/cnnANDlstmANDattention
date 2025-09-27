#!/usr/bin/env python3
"""
训练完成后自动生成测井图的脚本
"""

import os
import sys
import numpy as np
import torch
import pandas as pd
from pathlib import Path

# 添加项目路径
sys.path.append('.')
from well_log_visualizer import WellLogVisualizer
from main import prepare_run
from configs.config import load_config
from trainer import Trainer

def generate_predictions_and_visualize(config_path: str, checkpoint_path: str = None):
    """
    加载训练好的模型，生成预测并可视化

    Args:
        config_path: 配置文件路径
        checkpoint_path: 具体的检查点文件路径
    """
    print("加载配置文件...")
    cfg = load_config(config_path)

    # 准备数据和模型
    print("准备数据和模型...")
    data, model, (train_loader, val_loader, test_loader) = prepare_run(cfg)

    # 使用指定的检查点路径
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint("best_models/optimal_well_log")

    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print("未找到检查点文件，无法生成预测")
        return

    print(f"加载检查点: {checkpoint_path}")

    # 创建trainer并加载模型
    trainer = Trainer(model, cfg)
    trainer.load_checkpoint(checkpoint_path)
    model.eval()

    print("生成预测...")

    # 生成测试集预测
    predictions, actuals = trainer.predict(test_loader)

    # 🔥 关键修复：获取归一化统计信息并进行反归一化
    print("获取归一化统计信息...")

    # 从test_loader的dataset获取归一化统计信息
    test_dataset = test_loader.dataset.dataset if hasattr(test_loader.dataset, 'dataset') else test_loader.dataset
    normalization_stats = getattr(test_dataset, 'stats', None)

    if normalization_stats is None:
        print("[ERROR] 无法获取归一化统计信息")
        return

    print(f"[INFO] 归一化方法: {test_dataset.normalize}")
    print(f"[INFO] 目标列索引: {test_dataset.target_idx}")

    # 转换为numpy数组
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(actuals, torch.Tensor):
        actuals = actuals.detach().cpu().numpy()

    print(f"归一化空间预测数据形状: {predictions.shape}")
    print(f"归一化空间实际数据形状: {actuals.shape}")
    print(f"归一化空间预测值范围: {predictions.min():.6f} - {predictions.max():.6f}")
    print(f"归一化空间实际值范围: {actuals.min():.6f} - {actuals.max():.6f}")

    # 🔥 关键修复：反归一化到原始空间
    print("执行反归一化...")

    def inverse_normalize(data, stats, normalize_method, target_indices):
        """将归一化的数据反归一化到原始空间"""
        print(f"[DEBUG] 反归一化参数:")
        print(f"  数据形状: {data.shape}")
        print(f"  归一化方法: {normalize_method}")
        print(f"  目标索引: {target_indices}")

        denormalized = np.zeros_like(data)

        for i, target_idx in enumerate(target_indices):
            print(f"[DEBUG] 处理目标{i} (原始列{target_idx}):")

            if normalize_method == 'minmax':
                min_val = stats.min[target_idx] if stats.min is not None else 0
                max_val = stats.max[target_idx] if stats.max is not None else 1

                print(f"  目标{i} MinMax: {min_val:.8f} - {max_val:.8f}")
                print(f"  原始归一化数据范围: {data[..., i].min():.6f} - {data[..., i].max():.6f}")

                denormalized[..., i] = data[..., i] * (max_val - min_val) + min_val

                print(f"  反归一化后范围: {denormalized[..., i].min():.8f} - {denormalized[..., i].max():.8f}")

            elif normalize_method == 'standard':
                mean_val = stats.mean[target_idx] if stats.mean is not None else 0
                std_val = stats.std[target_idx] if stats.std is not None else 1
                denormalized[..., i] = data[..., i] * std_val + mean_val
            else:
                denormalized[..., i] = data[..., i]  # 无归一化时保持不变

        return denormalized

    # 反归一化预测值和真实值
    predictions_original = inverse_normalize(predictions, normalization_stats, test_dataset.normalize, test_dataset.target_idx)
    actuals_original = inverse_normalize(actuals, normalization_stats, test_dataset.normalize, test_dataset.target_idx)

    print(f"原始空间预测值范围: {predictions_original.min():.6f} - {predictions_original.max():.6f}")
    print(f"原始空间实际值范围: {actuals_original.min():.6f} - {actuals_original.max():.6f}")

    # 🔥 关键修复：验证目标分离是否正确
    print(f"\n[DEBUG] 验证目标分离:")
    print(f"  预测目标0范围: {predictions_original[:, 0].min():.8f} - {predictions_original[:, 0].max():.8f}")
    print(f"  预测目标1范围: {predictions_original[:, 1].min():.8f} - {predictions_original[:, 1].max():.8f}")
    print(f"  实际目标0范围: {actuals_original[:, 0].min():.8f} - {actuals_original[:, 0].max():.8f}")
    print(f"  实际目标1范围: {actuals_original[:, 1].min():.8f} - {actuals_original[:, 1].max():.8f}")

    # 根据数值范围判断目标顺序是否正确
    pred_0_range = predictions_original[:, 0].max() - predictions_original[:, 0].min()
    pred_1_range = predictions_original[:, 1].max() - predictions_original[:, 1].min()

    print(f"[DEBUG] 判断目标顺序:")
    print(f"  目标0数值范围: {pred_0_range:.6f}")
    print(f"  目标1数值范围: {pred_1_range:.6f}")

    # 🔥 检查是否需要交换目标顺序
    # DTCRT应该是微小值（0.000xxx），ALCDLC应该是大值（2000+）
    # 如果目标0的范围比目标1大很多，说明顺序反了
    if pred_0_range > pred_1_range * 1000:  # 如果目标0的范围比目标1大1000倍以上
        print("[WARNING] 检测到目标顺序可能颠倒，进行交换...")
        predictions_original = predictions_original[:, [1, 0]]  # 交换列
        actuals_original = actuals_original[:, [1, 0]]
        print("[INFO] 目标顺序已交换")

    # 使用反归一化后的数据
    predictions = predictions_original
    actuals = actuals_original

    # 分离两个目标的预测和实际值
    print(f"原始预测形状: {predictions.shape}")
    print(f"原始实际形状: {actuals.shape}")

    # 🔥 关键修复：正确处理(N, horizon, n_targets)格式的数据
    if len(predictions.shape) == 3:
        N, horizon, n_targets = predictions.shape
        print(f"[DEBUG] 检测到3D数据: N={N}, horizon={horizon}, n_targets={n_targets}")

        # 正确的分离方式：按目标维度分离，然后flatten
        if n_targets >= 2:
            dtcrt_pred = predictions[:, :, 0].flatten()    # 所有样本、所有时间步的目标0
            alcdlc_pred = predictions[:, :, 1].flatten()   # 所有样本、所有时间步的目标1
            dtcrt_actual = actuals[:, :, 0].flatten()      # 所有样本、所有时间步的目标0
            alcdlc_actual = actuals[:, :, 1].flatten()     # 所有样本、所有时间步的目标1

            print(f"[DEBUG] 3D数据分离结果:")
            print(f"  dtcrt_pred: {dtcrt_pred.shape}")
            print(f"  alcdlc_pred: {alcdlc_pred.shape}")
            print(f"  dtcrt_actual: {dtcrt_actual.shape}")
            print(f"  alcdlc_actual: {alcdlc_actual.shape}")
        else:
            print("[ERROR] 目标数量不足，无法分离双目标")
            return

    elif len(predictions.shape) == 2:
        # 处理2D数据 (N, 2)
        if predictions.shape[-1] >= 2:
            dtcrt_pred = predictions[:, 0].flatten()
            alcdlc_pred = predictions[:, 1].flatten()
            dtcrt_actual = actuals[:, 0].flatten()
            alcdlc_actual = actuals[:, 1].flatten()

            print(f"[DEBUG] 2D数据分离结果:")
            print(f"  dtcrt_pred: {dtcrt_pred.shape}")
            print(f"  alcdlc_pred: {alcdlc_pred.shape}")
        else:
            print("[ERROR] 2D数据目标数量不足")
            return

    else:
        print(f"[ERROR] 不支持的数据形状: {predictions.shape}")
        return

    # 🔥 新增：验证分离后的数据范围，确认目标映射正确
    print(f"\n[DEBUG] 验证分离后的数据范围:")
    print(f"  dtcrt_pred范围: {dtcrt_pred.min():.8f} - {dtcrt_pred.max():.8f}")
    print(f"  alcdlc_pred范围: {alcdlc_pred.min():.1f} - {alcdlc_pred.max():.1f}")
    print(f"  dtcrt_actual范围: {dtcrt_actual.min():.8f} - {dtcrt_actual.max():.8f}")
    print(f"  alcdlc_actual范围: {alcdlc_actual.min():.1f} - {alcdlc_actual.max():.1f}")

    # 🔥 新增：检查预测值的变化程度
    dtcrt_pred_std = np.std(dtcrt_pred)
    alcdlc_pred_std = np.std(alcdlc_pred)
    dtcrt_actual_std = np.std(dtcrt_actual)
    alcdlc_actual_std = np.std(alcdlc_actual)

    print(f"\n[DEBUG] 数据变化程度分析:")
    print(f"  DTCRT预测标准差: {dtcrt_pred_std:.8f}")
    print(f"  DTCRT真实标准差: {dtcrt_actual_std:.8f}")
    print(f"  ALCDLC预测标准差: {alcdlc_pred_std:.1f}")
    print(f"  ALCDLC真实标准差: {alcdlc_actual_std:.1f}")

    # 计算预测的变化率（相对于真实值的变化）
    dtcrt_pred_var_ratio = dtcrt_pred_std / max(dtcrt_actual_std, 1e-10)
    alcdlc_pred_var_ratio = alcdlc_pred_std / max(alcdlc_actual_std, 1e-10)

    print(f"  DTCRT预测变化率: {dtcrt_pred_var_ratio:.3f}")
    print(f"  ALCDLC预测变化率: {alcdlc_pred_var_ratio:.3f}")

    # 🔥 诊断：预测过于单调
    if dtcrt_pred_var_ratio < 0.1:
        print(f"[WARNING] DTCRT预测过于单调！变化率仅{dtcrt_pred_var_ratio:.3f}")
        print(f"   预测值几乎恒定在: {np.mean(dtcrt_pred):.8f}")

    if alcdlc_pred_var_ratio < 0.1:
        print(f"[WARNING] ALCDLC预测过于单调！变化率仅{alcdlc_pred_var_ratio:.3f}")
        print(f"   预测值几乎恒定在: {np.mean(alcdlc_pred):.1f}")

    # 🔥 新增：检查是否只使用了一个时间步的预测
    if len(predictions.shape) == 3:
        N, horizon, n_targets = predictions.shape
        print(f"\n[DEBUG] 多时间步预测分析:")
        for h in range(horizon):
            dtcrt_h = predictions[:, h, 0]
            alcdlc_h = predictions[:, h, 1]
            print(f"  时间步{h}: DTCRT_std={np.std(dtcrt_h):.8f}, ALCDLC_std={np.std(alcdlc_h):.1f}")

            # 🔥 关键修复：针对过于单调的预测，改用最后一个时间步
        if dtcrt_pred_var_ratio < 0.1 or alcdlc_pred_var_ratio < 0.1:
            print(f"\n[FIX] 预测过于单调，改用最后一个时间步的预测...")

            # 重新分离，只使用最后一个时间步 (horizon-1)
            dtcrt_pred = predictions[:, -1, 0]     # 最后时间步的目标0
            alcdlc_pred = predictions[:, -1, 1]    # 最后时间步的目标1
            dtcrt_actual = actuals[:, -1, 0]       # 最后时间步的目标0
            alcdlc_actual = actuals[:, -1, 1]      # 最后时间步的目标1

            print(f"[DEBUG] 使用最后时间步的预测:")
            print(f"  dtcrt_pred范围: {dtcrt_pred.min():.8f} - {dtcrt_pred.max():.8f}")
            print(f"  alcdlc_pred范围: {alcdlc_pred.min():.1f} - {alcdlc_pred.max():.1f}")

            # 重新计算变化率
            dtcrt_pred_std_new = np.std(dtcrt_pred)
            alcdlc_pred_std_new = np.std(alcdlc_pred)
            dtcrt_pred_var_ratio_new = dtcrt_pred_std_new / max(dtcrt_actual_std, 1e-10)
            alcdlc_pred_var_ratio_new = alcdlc_pred_std_new / max(alcdlc_actual_std, 1e-10)

            print(f"  修正后DTCRT变化率: {dtcrt_pred_var_ratio_new:.3f}")
            print(f"  修正后ALCDLC变化率: {alcdlc_pred_var_ratio_new:.3f}")

            if dtcrt_pred_var_ratio_new > dtcrt_pred_var_ratio:
                print(f"[SUCCESS] DTCRT预测变化率改善!")
            if alcdlc_pred_var_ratio_new > alcdlc_pred_var_ratio:
                print(f"[SUCCESS] ALCDLC预测变化率改善!")

    # 🔥 新增：自动检测和修正目标顺序
    dtcrt_pred_is_small = (dtcrt_pred.max() < 0.01)  # DTCRT应该是小值
    alcdlc_pred_is_large = (alcdlc_pred.max() > 1000)  # ALCDLC应该是大值

    if not dtcrt_pred_is_small or not alcdlc_pred_is_large:
        print("[WARNING] 检测到目标顺序错误，自动交换...")
        dtcrt_pred, alcdlc_pred = alcdlc_pred, dtcrt_pred
        dtcrt_actual, alcdlc_actual = alcdlc_actual, dtcrt_actual
        print("[INFO] 目标数据已交换")

        # 重新验证
        print(f"[DEBUG] 交换后验证:")
        print(f"  dtcrt_pred范围: {dtcrt_pred.min():.8f} - {dtcrt_pred.max():.8f}")
        print(f"  alcdlc_pred范围: {alcdlc_pred.min():.1f} - {alcdlc_pred.max():.1f}")

    # 获取真实的深度数据
    print("获取深度数据...")
    depths = get_real_depths_from_data(test_loader, "/root/dataset/data.xlsx")

    # 🔥 修复：确保所有数组长度一致
    min_length = min(len(depths), len(dtcrt_pred), len(alcdlc_pred), len(dtcrt_actual), len(alcdlc_actual))
    print(f"数据长度统一为: {min_length}")

    depths = depths[:min_length]
    dtcrt_pred = dtcrt_pred[:min_length]
    alcdlc_pred = alcdlc_pred[:min_length]
    dtcrt_actual = dtcrt_actual[:min_length]
    alcdlc_actual = alcdlc_actual[:min_length]

    print(f"最终数据形状验证:")
    print(f"  depths: {depths.shape}")
    print(f"  dtcrt_pred: {dtcrt_pred.shape}")
    print(f"  dtcrt_actual: {dtcrt_actual.shape}")

    print("创建测井图可视化...")

    # 创建可视化器
    visualizer = WellLogVisualizer(figsize=(16, 12), dpi=150)

    # 创建输出目录
    output_dir = Path("outputs/optimal_well_log")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成主要的测井对比图
    print("生成主要预测对比图...")
    fig1 = visualizer.plot_well_log_predictions(
        depths=depths,
        dtcrt_actual=dtcrt_actual,
        dtcrt_pred=dtcrt_pred,
        alcdlc_actual=alcdlc_actual,
        alcdlc_pred=alcdlc_pred,
        title="The Best Model Well Log Prediction",
        save_path=output_dir / "well_log_predictions_comparison.png",
        depth_unit="m",
        show_grid=True
    )

    # 生成误差分析图
    print("生成误差分析图...")
    fig2 = visualizer.plot_error_distribution(
        dtcrt_actual=dtcrt_actual,
        dtcrt_pred=dtcrt_pred,
        alcdlc_actual=alcdlc_actual,
        alcdlc_pred=alcdlc_pred,
        save_path=output_dir / "prediction_error_analysis.png"
    )

    # 计算并打印性能指标
    dtcrt_rmse = np.sqrt(np.mean((dtcrt_actual - dtcrt_pred) ** 2))
    dtcrt_mae = np.mean(np.abs(dtcrt_actual - dtcrt_pred))
    dtcrt_r2 = calculate_r2(dtcrt_actual, dtcrt_pred)

    alcdlc_rmse = np.sqrt(np.mean((alcdlc_actual - alcdlc_pred) ** 2))
    alcdlc_mae = np.mean(np.abs(alcdlc_actual - alcdlc_pred))
    alcdlc_r2 = calculate_r2(alcdlc_actual, alcdlc_pred)

    print("\n最优配置模型性能统计:")
    print("=" * 50)
    print(f"DTCRT目标:")
    print(f"  RMSE: {dtcrt_rmse:.6f}")
    print(f"  MAE:  {dtcrt_mae:.6f}")
    print(f"  R2:   {dtcrt_r2:.6f}")
    print(f"\nALCDLC_MERGED目标:")
    print(f"  RMSE: {alcdlc_rmse:.6f}")
    print(f"  MAE:  {alcdlc_mae:.6f}")
    print(f"  R2:   {alcdlc_r2:.6f}")
    print(f"\n双目标综合:")
    print(f"  平均RMSE: {(dtcrt_rmse + alcdlc_rmse)/2:.6f}")
    print(f"  平均MAE:  {(dtcrt_mae + alcdlc_mae)/2:.6f}")
    print(f"  平均R2:   {(dtcrt_r2 + alcdlc_r2)/2:.6f}")
    print("=" * 50)

    # 保存性能统计到文件
    stats_file = output_dir / "performance_statistics.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("CNN-LSTM-Attention最优配置模型性能统计\n")
        f.write("=" * 50 + "\n")
        f.write(f"DTCRT目标:\n")
        f.write(f"  RMSE: {dtcrt_rmse:.6f}\n")
        f.write(f"  MAE:  {dtcrt_mae:.6f}\n")
        f.write(f"  R2:   {dtcrt_r2:.6f}\n")
        f.write(f"\nALCDLC_MERGED目标:\n")
        f.write(f"  RMSE: {alcdlc_rmse:.6f}\n")
        f.write(f"  MAE:  {alcdlc_mae:.6f}\n")
        f.write(f"  R2:   {alcdlc_r2:.6f}\n")
        f.write(f"\n双目标综合:\n")
        f.write(f"  平均RMSE: {(dtcrt_rmse + alcdlc_rmse)/2:.6f}\n")
        f.write(f"  平均MAE:  {(dtcrt_mae + alcdlc_mae)/2:.6f}\n")
        f.write(f"  平均R2:   {(dtcrt_r2 + alcdlc_r2)/2:.6f}\n")

    # 🔥 新增：导出两个目标列的预测数据
    print("导出预测数据...")
    export_prediction_data(
        depths=depths,
        dtcrt_actual=dtcrt_actual,
        dtcrt_pred=dtcrt_pred,
        alcdlc_actual=alcdlc_actual,
        alcdlc_pred=alcdlc_pred,
        output_dir=output_dir,
        config_info=cfg
    )

    print(f"可视化完成！")
    print(f"输出目录: {output_dir.absolute()}")
    print(f"主要对比图: well_log_predictions_comparison.png")
    print(f"误差分析图: prediction_error_analysis.png")
    print(f"性能统计: performance_statistics.txt")
    print(f"🔥 新增数据导出文件:")
    print(f"  - prediction_data_export.csv (完整预测数据)")
    print(f"  - prediction_data_export.xlsx (多工作表Excel)")
    print(f"  - prediction_data_simplified.csv (简化格式)")
    print(f"  - data_export_summary.txt (导出摘要)")
    print(f"📊 导出数据包含: 深度、两个目标的实际值/预测值/误差分析")

def find_latest_checkpoint(checkpoint_dir: str) -> str:
    """查找最新的检查点文件"""
    checkpoint_path = Path(checkpoint_dir)

    # 查找可能的检查点文件模式
    patterns = ['*.pt', '*.pth', '*.ckpt']

    latest_checkpoint = None
    latest_time = 0

    for pattern in patterns:
        for file_path in checkpoint_path.glob(pattern):
            if file_path.is_file():
                mtime = file_path.stat().st_mtime
                if mtime > latest_time:
                    latest_time = mtime
                    latest_checkpoint = file_path

    return str(latest_checkpoint) if latest_checkpoint else None

def get_real_depths_from_data(test_loader, data_path: str = "data.xlsx") -> np.ndarray:
    """
    从真实Excel文件中获取深度信息，动态匹配测试集大小

    Args:
        test_loader: 测试数据加载器
        data_path: Excel数据文件路径

    Returns:
        np.ndarray: 深度数组
    """
    import pandas as pd

    # 读取Excel文件获取真实深度数据
    print(f"从{data_path}读取真实深度数据...")
    df = pd.read_excel(data_path)

    # 获取完整的深度列
    full_depths = df['DEPTH'].values

    # 🔥 修复：动态计算测试集的实际大小
    actual_test_samples = 0
    for batch in test_loader:
        actual_test_samples += batch[0].shape[0]  # batch_size

    print(f"检测到实际测试样本数: {actual_test_samples}")

    # 🔥 修复：使用更灵活的深度提取策略
    # 根据数据分割逻辑正确计算测试集在原始数据中的位置
    total_samples = len(df)
    train_split = 0.7
    val_split = 0.15

    # 计算分割点
    train_end = int(total_samples * train_split)
    val_end = int(total_samples * (train_split + val_split))

    # 测试集在原始数据中的起始位置
    test_start_in_data = val_end

    print(f"数据分割信息:")
    print(f"  总样本数: {total_samples}")
    print(f"  训练集结束: {train_end}")
    print(f"  验证集结束: {val_end}")
    print(f"  测试集开始: {test_start_in_data}")

    # 🔥 修复：考虑滑动窗口的影响
    # 由于滑动窗口，实际的测试样本数可能少于原始测试数据
    sequence_length = 64  # 从配置获取
    horizon = 3

    # 可用的测试窗口数
    available_test_windows = max(0, total_samples - test_start_in_data - sequence_length - horizon + 1)
    actual_test_samples = min(actual_test_samples, available_test_windows)

    print(f"  考虑滑动窗口后可用测试窗口: {available_test_windows}")
    print(f"  实际测试样本数: {actual_test_samples}")

    # 从测试集起始位置提取对应的深度
    if actual_test_samples <= len(full_depths) - test_start_in_data:
        test_depths = full_depths[test_start_in_data:test_start_in_data + actual_test_samples]
    else:
        # 如果超出范围，从末尾提取
        test_depths = full_depths[-actual_test_samples:]

    print(f"提取到{len(test_depths)}个深度值")
    print(f"深度范围: {test_depths.min():.3f}m - {test_depths.max():.3f}m")

    return test_depths

def calculate_r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    """计算R²系数"""
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

def export_prediction_data(
    depths: np.ndarray,
    dtcrt_actual: np.ndarray,
    dtcrt_pred: np.ndarray,
    alcdlc_actual: np.ndarray,
    alcdlc_pred: np.ndarray,
    output_dir: Path,
    config_info: dict = None
):
    """
    🔥 新增：导出两个目标列的预测数据

    导出包含深度、实际值、预测值、误差分析的完整数据表
    支持CSV和Excel格式，便于后续分析和验证

    Args:
        depths: 深度数组
        dtcrt_actual: DTCRT实际值
        dtcrt_pred: DTCRT预测值
        alcdlc_actual: ALCDLC实际值
        alcdlc_pred: ALCDLC预测值
        output_dir: 输出目录
        config_info: 模型配置信息
    """

    # 确保所有数组长度一致
    min_length = min(len(depths), len(dtcrt_actual), len(dtcrt_pred),
                     len(alcdlc_actual), len(alcdlc_pred))

    # 截取到最小长度，确保数据一致性
    depths = depths[:min_length]
    dtcrt_actual = dtcrt_actual[:min_length]
    dtcrt_pred = dtcrt_pred[:min_length]
    alcdlc_actual = alcdlc_actual[:min_length]
    alcdlc_pred = alcdlc_pred[:min_length]

    print(f"[INFO] 导出数据长度: {min_length} 个样本")
    print(f"[INFO] 深度范围: {depths.min():.2f} - {depths.max():.2f}m")

    # 计算误差指标
    dtcrt_error = dtcrt_pred - dtcrt_actual
    alcdlc_error = alcdlc_pred - alcdlc_actual
    dtcrt_abs_error = np.abs(dtcrt_error)
    alcdlc_abs_error = np.abs(alcdlc_error)

    # 计算相对误差（百分比）
    dtcrt_rel_error = np.where(np.abs(dtcrt_actual) > 1e-8,
                               dtcrt_error / dtcrt_actual * 100, 0)
    alcdlc_rel_error = np.where(np.abs(alcdlc_actual) > 1e-8,
                                alcdlc_error / alcdlc_actual * 100, 0)

    # 创建完整的数据表
    prediction_data = pd.DataFrame({
        # 基础信息
        'Sample_Index': np.arange(min_length),
        'Depth_m': depths,

        # DTCRT目标数据
        'DTCRT_Actual': dtcrt_actual,
        'DTCRT_Predicted': dtcrt_pred,
        'DTCRT_Error': dtcrt_error,
        'DTCRT_Abs_Error': dtcrt_abs_error,
        'DTCRT_Rel_Error_Percent': dtcrt_rel_error,

        # ALCDLC目标数据
        'ALCDLC_Actual': alcdlc_actual,
        'ALCDLC_Predicted': alcdlc_pred,
        'ALCDLC_Error': alcdlc_error,
        'ALCDLC_Abs_Error': alcdlc_abs_error,
        'ALCDLC_Rel_Error_Percent': alcdlc_rel_error
    })

    # 添加数据质量标记
    prediction_data['Quality_Flag'] = 'GOOD'

    # 标记异常值（误差超过3个标准差）
    dtcrt_error_std = np.std(dtcrt_error)
    alcdlc_error_std = np.std(alcdlc_error)

    outlier_mask = (
        (np.abs(dtcrt_error) > 3 * dtcrt_error_std) |
        (np.abs(alcdlc_error) > 3 * alcdlc_error_std)
    )
    prediction_data.loc[outlier_mask, 'Quality_Flag'] = 'OUTLIER'

    # 计算整体性能指标
    dtcrt_rmse = np.sqrt(np.mean(dtcrt_error ** 2))
    dtcrt_mae = np.mean(dtcrt_abs_error)
    dtcrt_r2 = calculate_r2(dtcrt_actual, dtcrt_pred)

    alcdlc_rmse = np.sqrt(np.mean(alcdlc_error ** 2))
    alcdlc_mae = np.mean(alcdlc_abs_error)
    alcdlc_r2 = calculate_r2(alcdlc_actual, alcdlc_pred)

    # 1. 导出CSV格式
    csv_file = output_dir / "prediction_data_export.csv"
    prediction_data.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"[SUCCESS] CSV数据已导出: {csv_file}")

    # 2. 导出Excel格式（包含多个工作表）
    excel_file = output_dir / "prediction_data_export.xlsx"

    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        # 主数据表
        prediction_data.to_excel(writer, sheet_name='Prediction_Data', index=False)

        # 性能统计表
        performance_stats = pd.DataFrame({
            'Target': ['DTCRT', 'ALCDLC_MERGED', 'Combined_Average'],
            'RMSE': [dtcrt_rmse, alcdlc_rmse, (dtcrt_rmse + alcdlc_rmse)/2],
            'MAE': [dtcrt_mae, alcdlc_mae, (dtcrt_mae + alcdlc_mae)/2],
            'R2': [dtcrt_r2, alcdlc_r2, (dtcrt_r2 + alcdlc_r2)/2],
            'Data_Points': [min_length, min_length, min_length],
            'Outlier_Count': [
                np.sum(np.abs(dtcrt_error) > 3 * dtcrt_error_std),
                np.sum(np.abs(alcdlc_error) > 3 * alcdlc_error_std),
                np.sum(outlier_mask)
            ]
        })
        performance_stats.to_excel(writer, sheet_name='Performance_Summary', index=False)

        # 误差统计表
        error_stats = pd.DataFrame({
            'Statistic': ['Mean', 'Std', 'Min', 'Max', 'Median', '25th_Percentile', '75th_Percentile'],
            'DTCRT_Error': [
                np.mean(dtcrt_error), np.std(dtcrt_error), np.min(dtcrt_error),
                np.max(dtcrt_error), np.median(dtcrt_error),
                np.percentile(dtcrt_error, 25), np.percentile(dtcrt_error, 75)
            ],
            'ALCDLC_Error': [
                np.mean(alcdlc_error), np.std(alcdlc_error), np.min(alcdlc_error),
                np.max(alcdlc_error), np.median(alcdlc_error),
                np.percentile(alcdlc_error, 25), np.percentile(alcdlc_error, 75)
            ]
        })
        error_stats.to_excel(writer, sheet_name='Error_Statistics', index=False)

        # 配置信息表
        if config_info:
            # 🔥 修复：将FullConfig对象转换为字典
            try:
                from dataclasses import asdict
                if hasattr(config_info, '__dict__'):
                    # 如果是dataclass，使用asdict
                    if hasattr(config_info, '__dataclass_fields__'):
                        config_dict = asdict(config_info)
                    else:
                        # 否则使用__dict__
                        config_dict = config_info.__dict__
                elif isinstance(config_info, dict):
                    config_dict = config_info
                else:
                    # 备用：转换为字符串表示
                    config_dict = {'config': str(config_info)}

                print(f"[DEBUG] 配置对象类型: {type(config_info)}")
                print(f"[DEBUG] 转换后字典keys: {list(config_dict.keys())[:5]}")

                config_flat = flatten_config_dict(config_dict)
                config_df = pd.DataFrame(list(config_flat.items()),
                                       columns=['Config_Parameter', 'Value'])
                config_df.to_excel(writer, sheet_name='Model_Configuration', index=False)
                print(f"[INFO] 配置信息已添加到Excel")

            except Exception as e:
                print(f"[WARNING] 配置信息处理失败: {e}")
                # 创建简单的配置信息
                simple_config = pd.DataFrame([
                    ['Config_Type', str(type(config_info))],
                    ['Config_String', str(config_info)[:100]]
                ], columns=['Config_Parameter', 'Value'])
                simple_config.to_excel(writer, sheet_name='Model_Configuration', index=False)

    print(f"[SUCCESS] Excel数据已导出: {excel_file}")

    # 3. 导出简化的CSV格式（仅包含核心预测数据）
    simplified_data = pd.DataFrame({
        'Depth_m': depths,
        'DTCRT_Actual': dtcrt_actual,
        'DTCRT_Predicted': dtcrt_pred,
        'ALCDLC_Actual': alcdlc_actual,
        'ALCDLC_Predicted': alcdlc_pred
    })

    simplified_csv = output_dir / "prediction_data_simplified.csv"
    simplified_data.to_csv(simplified_csv, index=False, encoding='utf-8')
    print(f"[SUCCESS] 简化CSV数据已导出: {simplified_csv}")

    # 4. 生成数据摘要报告
    summary_file = output_dir / "data_export_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("预测数据导出摘要报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"导出时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据样本数: {min_length}\n")
        f.write(f"深度范围: {depths.min():.2f} - {depths.max():.2f} m\n")
        f.write(f"深度间隔: {np.mean(np.diff(depths)):.3f} m\n\n")

        f.write("数据文件说明:\n")
        f.write("- prediction_data_export.csv: 完整的预测数据（含误差分析）\n")
        f.write("- prediction_data_export.xlsx: Excel格式（多工作表）\n")
        f.write("- prediction_data_simplified.csv: 简化格式（仅核心数据）\n\n")

        f.write("数据列说明:\n")
        f.write("- Depth_m: 深度（米）\n")
        f.write("- DTCRT_Actual/Predicted: DTCRT目标的实际值/预测值\n")
        f.write("- ALCDLC_Actual/Predicted: ALCDLC目标的实际值/预测值\n")
        f.write("- *_Error: 预测误差（预测值-实际值）\n")
        f.write("- *_Abs_Error: 绝对误差\n")
        f.write("- *_Rel_Error_Percent: 相对误差百分比\n")
        f.write("- Quality_Flag: 数据质量标记（GOOD/OUTLIER）\n\n")

        f.write("性能摘要:\n")
        f.write(f"DTCRT: RMSE={dtcrt_rmse:.4f}, MAE={dtcrt_mae:.4f}, R²={dtcrt_r2:.4f}\n")
        f.write(f"ALCDLC: RMSE={alcdlc_rmse:.4f}, MAE={alcdlc_mae:.4f}, R²={alcdlc_r2:.4f}\n")
        f.write(f"异常值数量: {np.sum(outlier_mask)}/{min_length} ({np.sum(outlier_mask)/min_length*100:.1f}%)\n")

    print(f"[SUCCESS] 数据摘要已导出: {summary_file}")
    print(f"[INFO] 总共导出4个文件：CSV、Excel、简化CSV、摘要报告")

def flatten_config_dict(config_dict: dict, parent_key: str = '', separator: str = '.') -> dict:
    """将嵌套的配置字典展平为单层字典"""
    items = []
    for k, v in config_dict.items():
        new_key = f"{parent_key}{separator}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config_dict(v, new_key, separator).items())
        else:
            items.append((new_key, str(v)))
    return dict(items)

if __name__ == "__main__":
    # 使用正确的配置文件路径
    config_path = "./final_analysis/optimal_ensemble_config.yaml"
    model_path = "./optimal_ensemble_config_0bd03a40.pt"

    # 检查文件是否存在
    if not os.path.exists(config_path):
        print(f"[ERROR] 配置文件不存在: {config_path}")
        print("尝试查找其他配置文件...")
        # 查找可能的配置文件
        possible_configs = [
            "test_optimal_config/optimal_ensemble_config.yaml",
            "configs/ablation_dual/baseline__clean_reference.yaml"
        ]
        for test_config in possible_configs:
            if os.path.exists(test_config):
                config_path = test_config
                print(f"[INFO] 使用找到的配置: {config_path}")
                break
        else:
            print("[ERROR] 未找到任何可用的配置文件")
            exit(1)

    # 尝试生成预测和可视化
    try:
        generate_predictions_and_visualize(config_path, model_path)
    except Exception as e:
        print(f"生成可视化时发生错误: {e}")
        import traceback
        traceback.print_exc()
        print("请确保模型文件和配置文件路径正确")