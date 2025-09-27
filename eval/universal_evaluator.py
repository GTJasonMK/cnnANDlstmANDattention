"""
通用评估器
解决通道数不匹配问题，提供简洁统一的评估接口
"""

from __future__ import annotations

import os
import json
import warnings
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 导入数据形状管理器和统一工具
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_shape_manager import DataShapeManager, DataShape, PreprocessConfig
from utils.data_loader import load_array_from_path
from utils.metrics import calculate_regression_metrics
from utils.system_utils import get_device
from model_architecture import CNNLSTMAttentionModel


class UniversalEvaluator:
    """通用评估器 - 自动处理通道数匹配问题"""
    
    def __init__(self, cache_dir: str = ".cache"):
        self.shape_manager = DataShapeManager(os.path.join(cache_dir, "data_shapes"))
        self.device = get_device()
        
    def load_data(self, data_path: str) -> np.ndarray:
        """加载数据 - 使用统一的数据加载工具"""
        data, _ = load_array_from_path(data_path, return_columns=False)
        return data
    
    def prepare_data(self, 
                    data: np.ndarray,
                    data_shape: DataShape,
                    preprocess_config: PreprocessConfig,
                    batch_size: int = 32) -> Tuple[DataLoader, np.ndarray]:
        """准备评估数据"""
        
        # 数据归一化
        if preprocess_config.normalize == "standard":
            mean = data.mean(axis=0)
            std = data.std(axis=0) + 1e-8
            data_norm = (data - mean) / std
        elif preprocess_config.normalize == "minmax":
            min_val = data.min(axis=0)
            max_val = data.max(axis=0)
            data_norm = (data - min_val) / (max_val - min_val + 1e-8)
        else:
            data_norm = data.copy()
        
        # 小波变换（如果需要）
        if preprocess_config.wavelet_enabled:
            data_norm = self._apply_wavelet_transform(data_norm, preprocess_config.wavelet_config)
        
        # 特征选择
        if data_shape.feature_indices:
            features = data_norm[:, data_shape.feature_indices]
        else:
            features = data_norm
            
        if data_shape.target_indices:
            targets = data_norm[:, data_shape.target_indices]
        else:
            targets = data_norm[:, -1:]  # 默认最后一列
        
        # 创建时间窗口
        X, y = self._create_windows(
            features, targets, 
            data_shape.sequence_length, 
            data_shape.horizon
        )
        
        # 创建DataLoader
        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        return dataloader, data_norm
    
    def _apply_wavelet_transform(self, data: np.ndarray, wavelet_config: Dict[str, Any]) -> np.ndarray:
        """应用小波变换"""
        try:
            import pywt
        except ImportError:
            warnings.warn("PyWavelets未安装，跳过小波变换")
            return data
            
        wavelet = wavelet_config.get('wavelet', 'db4')
        level = wavelet_config.get('level', 3)
        mode = wavelet_config.get('mode', 'symmetric')
        take = wavelet_config.get('take', 'all')
        
        N, F = data.shape
        results = []
        
        for f in range(F):
            coeffs = pywt.wavedec(data[:, f], wavelet=wavelet, level=level, mode=mode)
            
            if take == 'approx':
                sel_coeffs = [coeffs[0]]
            elif take == 'details':
                sel_coeffs = coeffs[1:]
            else:  # 'all'
                sel_coeffs = coeffs
                
            # 插值到原始长度
            bands = []
            for coeff in sel_coeffs:
                if len(coeff) != N:
                    # 插值到原始长度
                    indices_old = np.linspace(0, len(coeff) - 1, len(coeff))
                    indices_new = np.linspace(0, len(coeff) - 1, N)
                    coeff_interp = np.interp(indices_new, indices_old, coeff)
                else:
                    coeff_interp = coeff
                bands.append(coeff_interp)
                
            # 合并所有频带
            feature_expanded = np.column_stack(bands)
            results.append(feature_expanded)
        
        return np.concatenate(results, axis=1).astype(np.float32)
    
    def _create_windows(self, 
                       features: np.ndarray, 
                       targets: np.ndarray,
                       seq_len: int, 
                       horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """创建时间窗口"""
        N = features.shape[0]
        if N < seq_len + horizon:
            raise ValueError(f"数据长度不足: {N} < {seq_len + horizon}")
            
        num_windows = N - seq_len - horizon + 1
        
        X = np.zeros((num_windows, seq_len, features.shape[1]), dtype=np.float32)
        
        if targets.ndim == 1:
            y = np.zeros((num_windows, horizon), dtype=np.float32)
        else:
            y = np.zeros((num_windows, horizon, targets.shape[1]), dtype=np.float32)
        
        for i in range(num_windows):
            X[i] = features[i:i+seq_len]
            target_window = targets[i+seq_len:i+seq_len+horizon]
            
            if targets.ndim == 1:
                y[i] = target_window
            else:
                y[i] = target_window
                
        return X, y
    
    def load_and_build_model(self, checkpoint_path: str) -> Tuple[nn.Module, Dict[str, Any], DataShape]:
        """加载模型和配置"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config = checkpoint.get('cfg', {})
        
        if not config:
            raise ValueError("检查点中没有保存配置信息")
        
        # 从检查点推断数据形状
        checkpoint_shape = self.shape_manager._infer_shape_from_checkpoint(checkpoint, config)
        
        # 构建模型
        model = self._build_model_from_config(config, checkpoint_shape)
        
        # 加载权重
        try:
            model.load_state_dict(checkpoint['model_state'], strict=True)
        except RuntimeError as e:
            raise RuntimeError(f"模型权重加载失败，请检查配置和数据形状是否匹配: {e}")
        
        model.to(self.device)
        model.eval()
        
        return model, config, checkpoint_shape
    
    def _build_model_from_config(self, config: Dict[str, Any], data_shape: DataShape) -> CNNLSTMAttentionModel:
        """从配置构建模型"""
        model_config = config.get('model', {})
        
        # CNN配置
        cnn_config = model_config.get('cnn', {})
        cnn_layers = cnn_config.get('layers', [])
        cnn_variant = cnn_config.get('variant', 'standard')
        
        # LSTM配置
        lstm_config = model_config.get('lstm', {})
        rnn_type = lstm_config.get('rnn_type', 'lstm')
        
        # 注意力配置
        attn_config = model_config.get('attention', {})
        
        # TCN配置（如果有）
        tcn_config = model_config.get('tcn', {})
        if tcn_config.get('enabled', False) or cnn_variant == 'tcn':
            cnn_layers = tcn_config.get('layers', cnn_layers)
            cnn_variant = 'tcn'
        
        model = CNNLSTMAttentionModel(
            num_features=data_shape.input_channels,
            cnn_layers=cnn_layers,
            use_batchnorm=cnn_config.get('use_batchnorm', True),
            cnn_dropout=cnn_config.get('dropout', 0.1),
            lstm_hidden=lstm_config.get('hidden_size', 128),
            lstm_layers=lstm_config.get('num_layers', 2),
            bidirectional=lstm_config.get('bidirectional', True),
            attn_enabled=attn_config.get('enabled', True),
            attn_heads=attn_config.get('num_heads', 8),
            attn_dropout=attn_config.get('dropout', 0.1),
            fc_hidden=model_config.get('fc_hidden', 128),
            forecast_horizon=data_shape.horizon,
            n_targets=data_shape.n_targets,
            attn_add_pos_enc=attn_config.get('add_positional_encoding', 
                                           attn_config.get('add_posional_encoding', False)),
            lstm_dropout=lstm_config.get('dropout', 0.1),
            cnn_variant=cnn_variant,
            attn_variant=attn_config.get('variant', 'standard'),
            multiscale_scales=attn_config.get('multiscale_scales', [1, 2]),
            multiscale_fuse=attn_config.get('multiscale_fuse', 'sum'),
            attn_positional_mode=attn_config.get('positional_mode', 'none'),
            local_window_size=attn_config.get('local_window_size', 64),
            local_dilation=attn_config.get('local_dilation', 1),
            st_mode=attn_config.get('st_mode', 'serial'),
            st_fuse=attn_config.get('st_fuse', 'sum'),
            cnn_use_channel_attention=cnn_config.get('use_channel_attention', False),
            cnn_channel_attention_type=cnn_config.get('channel_attention_type', 'eca'),
            normalization=model_config.get('normalization'),
            decomposition=model_config.get('decomposition')
        )
        
        # 设置RNN类型
        model.rnn_type = rnn_type
        
        return model
    
    def evaluate_model(self, 
                      model: nn.Module, 
                      dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """评估模型"""
        model.eval()
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                preds = model(batch_x)
                
                all_preds.append(preds.cpu())
                all_targets.append(batch_y.cpu())
        
        predictions = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        # 计算指标
        metrics = self._calculate_metrics(predictions, targets)
        
        return predictions, targets, metrics
    
    def _calculate_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """计算评估指标 - 使用统一的指标计算工具"""
        return calculate_regression_metrics(predictions, targets)
    
    def auto_evaluate(self, 
                     checkpoint_path: str,
                     data_path: str,
                     output_dir: Optional[str] = None,
                     batch_size: int = 32,
                     save_results: bool = True) -> Dict[str, Any]:
        """自动评估（处理所有兼容性问题）"""
        
        print(f"[INFO] 加载数据: {data_path}")
        data = self.load_data(data_path)
        
        print(f"[INFO] 加载模型: {checkpoint_path}")
        model, config, checkpoint_shape = self.load_and_build_model(checkpoint_path)
        
        print(f"[INFO] 分析数据形状...")
        print(f"  - 检查点期望: 输入通道={checkpoint_shape.input_channels}, 输出通道={checkpoint_shape.output_channels}")
        print(f"  - 数据形状: {data.shape}")
        
        # 提取预处理配置
        preprocess_config = self.shape_manager._extract_preprocess_config(config)
        
        # 验证兼容性
        is_compatible, message, _ = self.shape_manager.validate_compatibility(
            checkpoint_path, data, config
        )
        
        if not is_compatible:
            print(f"[WARN] 检测到不兼容: {message}")
            print(f"[INFO] 尝试自动修复配置...")
            
            # 自动修复配置
            fixed_config = self.shape_manager.auto_fix_config(data, config, checkpoint_shape)
            
            # 重新提取预处理配置
            preprocess_config = self.shape_manager._extract_preprocess_config(fixed_config)
            
            print(f"[INFO] 配置已自动修复")
            print(f"  - 小波变换: {'开启' if preprocess_config.wavelet_enabled else '关闭'}")
            if preprocess_config.wavelet_enabled:
                wc = preprocess_config.wavelet_config
                print(f"    小波: {wc['wavelet']}, 层级: {wc['level']}, 模式: {wc['take']}")
        
        # 准备数据
        print(f"[INFO] 准备评估数据...")
        dataloader, processed_data = self.prepare_data(
            data, checkpoint_shape, preprocess_config, batch_size
        )
        
        print(f"[INFO] 开始评估...")
        predictions, targets, metrics = self.evaluate_model(model, dataloader)
        
        # 格式化输出
        print(f"[INFO] 评估完成!")
        print("=" * 50)
        print("评估结果:")
        for key, value in metrics.items():
            if not key.startswith('test_'):  # 避免重复显示
                print(f"  {key.upper()}: {value:.6f}")
        print("=" * 50)
        
        # 保存结果
        if save_results and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存指标
            with open(os.path.join(output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            
            # 保存预测结果
            np.save(os.path.join(output_dir, 'predictions.npy'), predictions.numpy())
            np.save(os.path.join(output_dir, 'targets.npy'), targets.numpy())
            
            print(f"[INFO] 结果已保存到: {output_dir}")
        
        return {
            'metrics': metrics,
            'predictions': predictions,
            'targets': targets,
            'data_shape': checkpoint_shape,
            'preprocess_config': preprocess_config,
            'model_config': config
        }