"""
数据形状管理器
统一管理训练、评估、可视化过程中的数据形状一致性问题
"""

from __future__ import annotations

import os
import json
import hashlib
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass, asdict
import numpy as np
import torch


@dataclass
class DataShape:
    """数据形状规范"""
    input_channels: int
    output_channels: int
    sequence_length: int
    horizon: int
    n_features: int  # 原始数据特征数
    n_targets: int  # 目标变量数
    feature_indices: Optional[List[int]] = None
    target_indices: Optional[List[int]] = None
    
    def __post_init__(self):
        if self.feature_indices is None:
            self.feature_indices = list(range(self.n_features))
        if self.target_indices is None:
            self.target_indices = [self.n_features - 1]  # 默认最后一列为目标
            
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataShape':
        return cls(**data)
    
    def get_hash(self) -> str:
        """获取数据形状的唯一哈希值"""
        data_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]


@dataclass
class PreprocessConfig:
    """预处理配置"""
    normalize: str = "minmax"  # minmax, none (standard removed as default)
    wavelet_enabled: bool = False
    wavelet_config: Optional[Dict[str, Any]] = None
    revin_enabled: bool = False
    decomposition_enabled: bool = False
    
    def __post_init__(self):
        if self.wavelet_config is None:
            self.wavelet_config = {
                'wavelet': 'db4',
                'level': 3,
                'mode': 'symmetric',
                'take': 'all'
            }
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PreprocessConfig':
        return cls(**data)


class DataShapeManager:
    """数据形状管理器"""
    
    def __init__(self, cache_dir: str = ".cache/data_shapes"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def analyze_data_shape(self, 
                          data: np.ndarray, 
                          config: Dict[str, Any],
                          preprocess_config: Optional[PreprocessConfig] = None) -> DataShape:
        """分析数据形状"""
        if preprocess_config is None:
            preprocess_config = self._extract_preprocess_config(config)
        
        n_samples, n_features = data.shape
        
        # 从配置中提取基本参数
        sequence_length = config.get('data', {}).get('sequence_length', 64)
        horizon = config.get('model', {}).get('forecast_horizon', 
                           config.get('data', {}).get('horizon', 1))
        
        feature_indices = config.get('data', {}).get('feature_indices')
        target_indices = config.get('data', {}).get('target_indices')
        
        if feature_indices is None:
            feature_indices = list(range(n_features))
        if target_indices is None:
            target_indices = [n_features - 1]
            
        # 计算输入通道数（考虑小波变换等预处理）
        input_channels = len(feature_indices)
        if preprocess_config.wavelet_enabled:
            # 小波变换会扩展通道数
            level = preprocess_config.wavelet_config.get('level', 3)
            take = preprocess_config.wavelet_config.get('take', 'all')
            if take == 'all':
                expansion_factor = level + 1  # 近似 + 细节系数
            elif take == 'approx':
                expansion_factor = 1
            elif take == 'details':
                expansion_factor = level
            else:
                expansion_factor = 1
            input_channels *= expansion_factor
            
        # 计算输出通道数
        n_targets = len(target_indices)
        output_channels = horizon * n_targets
        
        return DataShape(
            input_channels=input_channels,
            output_channels=output_channels,
            sequence_length=sequence_length,
            horizon=horizon,
            n_features=n_features,
            n_targets=n_targets,
            feature_indices=feature_indices,
            target_indices=target_indices
        )
    
    def _extract_preprocess_config(self, config: Dict[str, Any]) -> PreprocessConfig:
        """从配置中提取预处理配置"""
        data_cfg = config.get('data', {})
        model_cfg = config.get('model', {})
        
        # 小波配置
        wavelet_cfg = data_cfg.get('wavelet', {})
        wavelet_enabled = wavelet_cfg.get('enabled', False)
        
        # RevIN配置
        revin_enabled = False
        if 'normalization' in model_cfg:
            revin_cfg = model_cfg['normalization'].get('revin', {})
            revin_enabled = revin_cfg.get('enabled', False)
            
        # 分解配置
        decomp_enabled = False
        if 'decomposition' in model_cfg:
            decomp_enabled = model_cfg['decomposition'].get('enabled', False)
            
        return PreprocessConfig(
            normalize=data_cfg.get('normalize', 'standard'),
            wavelet_enabled=wavelet_enabled,
            wavelet_config=wavelet_cfg if wavelet_enabled else None,
            revin_enabled=revin_enabled,
            decomposition_enabled=decomp_enabled
        )
    
    def save_shape_info(self, data_shape: DataShape, 
                       preprocess_config: PreprocessConfig,
                       model_config: Dict[str, Any],
                       identifier: str) -> str:
        """保存形状信息到缓存"""
        info = {
            'data_shape': data_shape.to_dict(),
            'preprocess_config': preprocess_config.to_dict(),
            'model_config': model_config,
            'identifier': identifier
        }
        
        cache_file = os.path.join(self.cache_dir, f"{identifier}.json")
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        
        return cache_file
    
    def load_shape_info(self, identifier: str) -> Tuple[DataShape, PreprocessConfig, Dict[str, Any]]:
        """从缓存加载形状信息"""
        cache_file = os.path.join(self.cache_dir, f"{identifier}.json")
        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"Shape cache not found: {cache_file}")
            
        with open(cache_file, 'r', encoding='utf-8') as f:
            info = json.load(f)
            
        data_shape = DataShape.from_dict(info['data_shape'])
        preprocess_config = PreprocessConfig.from_dict(info['preprocess_config'])
        model_config = info['model_config']
        
        return data_shape, preprocess_config, model_config
    
    def validate_compatibility(self, 
                             checkpoint_path: str,
                             data: np.ndarray,
                             config: Dict[str, Any]) -> Tuple[bool, str, Optional[DataShape]]:
        """验证检查点与数据的兼容性"""
        try:
            # 加载检查点
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            ckpt_config = checkpoint.get('cfg', {})
            
            # 从检查点推断形状
            ckpt_shape = self._infer_shape_from_checkpoint(checkpoint, ckpt_config)
            
            # 分析当前数据形状
            current_shape = self.analyze_data_shape(data, config)
            
            # 检查兼容性
            if ckpt_shape.input_channels != current_shape.input_channels:
                return False, f"输入通道数不匹配: checkpoint={ckpt_shape.input_channels}, data={current_shape.input_channels}", ckpt_shape
            
            if ckpt_shape.output_channels != current_shape.output_channels:
                return False, f"输出通道数不匹配: checkpoint={ckpt_shape.output_channels}, data={current_shape.output_channels}", ckpt_shape
            
            return True, "兼容", ckpt_shape
            
        except Exception as e:
            return False, f"验证失败: {str(e)}", None
    
    def _infer_shape_from_checkpoint(self, checkpoint: Dict[str, Any], config: Dict[str, Any]) -> DataShape:
        """从检查点推断数据形状"""
        state_dict = checkpoint.get('model_state', {})
        
        # 推断输入通道数
        input_channels = None
        for key, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor) and tensor.ndim == 3 and key.endswith('.weight'):
                # 卷积层权重: [out_channels, in_channels, kernel_size]
                in_ch = tensor.shape[1]
                if input_channels is None or in_ch > input_channels:
                    input_channels = in_ch
                    
        # 推断输出通道数
        output_channels = None
        for key, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor) and tensor.ndim == 2 and key.startswith('head') and key.endswith('.weight'):
                # 输出层权重: [out_features, in_features]
                output_channels = tensor.shape[0]
                break
                
        # 从配置中获取其他信息
        model_cfg = config.get('model', {})
        data_cfg = config.get('data', {})
        
        sequence_length = data_cfg.get('sequence_length', 64)
        horizon = model_cfg.get('forecast_horizon', data_cfg.get('horizon', 1))
        
        # 推断目标变量数
        n_targets = output_channels // horizon if output_channels and horizon else 1
        
        feature_indices = data_cfg.get('feature_indices')
        target_indices = data_cfg.get('target_indices')
        
        # 推断原始特征数（考虑小波变换）
        n_features = input_channels
        preprocess_config = self._extract_preprocess_config(config)
        if preprocess_config.wavelet_enabled:
            level = preprocess_config.wavelet_config.get('level', 3)
            take = preprocess_config.wavelet_config.get('take', 'all')
            if take == 'all':
                expansion_factor = level + 1
            elif take == 'details':
                expansion_factor = level
            else:
                expansion_factor = 1
            n_features = input_channels // expansion_factor
        
        return DataShape(
            input_channels=input_channels or 1,
            output_channels=output_channels or 1,
            sequence_length=sequence_length,
            horizon=horizon,
            n_features=n_features,
            n_targets=n_targets,
            feature_indices=feature_indices,
            target_indices=target_indices
        )
    
    def auto_fix_config(self, 
                       data: np.ndarray,
                       config: Dict[str, Any],
                       target_shape: DataShape) -> Dict[str, Any]:
        """自动修复配置以匹配目标形状"""
        fixed_config = config.copy()
        
        # 修复数据配置
        data_cfg = fixed_config.setdefault('data', {})
        
        # 设置特征和目标索引
        if target_shape.feature_indices is not None:
            data_cfg['feature_indices'] = target_shape.feature_indices
        if target_shape.target_indices is not None:
            data_cfg['target_indices'] = target_shape.target_indices
            
        data_cfg['sequence_length'] = target_shape.sequence_length
        
        # 修复模型配置
        model_cfg = fixed_config.setdefault('model', {})
        model_cfg['forecast_horizon'] = target_shape.horizon
        
        # 如果需要小波变换来匹配输入通道数
        current_features = len(target_shape.feature_indices) if target_shape.feature_indices else data.shape[1]
        if target_shape.input_channels > current_features:
            ratio = target_shape.input_channels // current_features
            if ratio > 1 and target_shape.input_channels % current_features == 0:
                # 启用小波变换
                wavelet_cfg = data_cfg.setdefault('wavelet', {})
                wavelet_cfg['enabled'] = True
                level = ratio - 1
                wavelet_cfg['level'] = max(1, level)
                wavelet_cfg['take'] = 'all'
                wavelet_cfg.setdefault('wavelet', 'db4')
                wavelet_cfg.setdefault('mode', 'symmetric')
        
        return fixed_config