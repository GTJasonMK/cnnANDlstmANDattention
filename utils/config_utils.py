"""
统一的配置工具
整合重复的配置处理和验证逻辑
"""
from __future__ import annotations

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """统一的YAML配置加载函数"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml_config(config: Dict[str, Any], output_path: str):
    """统一的YAML配置保存函数"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)


def patch_yaml_config(config: Dict[str, Any], 
                     output_dir: Optional[str] = None,
                     data_path: Optional[str] = None,
                     epochs: Optional[int] = None,
                     export_best_dir: Optional[str] = None) -> Dict[str, Any]:
    """统一的YAML配置补丁函数"""
    patched = yaml.safe_load(yaml.dump(config))  # 深拷贝
    
    # 设置输出目录
    if output_dir:
        patched.setdefault('train', {})
        patched['train'].setdefault('checkpoints', {})
        patched['train']['checkpoints']['dir'] = str(Path(output_dir) / 'checkpoints')
        patched['train']['log_dir'] = str(Path(output_dir) / 'runs')
    
    # 设置数据路径
    if data_path:
        patched.setdefault('data', {})
        patched['data']['data_path'] = data_path
    
    # 设置训练轮数
    if epochs is not None:
        patched['train']['epochs'] = int(epochs)
        if patched['train'].get('scheduler', {}).get('name') == 'cosine':
            patched['train']['scheduler']['T_max'] = int(epochs)
    
    # 设置最佳模型导出目录
    if export_best_dir:
        patched['train']['checkpoints']['export_best_dir'] = export_best_dir
    
    return patched


def validate_config_compatibility(config: Dict[str, Any], 
                                data_shape: tuple,
                                strict: bool = False) -> tuple[bool, str]:
    """统一的配置兼容性验证"""
    try:
        model_cfg = config.get('model', {})
        data_cfg = config.get('data', {})
        
        # 检查必需字段
        required_fields = {
            'model.forecast_horizon': model_cfg.get('forecast_horizon'),
            'data.sequence_length': data_cfg.get('sequence_length'),
            'data.horizon': data_cfg.get('horizon')
        }
        
        missing_fields = [k for k, v in required_fields.items() if v is None]
        if missing_fields and strict:
            return False, f"Missing required fields: {missing_fields}"
        
        # 检查数据维度兼容性
        n_samples, n_features = data_shape
        feature_indices = data_cfg.get('feature_indices')
        target_indices = data_cfg.get('target_indices')
        
        if feature_indices:
            invalid_features = [i for i in feature_indices if i >= n_features]
            if invalid_features:
                return False, f"Feature indices out of range: {invalid_features} >= {n_features}"
        
        if target_indices:
            invalid_targets = [i for i in target_indices if i >= n_features] 
            if invalid_targets:
                return False, f"Target indices out of range: {invalid_targets} >= {n_features}"
        
        return True, "Compatible"
        
    except Exception as e:
        return False, f"Validation error: {e}"


def extract_model_info_from_config(config: Dict[str, Any]) -> Dict[str, str]:
    """从配置中提取模型信息用于命名"""
    model_cfg = config.get('model', {})
    
    cnn_variant = model_cfg.get('cnn', {}).get('variant', 'standard')
    tcn_enabled = model_cfg.get('tcn', {}).get('enabled', False)
    if tcn_enabled:
        cnn_variant = 'tcn'
    
    rnn_type = model_cfg.get('lstm', {}).get('rnn_type', 'lstm')
    attn_variant = model_cfg.get('attention', {}).get('variant', 'standard')
    pos_mode = model_cfg.get('attention', {}).get('positional_mode', 'none')
    
    ca_enabled = model_cfg.get('cnn', {}).get('use_channel_attention', False)
    ca_type = model_cfg.get('cnn', {}).get('channel_attention_type', 'eca') if ca_enabled else 'off'
    
    return {
        'cnn_variant': cnn_variant,
        'rnn_type': rnn_type,
        'attn_variant': attn_variant,
        'pos_mode': pos_mode,
        'ca_type': ca_type
    }