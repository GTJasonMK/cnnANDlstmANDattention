#!/usr/bin/env python3
"""
🔬 优化后的消融实验YAML生成器
严格遵循单因子变化原则，确保科学的消融实验设计

主要改进：
1. ✅ 修复控制变量污染问题
2. ✅ 重新设计CA消融实验逻辑
3. ✅ 添加缺失的核心消融因子
4. ✅ 引入多种子统计验证
5. ✅ 简化和修正wavelet实验
6. ✅ 参数化数据索引配置
7. ✅ 实现分层实验策略
8. ✅ 改进实验命名和组织
"""

from __future__ import annotations
import argparse
import copy
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import yaml  # type: ignore
from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    """实验配置数据类，确保类型安全"""
    factor: str
    value: Any
    baseline: Dict[str, Any]
    note: str
    tier: str = "core"  # core, optimization, hyperparams, statistical

# 🔧 默认配置模板 - 移除所有预设偏好
TCN_DEFAULT: Dict[str, Any] = {
    'enabled': True,
    'dropout': 0.05,
    'use_batchnorm': True,
    'layers': [
        {'out_channels': 64, 'kernel_size': 3, 'dilation': 1, 'activation': 'gated', 'use_weightnorm': True},
        {'out_channels': 128, 'kernel_size': 3, 'dilation': 2, 'activation': 'gated', 'use_weightnorm': True},
        {'out_channels': 128, 'kernel_size': 3, 'dilation': 4, 'activation': 'gated', 'use_weightnorm': True},
    ],
}

# 🎯 纯净的基准模板 - 无任何架构偏好
CLEAN_TEMPLATE: Dict[str, Any] = {
    'device': None,
    'model': {
        'fc_hidden': 128,
        'forecast_horizon': 3,
        'cnn': {
            'variant': 'standard',  # 最简单的baseline
            'dropout': 0.1,
            'use_batchnorm': True,
            'use_channel_attention': False,  # 🔥 默认关闭CA，避免污染
            'layers': [
                {'out_channels': 32, 'kernel_size': 5, 'activation': 'relu', 'pool': 'max', 'pool_kernel_size': 2},
                {'out_channels': 64, 'kernel_size': 3, 'activation': 'gelu', 'pool': 'max', 'pool_kernel_size': 2},
            ],
        },
        'lstm': {
            'rnn_type': 'lstm',
            'hidden_size': 128,
            'num_layers': 2,
            'bidirectional': True,
            'dropout': 0.1,
        },
        'attention': {
            'enabled': True,
            'variant': 'standard',
            'num_heads': 4,
            'dropout': 0.1,
            'add_positional_encoding': False,  # 🔥 默认关闭位置编码
            'positional_mode': 'none',
        },
        'normalization': {
            'revin': {'enabled': False}  # 🔥 默认关闭RevIN
        },
        'decomposition': {
            'enabled': False  # 🔥 默认关闭分解
        }
    },
    'data': {
        'data_path': 'CHANGE_ME',
        'sequence_length': 64,
        'horizon': 3,
        'feature_indices': 'CHANGE_ME',  # 🔥 参数化
        'target_indices': 'CHANGE_ME',   # 🔥 参数化
        'train_split': 0.7,
        'val_split': 0.15,
        'normalize': 'minmax',  # 🔥 固定使用minmax，无需消融实验
        'batch_size': 64,
        'num_workers': 0,
        'shuffle_train': True,
        'drop_last': False,
        'wavelet': {
            'enabled': False,  # 🔥 默认关闭wavelet
            'wavelet': 'db4',
            'level': 3,
            'mode': 'symmetric',
            'take': 'approx',
            'resample_method': 'adaptive',
        },
    },
    'train': {
        'epochs': 50,
        'loss': 'mse',
        'optimizer': {'name': 'adam', 'lr': 0.001, 'weight_decay': 0.0001},
        'scheduler': {'name': 'cosine', 'T_max': 50},
        'early_stopping': {'enabled': True, 'patience': 20, 'min_delta': 0.001},
        'checkpoints': {'dir': 'checkpoints_o', 'save_best_only': True, 'export_best_dir': ''},
        'gradient_clip': 1.0,
        'mixed_precision': False,
        'log_dir': 'runs',
        'seed': 42,  # 基准种子，后续会为多种子实验修改
        'print_every': 50,
    },
}

def write_yaml(config: Dict[str, Any], out_path: Path) -> None:
    """安全写入YAML文件"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding='utf-8')

def create_clean_baseline(
    data_path: str,
    feature_indices: List[int],
    target_indices: List[int],
    export_root: Path,
    epochs: Optional[int] = None,
    seed: int = 42
) -> Dict[str, Any]:
    """创建完全纯净的基准配置，无任何架构偏好"""
    cfg = copy.deepcopy(CLEAN_TEMPLATE)

    # 数据配置
    cfg['data']['data_path'] = data_path
    cfg['data']['feature_indices'] = feature_indices
    cfg['data']['target_indices'] = target_indices

    # 训练配置
    cfg['train']['seed'] = seed
    if epochs is not None:
        cfg['train']['epochs'] = int(epochs)
        if cfg['train'].get('scheduler', {}).get('name') == 'cosine':
            cfg['train']['scheduler']['T_max'] = int(epochs)

    # 导出配置
    cfg['train']['checkpoints']['export_best_dir'] = str(export_root)

    return cfg

def apply_single_factor_change(base_config: Dict[str, Any], factor: str, value: Any) -> Dict[str, Any]:
    """🔥 核心函数：严格应用单因子变化，确保其他因子不受影响"""
    cfg = copy.deepcopy(base_config)

    # 🎯 架构因子
    if factor == 'cnn_variant':
        cfg['model']['cnn']['variant'] = value
        # TCN特殊处理
        if value == 'tcn':
            cfg['model']['tcn'] = copy.deepcopy(TCN_DEFAULT)
        else:
            if 'tcn' in cfg['model']:
                cfg['model']['tcn']['enabled'] = False

    elif factor == 'rnn_type':
        cfg['model']['lstm']['rnn_type'] = value

    elif factor == 'attention_variant':
        cfg['model']['attention']['variant'] = value
        # spatiotemporal特殊配置
        if value == 'spatiotemporal':
            cfg['model']['attention']['st_mode'] = 'serial'
            cfg['model']['attention']['st_fuse'] = 'sum'

    # 🔧 优化因子
    elif factor == 'channel_attention':
        # 🔥 关键修复：CA实验必须在支持CA的CNN上进行
        ca_supported_cnns = ['depthwise', 'dilated', 'inception']
        current_cnn = cfg['model']['cnn']['variant']

        if value == 'off':
            cfg['model']['cnn']['use_channel_attention'] = False
            # 如果当前CNN不支持CA，保持CNN不变（因为CA本来就是off）
        else:  # 'eca' or 'se'
            # 如果当前CNN不支持CA，先切换到depthwise
            if current_cnn not in ca_supported_cnns:
                raise ValueError(f"CA实验必须在支持CA的CNN({ca_supported_cnns})上进行，当前CNN: {current_cnn}")
            cfg['model']['cnn']['use_channel_attention'] = True
            cfg['model']['cnn']['channel_attention_type'] = value

    elif factor == 'positional_encoding':
        cfg['model']['attention']['positional_mode'] = value
        cfg['model']['attention']['add_positional_encoding'] = (value != 'none')

    # 🎛️ 超参数因子
    elif factor == 'learning_rate':
        cfg['train']['optimizer']['lr'] = value

    elif factor == 'batch_size':
        cfg['data']['batch_size'] = value

    elif factor == 'sequence_length':
        cfg['data']['sequence_length'] = value

    elif factor == 'hidden_size':
        cfg['model']['lstm']['hidden_size'] = value
        cfg['model']['fc_hidden'] = value  # 保持一致

    elif factor == 'num_layers':
        cfg['model']['lstm']['num_layers'] = value

    elif factor == 'attention_heads':
        cfg['model']['attention']['num_heads'] = value

    # 🔬 预处理因子
    elif factor == 'revin':
        cfg['model']['normalization']['revin']['enabled'] = value

    elif factor == 'decomposition':
        cfg['model']['decomposition']['enabled'] = value

    elif factor == 'wavelet':
        cfg['data']['wavelet']['enabled'] = value

    elif factor == 'wavelet_base':
        cfg['data']['wavelet']['enabled'] = True
        cfg['data']['wavelet']['wavelet'] = value

    elif factor == 'wavelet_level':
        cfg['data']['wavelet']['enabled'] = True
        cfg['data']['wavelet']['level'] = value

    elif factor == 'wavelet_take':
        cfg['data']['wavelet']['enabled'] = True
        cfg['data']['wavelet']['take'] = value

    elif factor == 'seed':
        cfg['train']['seed'] = value

    else:
        raise ValueError(f"未知的因子: {factor}")

    return cfg

def generate_tier1_core_experiments(
    base_config: Dict[str, Any],
    experiment_configs: List[ExperimentConfig]
) -> None:
    """生成一级核心架构实验 - 最重要的消融"""

    # RNN类型消融
    for rnn_type in ['lstm', 'gru', 'ssm']:
        if rnn_type != 'lstm':  # lstm是baseline
            experiment_configs.append(ExperimentConfig(
                factor='rnn_type',
                value=rnn_type,
                baseline=base_config,
                note=f"核心消融: RNN类型 = {rnn_type}",
                tier='tier1_core'
            ))

    # CNN架构消融
    for cnn_variant in ['depthwise', 'dilated', 'inception', 'tcn']:
        experiment_configs.append(ExperimentConfig(
            factor='cnn_variant',
            value=cnn_variant,
            baseline=base_config,
            note=f"核心消融: CNN架构 = {cnn_variant}",
            tier='tier1_core'
        ))

    # Attention变体消融
    for attn_variant in ['multiscale', 'local', 'conformer', 'spatiotemporal']:
        experiment_configs.append(ExperimentConfig(
            factor='attention_variant',
            value=attn_variant,
            baseline=base_config,
            note=f"核心消融: Attention变体 = {attn_variant}",
            tier='tier1_core'
        ))

def generate_tier2_optimization_experiments(
    base_config: Dict[str, Any],
    experiment_configs: List[ExperimentConfig]
) -> None:
    """生成二级优化策略实验"""

    # 🔥 修复后的CA实验：确保在支持CA的CNN上进行
    ca_supported_cnns = ['depthwise', 'dilated', 'inception']

    for cnn_variant in ca_supported_cnns:
        # 先创建该CNN的基准配置
        cnn_base = apply_single_factor_change(base_config, 'cnn_variant', cnn_variant)

        # CA关闭实验
        experiment_configs.append(ExperimentConfig(
            factor='channel_attention',
            value='off',
            baseline=cnn_base,
            note=f"优化策略消融: CA=关闭 (CNN={cnn_variant})",
            tier='tier2_optimization'
        ))

        # CA ECA实验
        experiment_configs.append(ExperimentConfig(
            factor='channel_attention',
            value='eca',
            baseline=cnn_base,
            note=f"优化策略消融: CA=ECA (CNN={cnn_variant})",
            tier='tier2_optimization'
        ))

        # CA SE实验
        experiment_configs.append(ExperimentConfig(
            factor='channel_attention',
            value='se',
            baseline=cnn_base,
            note=f"优化策略消融: CA=SE (CNN={cnn_variant})",
            tier='tier2_optimization'
        ))

    # 位置编码消融
    for pos_encoding in ['absolute', 'alibi', 'rope']:
        experiment_configs.append(ExperimentConfig(
            factor='positional_encoding',
            value=pos_encoding,
            baseline=base_config,
            note=f"优化策略消融: 位置编码 = {pos_encoding}",
            tier='tier2_optimization'
        ))

    # 预处理消融
    experiment_configs.append(ExperimentConfig(
        factor='revin', value=True, baseline=base_config,
        note="优化策略消融: RevIN = 开启", tier='tier2_optimization'
    ))

    experiment_configs.append(ExperimentConfig(
        factor='decomposition', value=True, baseline=base_config,
        note="优化策略消融: 分解 = 开启", tier='tier2_optimization'
    ))

def generate_tier3_hyperparams_experiments(
    base_config: Dict[str, Any],
    experiment_configs: List[ExperimentConfig]
) -> None:
    """生成三级超参数实验"""

    # 学习率消融
    for lr in [0.0001, 0.01]:  # 0.001是baseline
        experiment_configs.append(ExperimentConfig(
            factor='learning_rate', value=lr, baseline=base_config,
            note=f"超参数消融: 学习率 = {lr}", tier='tier3_hyperparams'
        ))

    # 批次大小消融
    for batch_size in [32, 128]:  # 64是baseline
        experiment_configs.append(ExperimentConfig(
            factor='batch_size', value=batch_size, baseline=base_config,
            note=f"超参数消融: 批次大小 = {batch_size}", tier='tier3_hyperparams'
        ))

    # 序列长度消融
    for seq_len in [32, 128]:  # 64是baseline
        experiment_configs.append(ExperimentConfig(
            factor='sequence_length', value=seq_len, baseline=base_config,
            note=f"超参数消融: 序列长度 = {seq_len}", tier='tier3_hyperparams'
        ))

    # 模型容量消融
    for hidden_size in [64, 256]:  # 128是baseline
        experiment_configs.append(ExperimentConfig(
            factor='hidden_size', value=hidden_size, baseline=base_config,
            note=f"超参数消融: 隐藏层大小 = {hidden_size}", tier='tier3_hyperparams'
        ))

    # 层数消融
    for num_layers in [1, 3]:  # 2是baseline
        experiment_configs.append(ExperimentConfig(
            factor='num_layers', value=num_layers, baseline=base_config,
            note=f"超参数消融: LSTM层数 = {num_layers}", tier='tier3_hyperparams'
        ))

    # 注意力头数消融
    for num_heads in [2, 8]:  # 4是baseline
        experiment_configs.append(ExperimentConfig(
            factor='attention_heads', value=num_heads, baseline=base_config,
            note=f"超参数消融: 注意力头数 = {num_heads}", tier='tier3_hyperparams'
        ))

def generate_wavelet_experiments(
    base_config: Dict[str, Any],
    experiment_configs: List[ExperimentConfig],
    wavelet_bases: List[str],
    wavelet_levels: List[int],
    wavelet_takes: List[str]
) -> None:
    """生成简化的wavelet消融实验 - 严格单因子变化"""

    # Wavelet开关消融
    experiment_configs.append(ExperimentConfig(
        factor='wavelet', value=True, baseline=base_config,
        note="Wavelet消融: 开启 vs 关闭", tier='tier2_optimization'
    ))

    # Wavelet基函数消融（在开启wavelet的基础上）
    wavelet_base_config = apply_single_factor_change(base_config, 'wavelet', True)
    for base in wavelet_bases:
        if base != 'db4':  # db4是默认baseline
            experiment_configs.append(ExperimentConfig(
                factor='wavelet_base', value=base, baseline=wavelet_base_config,
                note=f"Wavelet消融: 基函数 = {base}", tier='tier2_optimization'
            ))

    # Wavelet分解级别消融
    for level in wavelet_levels:
        if level != 3:  # 3是默认baseline
            experiment_configs.append(ExperimentConfig(
                factor='wavelet_level', value=level, baseline=wavelet_base_config,
                note=f"Wavelet消融: 分解级别 = {level}", tier='tier2_optimization'
            ))

    # Wavelet系数选择消融
    for take in wavelet_takes:
        if take != 'approx':  # approx是默认baseline
            experiment_configs.append(ExperimentConfig(
                factor='wavelet_take', value=take, baseline=wavelet_base_config,
                note=f"Wavelet消融: 系数选择 = {take}", tier='tier2_optimization'
            ))

def generate_multiseed_experiments(
    base_config: Dict[str, Any],
    experiment_configs: List[ExperimentConfig],
    seeds: List[int]
) -> None:
    """生成多种子统计验证实验"""

    for seed in seeds:
        if seed != 42:  # 42是baseline种子
            experiment_configs.append(ExperimentConfig(
                factor='seed', value=seed, baseline=base_config,
                note=f"统计验证: 随机种子 = {seed}", tier='statistical'
            ))

def save_experiment_config(
    exp_config: ExperimentConfig,
    out_dir: Path,
    multiseed: bool = False
) -> None:
    """保存单个实验配置"""

    # 应用因子变化
    final_config = apply_single_factor_change(exp_config.baseline, exp_config.factor, exp_config.value)

    # 添加元数据
    final_config['experiment_metadata'] = {
        'factor': exp_config.factor,
        'value': exp_config.value,
        'tier': exp_config.tier,
        'note': exp_config.note,
        'is_multiseed': multiseed
    }

    # 🔥 修复：生成文件名并统一保存到同一目录
    if multiseed:
        # 统计验证也添加tier前缀
        filename = f"statistical_seed_{exp_config.value:03d}__baseline_multiseed.yaml"
    else:
        safe_value = str(exp_config.value).replace('/', '_').replace(' ', '_')
        # 🔥 关键修改：将tier信息添加到文件名中，避免冲突
        tier_prefix = exp_config.tier.replace('tier', 't').replace('_', '')  # tier1_core -> t1core
        filename = f"{tier_prefix}_{exp_config.factor}-{safe_value}__ablation.yaml"

    # 🔥 关键修改：所有文件都保存到同一个目录
    target_dir = out_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置
    write_yaml(final_config, target_dir / filename)

def main():
    parser = argparse.ArgumentParser(
        description='🔬 优化后的消融实验YAML生成器 - 严格遵循单因子变化原则'
    )

    # 必需参数
    parser.add_argument('--data-path', required=True, help='数据集路径')
    parser.add_argument('--export-best-root', required=True, help='最优模型导出根目录')
    parser.add_argument('--feature-indices', required=True, help='特征列索引，逗号分隔，例如: 1,2,3,4,5')
    parser.add_argument('--target-indices', required=True, help='目标列索引，逗号分隔，例如: 10,11')

    # 可选参数
    parser.add_argument('--out-dir', default='configs/ablation', help='输出目录')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--tiers', default='1,2,3', help='要生成的实验层级，逗号分隔: 1=核心,2=优化,3=超参数')

    # 多种子统计验证
    parser.add_argument('--enable-multiseed', action='store_true', help='启用多种子统计验证')
    parser.add_argument('--seeds', default='42,123,456,789,999', help='多种子列表，逗号分隔')

    # Wavelet参数
    parser.add_argument('--enable-wavelet', action='store_true', help='启用Wavelet消融实验')
    parser.add_argument('--wavelet-bases', default='db4,sym5,coif3', help='Wavelet基函数列表')
    parser.add_argument('--wavelet-levels', default='2,3,4', help='Wavelet分解级别列表')
    parser.add_argument('--wavelet-takes', default='approx,details,all', help='Wavelet系数选择列表')

    args = parser.parse_args()

    # 解析参数
    out_dir = Path(args.out_dir)
    export_root = Path(args.export_best_root)
    feature_indices = [int(x.strip()) for x in args.feature_indices.split(',')]
    target_indices = [int(x.strip()) for x in args.target_indices.split(',')]
    tiers = [int(x.strip()) for x in args.tiers.split(',')]
    seeds = [int(x.strip()) for x in args.seeds.split(',')]

    # Wavelet参数
    wavelet_bases = [x.strip() for x in args.wavelet_bases.split(',')]
    wavelet_levels = [int(x.strip()) for x in args.wavelet_levels.split(',')]
    wavelet_takes = [x.strip() for x in args.wavelet_takes.split(',')]

    # 创建基准配置
    base_config = create_clean_baseline(
        data_path=args.data_path,
        feature_indices=feature_indices,
        target_indices=target_indices,
        export_root=export_root,
        epochs=args.epochs,
        seed=42
    )

    # 保存基准配置
    out_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(base_config, out_dir / "baseline__clean_reference.yaml")

    # 生成实验配置列表
    experiment_configs: List[ExperimentConfig] = []

    # 一级核心实验
    if 1 in tiers:
        generate_tier1_core_experiments(base_config, experiment_configs)
        print(f"[INFO] 生成一级核心实验: {sum(1 for e in experiment_configs if e.tier == 'tier1_core')} 个")

    # 二级优化实验
    if 2 in tiers:
        generate_tier2_optimization_experiments(base_config, experiment_configs)
        if args.enable_wavelet:
            generate_wavelet_experiments(base_config, experiment_configs, wavelet_bases, wavelet_levels, wavelet_takes)
        print(f"[INFO] 生成二级优化实验: {sum(1 for e in experiment_configs if e.tier == 'tier2_optimization')} 个")

    # 三级超参数实验
    if 3 in tiers:
        generate_tier3_hyperparams_experiments(base_config, experiment_configs)
        print(f"[INFO] 生成三级超参数实验: {sum(1 for e in experiment_configs if e.tier == 'tier3_hyperparams')} 个")

    # 多种子统计验证
    if args.enable_multiseed:
        generate_multiseed_experiments(base_config, experiment_configs, seeds)
        print(f"[INFO] 生成多种子验证实验: {len(seeds)-1} 个")

    # 保存所有实验配置
    for exp_config in experiment_configs:
        save_experiment_config(exp_config, out_dir, multiseed=(exp_config.tier == 'statistical'))

    # 生成实验总结
    summary = {
        'total_experiments': len(experiment_configs) + 1,  # +1 for baseline
        'baseline_config': str(out_dir / "baseline__clean_reference.yaml"),
        'tier_breakdown': {
            'tier1_core': sum(1 for e in experiment_configs if e.tier == 'tier1_core'),
            'tier2_optimization': sum(1 for e in experiment_configs if e.tier == 'tier2_optimization'),
            'tier3_hyperparams': sum(1 for e in experiment_configs if e.tier == 'tier3_hyperparams'),
            'statistical': sum(1 for e in experiment_configs if e.tier == 'statistical'),
        },
        'experiment_factors': list(set(e.factor for e in experiment_configs)),
        'generated_tiers': tiers,
        'multiseed_enabled': args.enable_multiseed,
        'wavelet_enabled': args.enable_wavelet,
    }

    write_yaml(summary, out_dir / "experiment_summary.yaml")

    print(f"\n[INFO] 消融实验生成完成!")
    print(f"[INFO] 输出目录: {out_dir}")
    print(f"[INFO] 总实验数: {summary['total_experiments']}")
    print(f"[INFO] 实验层级分布: {summary['tier_breakdown']}")
    print(f"[INFO] 测试因子: {', '.join(summary['experiment_factors'])}")
    print(f"\n[INFO] 使用说明:")
    print(f"1. 基准配置: {summary['baseline_config']}")
    print(f"2. [FIXED] 所有实验配置统一保存在: {out_dir}/")
    print(f"3. 文件命名规则: [tier前缀]_[因子]-[值]__ablation.yaml")
    print(f"   - t1core_: 核心架构实验 (CNN/RNN/Attention)")
    print(f"   - t2optimization_: 优化策略实验 (CA/位置编码/预处理)")
    print(f"   - t3hyperparams_: 超参数实验 (学习率/批次大小等)")
    print(f"   - statistical_: 多种子统计验证实验")
    print(f"4. 建议训练顺序: 先t1core，再t2optimization，最后t3hyperparams")


if __name__ == '__main__':
    main()