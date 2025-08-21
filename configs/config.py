from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import copy


def _try_load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "YAML config requested but PyYAML is not installed. Install with: pip install pyyaml"
        ) from e
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class CNNLayerConfig:
    out_channels: int
    kernel_size: int = 3
    stride: int = 1
    padding: Optional[int] = None  # 如果为 None 表示 same padding
    dilation: int = 1
    activation: str = "relu"
    pool: Optional[str] = "max"  # "max", "avg", 或 None
    pool_kernel_size: int = 2
    # 为支持高级 CNN 结构（如 dilated），允许在层级上提供可选的扩张率列表
    dilation_rates: Optional[List[int]] = None


@dataclass
class CNNConfig:
    # CNN 变体：standard|depthwise|dilated
    variant: str = "standard"
    layers: List[CNNLayerConfig] = field(default_factory=lambda: [
        CNNLayerConfig(out_channels=32, kernel_size=5),
        CNNLayerConfig(out_channels=64, kernel_size=3),
    ])
    dropout: float = 0.1
    use_batchnorm: bool = True


@dataclass
class LSTMConfig:
    hidden_size: int = 128
    num_layers: int = 2
    bidirectional: bool = True
    dropout: float = 0.1
    rnn_type: str = "lstm"  # lstm|gru


@dataclass
class AttentionConfig:
    enabled: bool = True
    # 注意力变体：standard|multiscale
    variant: str = "standard"
    num_heads: int = 4
    dropout: float = 0.1
    add_positional_encoding: bool = False
    # 多尺度注意力的可选参数
    multiscale_scales: Optional[List[int]] = field(default_factory=lambda: [1, 2])
    multiscale_fuse: str = "sum"  # "sum" 或 "concat"


@dataclass
class TCNLayerConfig:
    out_channels: int
    kernel_size: int = 3
    dilation: Optional[int] = None  # 若为 None，将按 1,2,4,... 自动推断
    activation: str = "relu"       # relu|gelu|silu
    use_weightnorm: bool = True


@dataclass
class TCNConfig:
    enabled: bool = False
    layers: List[TCNLayerConfig] = field(default_factory=lambda: [
        TCNLayerConfig(out_channels=32, kernel_size=3),
        TCNLayerConfig(out_channels=64, kernel_size=3),
    ])
    dropout: float = 0.0
    use_batchnorm: bool = True


@dataclass
class ModelConfig:
    cnn: CNNConfig = field(default_factory=CNNConfig)
    tcn: TCNConfig = field(default_factory=TCNConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    fc_hidden: int = 128
    forecast_horizon: int = 1


@dataclass
class OptimizerConfig:
    name: str = "adam"  # adam, sgd, adamw
    lr: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9  # used for SGD
    betas: Optional[List[float]] = None  # used for Adam/AdamW


@dataclass
class SchedulerConfig:
    name: Optional[str] = "cosine"  # cosine, step, plateau, or None
    step_size: int = 10
    gamma: float = 0.5
    T_max: int = 50
    reduce_on_plateau_mode: str = "min"
    reduce_on_plateau_patience: int = 5


@dataclass
class EarlyStoppingConfig:
    enabled: bool = True
    patience: int = 10
    min_delta: float = 0.0


@dataclass
class CheckpointConfig:
    dir: str = "checkpoints"
    save_best_only: bool = True
    monitor: str = "val_loss"
    # 额外导出最佳模型的目录；若为 None 则不额外导出
    export_best_dir: Optional[str] = None


@dataclass
class DataConfig:
    data_path: Optional[str] = None  # path to CSV or NPZ
    sequence_length: int = 64
    horizon: int = 1
    feature_indices: Optional[List[int]] = None  # select subset of features
    target_indices: Optional[List[int]] = None  # default last one
    train_split: float = 0.7
    val_split: float = 0.15
    normalize: str = "standard"  # "standard", "minmax", or "none"
    batch_size: int = 64
    num_workers: int = 0
    shuffle_train: bool = True
    drop_last: bool = False


@dataclass
class TrainingConfig:
    epochs: int = 50
    loss: str = "mse"  # mse, mae, huber
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    checkpoints: CheckpointConfig = field(default_factory=CheckpointConfig)
    gradient_clip: Optional[float] = 1.0
    mixed_precision: bool = False
    log_dir: str = "runs"
    seed: int = 42
    print_every: int = 50
    deterministic: Optional[bool] = None
    cudnn_benchmark: Optional[bool] = None
    matmul_precision: Optional[str] = None  # e.g., "high", "medium"


@dataclass
class FullConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    device: Optional[str] = None  # auto if None
    output_dir: Optional[str] = None  # base dir for all outputs
    visual_save_dir: str = "image"
    visual_enabled: bool = True

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "FullConfig":
        def map_dataclass(dc_cls, d):
            if d is None:
                return dc_cls()
            if not isinstance(d, dict):
                return d
            fields = {}
            for k, v in d.items():
                fields[k] = v
            return dc_cls(**fields)

        model = cfg.get("model", {})
        cnn = map_dataclass(CNNConfig, model.get("cnn"))
        # convert cnn layers
        if isinstance(cnn.layers, list):
            cnn.layers = [
                CNNLayerConfig(**(l if isinstance(l, dict) else asdict(l)))
                for l in cnn.layers
            ]
        # TCN config
        tcn = map_dataclass(TCNConfig, model.get("tcn"))
        if isinstance(tcn.layers, list):
            tcn.layers = [
                TCNLayerConfig(**(l if isinstance(l, dict) else asdict(l)))
                for l in tcn.layers
            ]
        lstm = map_dataclass(LSTMConfig, model.get("lstm"))
        attention = map_dataclass(AttentionConfig, model.get("attention"))
        model_cfg = ModelConfig(cnn=cnn, tcn=tcn, lstm=lstm, attention=attention,
                                fc_hidden=model.get("fc_hidden", 128),
                                forecast_horizon=model.get("forecast_horizon", 1))

        data_cfg = map_dataclass(DataConfig, cfg.get("data"))
        opt_cfg = map_dataclass(OptimizerConfig, cfg.get("train", {}).get("optimizer"))
        sch_cfg = map_dataclass(SchedulerConfig, cfg.get("train", {}).get("scheduler"))
        es_cfg = map_dataclass(EarlyStoppingConfig, cfg.get("train", {}).get("early_stopping"))
        ckpt_cfg = map_dataclass(CheckpointConfig, cfg.get("train", {}).get("checkpoints"))
        train_cfg = TrainingConfig(
            epochs=cfg.get("train", {}).get("epochs", 50),
            loss=cfg.get("train", {}).get("loss", "mse"),
            optimizer=opt_cfg,
            scheduler=sch_cfg,
            early_stopping=es_cfg,
            checkpoints=ckpt_cfg,
            gradient_clip=cfg.get("train", {}).get("gradient_clip", 1.0),
            mixed_precision=cfg.get("train", {}).get("mixed_precision", False),
            log_dir=cfg.get("train", {}).get("log_dir", "runs"),
            seed=cfg.get("train", {}).get("seed", 42),
            print_every=cfg.get("train", {}).get("print_every", 50),
            deterministic=cfg.get("train", {}).get("deterministic", None),
            cudnn_benchmark=cfg.get("train", {}).get("cudnn_benchmark", None),
            matmul_precision=cfg.get("train", {}).get("matmul_precision", None),
        )
        return FullConfig(model=model_cfg, data=data_cfg, train=train_cfg, device=cfg.get("device"),
                          output_dir=cfg.get("output_dir", None),
                          visual_save_dir=cfg.get("visual_save_dir", "image"),
                          visual_enabled=cfg.get("visual_enabled", True))


def load_config(path: Optional[str]) -> FullConfig:
    if path is None:
        return FullConfig()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    lower = path.lower()
    if lower.endswith(".json"):
        cfg = _load_json(path)
    elif lower.endswith(".yaml") or lower.endswith(".yml"):
        cfg = _try_load_yaml(path)
    else:
        raise ValueError("Unsupported config format. Use .json or .yaml")
    return FullConfig.from_dict(cfg)


def to_dict(cfg: FullConfig) -> Dict[str, Any]:
    return asdict(cfg)

# ============ 严格评估所需：训练端规范化配置保存与校验 ============

def canonicalize_for_checkpoint(cfg: 'FullConfig') -> 'FullConfig':
    """生成一个配置副本，填充训练时实际使用/期望使用的关键字段，确保评估端严格解析。
    - 当使用 TCN：确保 model.tcn.enabled=True 且 tcn.layers 完整，dilation 显式为整型；同时同步 cnn.variant='tcn'
    - 当使用 dilated CNN：每层 dilation_rates 显式存在
    - 注意力 multiscale：scales/fuse 显式存在
    - 统一 add_positional_encoding 字段名
    """
    c = copy.deepcopy(cfg)
    m = c.model

    # 统一注意力字段名
    try:
        _ = m.attention.add_positional_encoding
    except Exception:
        try:
            m.attention.add_positional_encoding = bool(getattr(m.attention, 'add_posional_encoding'))
        except Exception:
            m.attention.add_positional_encoding = False

    cnn_variant = str(getattr(m.cnn, 'variant', 'standard')).lower()
    is_tcn = bool(getattr(m.tcn, 'enabled', False) or cnn_variant == 'tcn')

    if is_tcn:
        # 显式开启并同步 variant
        m.tcn.enabled = True
        m.cnn.variant = 'tcn'
        layers = list(getattr(m.tcn, 'layers', []) or [])
        if len(layers) == 0:
            # 从 cnn.layers 转换
            default_d = 1
            converted: List[TCNLayerConfig] = []
            for l in m.cnn.layers:
                d = getattr(l, 'dilation', None)
                dil = default_d if d is None else int(d)
                default_d = max(1, int(dil) * 2)
                converted.append(TCNLayerConfig(
                    out_channels=int(l.out_channels),
                    kernel_size=int(getattr(l, 'kernel_size', 3)),
                    dilation=int(dil),
                    activation=str(getattr(l, 'activation', 'relu')),
                    use_weightnorm=True,
                ))
            m.tcn.layers = converted
        else:
            # 修正 None dilation 为递增显式值
            default_d = 1
            fixed: List[TCNLayerConfig] = []
            for tl in layers:
                d = getattr(tl, 'dilation', None)
                dil = default_d if d is None else int(d)
                default_d = max(1, int(dil) * 2)
                fixed.append(TCNLayerConfig(
                    out_channels=int(tl.out_channels),
                    kernel_size=int(getattr(tl, 'kernel_size', 3)),
                    dilation=int(dil),
                    activation=str(getattr(tl, 'activation', 'relu')),
                    use_weightnorm=bool(getattr(tl, 'use_weightnorm', True)),
                ))
            m.tcn.layers = fixed

    if str(getattr(m.cnn, 'variant', 'standard')).lower() == 'dilated':
        fixed_layers: List[CNNLayerConfig] = []
        for l in m.cnn.layers:
            rates = getattr(l, 'dilation_rates', None) or [1, 2, 4]
            rates = [int(x) for x in rates]
            fixed = CNNLayerConfig(
                out_channels=int(l.out_channels),
                kernel_size=int(getattr(l, 'kernel_size', 3)),
                stride=int(getattr(l, 'stride', 1)),
                padding=getattr(l, 'padding', None),
                dilation=int(getattr(l, 'dilation', 1)),
                activation=str(getattr(l, 'activation', 'relu')),
                pool=getattr(l, 'pool', 'max'),
                pool_kernel_size=int(getattr(l, 'pool_kernel_size', 2)),
            )
            fixed.dilation_rates = rates
            fixed_layers.append(fixed)
        m.cnn.layers = fixed_layers

    if str(getattr(m.attention, 'variant', 'standard')).lower() == 'multiscale':
        if not getattr(m.attention, 'multiscale_scales', None):
            m.attention.multiscale_scales = [1, 2]
        if not getattr(m.attention, 'multiscale_fuse', None):
            m.attention.multiscale_fuse = 'sum'

    return c


def validate_full_config_strict(cfg: 'FullConfig') -> None:
    """训练端严格校验：缺失/None/错误类型直接抛错，避免保存无效 cfg。"""
    m = cfg.model
    # CNN/TCN 骨干
    if str(getattr(m.cnn, 'variant', 'standard')).lower() == 'tcn' or getattr(m.tcn, 'enabled', False):
        if not getattr(m.tcn, 'layers', None):
            raise ValueError('TCN 启用但缺少 model.tcn.layers')
        for i, l in enumerate(m.tcn.layers):
            if l.out_channels is None or l.kernel_size is None or l.dilation is None:
                raise ValueError(f'TCN 第{i}层缺少 out_channels/kernel_size/dilation')
    else:
        if not getattr(m.cnn, 'layers', None):
            raise ValueError('CNN 缺少 model.cnn.layers')
        if str(getattr(m.cnn, 'variant', 'standard')).lower() == 'dilated':
            for i, l in enumerate(m.cnn.layers):
                if not getattr(l, 'dilation_rates', None):
                    raise ValueError(f'dilated CNN 第{i}层缺少 dilation_rates')
    # RNN
    if m.lstm.rnn_type not in ('lstm','gru'):
        raise ValueError('lstm.rnn_type 必须为 lstm 或 gru')
    for k in ['hidden_size','num_layers','bidirectional','dropout']:
        if getattr(m.lstm, k, None) is None:
            raise ValueError(f'lstm.{k} 不能为空')
    # Attention
    if m.attention.enabled:
        if m.attention.variant not in ('standard','multiscale'):
            raise ValueError('attention.variant 必须为 standard 或 multiscale')
        if m.attention.num_heads is None or m.attention.dropout is None:
            raise ValueError('attention.num_heads/dropout 不能为空')
        if not hasattr(m.attention, 'add_positional_encoding'):
            raise ValueError('attention 缺少 add_positional_encoding 字段')
        if m.attention.variant == 'multiscale':
            if not getattr(m.attention, 'multiscale_scales', None):
                raise ValueError('attention.multiscale_scales 不能为空')
            if not getattr(m.attention, 'multiscale_fuse', None):
                raise ValueError('attention.multiscale_fuse 不能为空')


