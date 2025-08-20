from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


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
class ModelConfig:
    cnn: CNNConfig = field(default_factory=CNNConfig)
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
        lstm = map_dataclass(LSTMConfig, model.get("lstm"))
        attention = map_dataclass(AttentionConfig, model.get("attention"))
        model_cfg = ModelConfig(cnn=cnn, lstm=lstm, attention=attention,
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

