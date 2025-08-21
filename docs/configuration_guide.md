## Configuration Guide

This document explains all configuration options available in the project. It reflects the current codebase (configs/config.py and related modules) and marks which options were introduced in optimization phases.

- Phase 1: ECA/SE channel attention, positional modes (ALiBi/RoPE), Local Attention, TCN gated activation
- Phase 2: Inception CNN, Conformer block, RevIN, Trend/Residual decomposition
- Phase 3: SSM (state-space) backbone (lightweight placeholder implemented)

### Table of Contents
- Overview and File Formats
- FullConfig (root)
  - ModelConfig
    - CNNConfig
    - TCNConfig
    - LSTMConfig
    - AttentionConfig
    - Normalization (RevIN)
    - Decomposition (Trend/Residual)
    - Other model-level options
  - DataConfig
  - TrainingConfig
    - OptimizerConfig
    - SchedulerConfig
    - EarlyStoppingConfig
    - CheckpointConfig
  - Global (device/output/visual)
- Recommended Config Combinations
- Notes on Performance and Scenarios
- Examples

---

## Overview and File Formats
- Configs can be YAML (.yaml/.yml) or JSON (.json). The loader chooses by file extension.
- Root structure corresponds to FullConfig dataclass: { model, data, train, device, output_dir, visual_* }.
- See sample configs in configs/yaml/*.yaml

---

## FullConfig (root)
Top-level configuration container.

- model: ModelConfig. Default: see each subsection
- data: DataConfig
- train: TrainingConfig
- device: Optional[str] GPU/CPU selection; if null uses auto
- output_dir: Optional[str] base output dir; if null, defaults per code paths
- visual_save_dir: str default "image"
- visual_enabled: bool default true

Example:
```yaml
model: { ... }
data: { ... }
train: { ... }
device: null
output_dir: null
visual_save_dir: image
visual_enabled: true
```

---

## ModelConfig
Holds all model-related sub-configs and simple scalars.

- cnn: CNNConfig
- tcn: TCNConfig
- lstm: LSTMConfig
- attention: AttentionConfig
- fc_hidden: int, default 128. Hidden size of the prediction head
- forecast_horizon: int, default 1. Number of future steps to predict
- normalization: Dict, default { revin: { enabled: false } } (Phase 2)
- decomposition: Dict, default { enabled: false, method: "ma", kernel: 25, seasonal_period: 24 } (Phase 2)

Example:
```yaml
model:
  fc_hidden: 128
  forecast_horizon: 3
  cnn: { ... }
  lstm: { ... }
  attention: { ... }
  normalization: { revin: { enabled: true } }
  decomposition: { enabled: true, method: ma, kernel: 25 }
```

---

## CNNConfig
Convolutional feature extractor settings.

- variant: str, default "standard". Choices: standard|depthwise|dilated|inception (Phase 2)
  - standard: regular Conv1d blocks
  - depthwise: depthwise-separable conv blocks
  - dilated: dilated conv blocks; per-layer "dilation_rates" allowed
  - inception (Phase 2): multi-branch conv with multiple kernel sizes/dilations
- layers: List[CNNLayerConfig]
- dropout: float, default 0.1
- use_batchnorm: bool, default true
- use_channel_attention: bool, default false (Phase 1)
- channel_attention_type: str, default "eca". Choices: eca|se (Phase 1)

CNNLayerConfig (per-layer):
- out_channels: int (required)
- kernel_size: int, default 3
- stride: int, default 1
- padding: Optional[int], default null => same padding
- dilation: int, default 1
- activation: str, default "relu" (e.g., relu|gelu|silu)
- pool: Optional[str], default "max" (max|avg|null)
- pool_kernel_size: int, default 2
- dilation_rates: Optional[List[int]] (used by dilated variant to specify multi-rate block)
- inception_kernel_sizes: Optional[List[int]] (only when variant=inception)
- inception_dilations: Optional[List[int]] (only when variant=inception)

Example:
```yaml
model:
  cnn:
    variant: inception
    use_channel_attention: true
    channel_attention_type: eca
    layers:
      - { out_channels: 64, kernel_size: 3, activation: relu, inception_kernel_sizes: [3,5,7], inception_dilations: [1,2] }
      - { out_channels: 128, kernel_size: 3, activation: gelu }
```

---

## TCNConfig
Causal temporal convolution network settings (alternative backbone).

- enabled: bool, default false. If true, TCNFeatureExtractor is used (cnn.variant treated as "tcn")
- layers: List[TCNLayerConfig]
- dropout: float, default 0.0
- use_batchnorm: bool, default true

TCNLayerConfig:
- out_channels: int (required)
- kernel_size: int, default 3
- dilation: Optional[int], default null (if null, code inflates as 1,2,4,...)
- activation: str, default "relu". Choices: relu|gelu|silu|gated (Phase 1)
- use_weightnorm: bool, default true

Example:
```yaml
model:
  cnn: { variant: tcn }
  tcn:
    enabled: true
    layers:
      - { out_channels: 64, kernel_size: 3, dilation: 1, activation: gated }
      - { out_channels: 128, kernel_size: 3, dilation: 2, activation: gated }
```

---

## LSTMConfig
Recurrent backbone settings; supports LSTM/GRU/SSM.

- rnn_type: str, default "lstm". Choices: lstm|gru|ssm (Phase 3)
- hidden_size: int, default 128
- num_layers: int, default 2
- bidirectional: bool, default true
- dropout: float, default 0.1 (applied between RNN layers when num_layers>1)

Example:
```yaml
model:
  lstm:
    rnn_type: ssm   # Phase 3 option; lightweight placeholder implemented
    hidden_size: 128
    num_layers: 2
    bidirectional: false
    dropout: 0.05
```

---

## AttentionConfig
Self-attention block settings.

- enabled: bool, default true
- variant: str, default "standard". Choices: standard|multiscale|local|conformer
  - local (Phase 1): Local window attention with dilation
  - conformer (Phase 2): uses ConformerBlock stack internally (lightweight)
- num_heads: int, default 4
- dropout: float, default 0.1
- add_positional_encoding: bool, default false (absolute sinusoidal)
- positional_mode: str, default "none". Choices: none|absolute|alibi|rope (Phase 1)
- multiscale_scales: Optional[List[int]], default [1,2] (for variant=multiscale)
- multiscale_fuse: str, default "sum" (sum|concat) (for multiscale)
- local_window_size: int, default 64 (for variant=local) (Phase 1)
- local_dilation: int, default 1 (for variant=local) (Phase 1)

Examples:
```yaml
model:
  attention:
    enabled: true
    variant: local
    num_heads: 4
    positional_mode: rope
    local_window_size: 64
    local_dilation: 2
```
```yaml
model:
  attention:
    enabled: true
    variant: conformer
    num_heads: 4
    dropout: 0.1
```

---

## Normalization (RevIN) — Phase 2
Applied around the model input (B,T,D). Reversible normalization.

- normalization: Dict
  - revin.enabled: bool, default false

Example:
```yaml
model:
  normalization:
    revin: { enabled: true }
```

---

## Decomposition (Trend/Residual) — Phase 2
Two-branch processing: input is decomposed into residual and trend; both pass through the same backbone and are fused.

- decomposition.enabled: bool, default false
- decomposition.method: str, default "ma". Choices: ma|ets
- decomposition.kernel: int, default 25 (for method=ma)
- decomposition.alpha: float, default 0.3 (for method=ets)

Example:
```yaml
model:
  decomposition:
    enabled: true
    method: ma
    kernel: 25
```

---

## DataConfig
Dataset and loader settings.

- data_path: Optional[str], default null. Path to CSV/NPZ/NPY/NPZ
- sequence_length: int, default 64
- horizon: int, default 1
- feature_indices: Optional[List[int]], default null
- target_indices: Optional[List[int]], default null
- train_split: float, default 0.7
- val_split: float, default 0.15
- normalize: str, default "standard". Choices: standard|minmax|none (data-level normalization)
- batch_size: int, default 64
- num_workers: int, default 0
- shuffle_train: bool, default true
- drop_last: bool, default false

Example:
```yaml
data:
  data_path: data/simulated.csv
  sequence_length: 96
  horizon: 6
  normalize: standard
  batch_size: 128
  num_workers: 4
```

---

## TrainingConfig
Training loop and optimization settings.

- epochs: int, default 50
- loss: str, default "mse". Choices: mse|mae|huber
- optimizer: OptimizerConfig
- scheduler: SchedulerConfig
- early_stopping: EarlyStoppingConfig
- checkpoints: CheckpointConfig
- gradient_clip: Optional[float], default 1.0
- mixed_precision: bool, default false
- log_dir: str, default "runs"
- seed: int, default 42
- print_every: int, default 50
- deterministic: Optional[bool], default null
- cudnn_benchmark: Optional[bool], default null
- matmul_precision: Optional[str], default null (e.g., "high", "medium")

### OptimizerConfig
- name: str, default "adam". Choices: adam|sgd|adamw
- lr: float, default 1e-3
- weight_decay: float, default 1e-4
- momentum: float, default 0.9 (SGD)
- betas: Optional[List[float]] (Adam/AdamW)

### SchedulerConfig
- name: Optional[str], default "cosine". Choices: cosine|step|plateau|null
- step_size: int, default 10 (step)
- gamma: float, default 0.5 (step)
- T_max: int, default 50 (cosine)
- reduce_on_plateau_mode: str, default "min"
- reduce_on_plateau_patience: int, default 5

### EarlyStoppingConfig
- enabled: bool, default true
- patience: int, default 10
- min_delta: float, default 0.0

### CheckpointConfig
- dir: str, default "checkpoints"
- save_best_only: bool, default true
- monitor: str, default "val_loss"
- export_best_dir: Optional[str], default null

Examples:
```yaml
train:
  epochs: 20
  optimizer: { name: adamw, lr: 0.0005, weight_decay: 0.01 }
  scheduler: { name: cosine, T_max: 20 }
  early_stopping: { enabled: true, patience: 5 }
  checkpoints: { dir: checkpoints, save_best_only: true }
  gradient_clip: 1.0
```

---

## Global (device / output / visual)
- device: null|"cpu"|"cuda"|"cuda:0" etc. If null, auto-selects
- output_dir: Optional base output directory (if used by scripts)
- visual_save_dir: str, default "image" (used by visualization helpers)
- visual_enabled: bool, default true

---

## Recommended Config Combinations
- ECA + RoPE + Local (Phase 1): configs/yaml/eca_rope_local.yaml
- Inception + ALiBi (Phase 2): configs/yaml/inception_alibi.yaml
- RevIN + Decomposition + Local (Phase 2 + Phase 1): configs/yaml/revin_decomp_local.yaml
- Conformer + RevIN (Phase 2): configs/yaml/conformer_revin.yaml
- TCN with Gated activation (Phase 1): configs/yaml/tcn_gated.yaml
- SSM + Local (Phase 3 + Phase 1): configs/yaml/ssm_local.yaml

Tips:
- For long sequences: local attention with dilation, TCN gated, RoPE/ALiBi
- For multi-scale patterns: inception CNN or dilated CNN; multiscale attention
- For non-stationary data: enable RevIN and decomposition (ma/ets)
- For strong variable correlation: enable channel attention and consider local attention

---

## Notes on Performance and Scenarios
- Channel Attention (ECA/SE): small overhead; typical +1–3% accuracy gains
- Local Attention: reduces O(T^2) to approx O(T·W); set window_size 32–128
- RoPE/ALiBi: improves long-range generalization; minimal overhead
- TCN Gated: improves nonlinearity, stable gradients on non-stationary data
- Inception CNN: stronger multi-scale capture; moderate compute increase
- Conformer: better short/long-term balance; moderate increase in params/latency
- RevIN: improves robustness to distribution shift; negligible overhead
- Decomposition: stabilizes trends/seasons; small conv cost (MA) or linear ETS loop
- SSM (placeholder): efficient long-sequence modeling; replaceable with advanced SSM libs later

---

## Examples
Run with config:
```bash
python main.py --config configs/yaml/eca_rope_local.yaml
```

Adjust sub-sections above to compose your own configuration. Keep defaults for backward compatibility if unsure.

