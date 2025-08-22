# CNN+LSTM+Attention 时间序列预测模型配置指南

## 目录
- [概述](#概述)
- [模型配置 (Model)](#模型配置-model)
  - [CNN配置 (CNNConfig)](#cnn配置-cnnconfig)
  - [TCN配置 (TCNConfig)](#tcn配置-tcnconfig)
  - [LSTM配置 (LSTMConfig)](#lstm配置-lstmconfig)
  - [注意力配置 (AttentionConfig)](#注意力配置-attentionconfig)
  - [归一化配置 (Normalization)](#归一化配置-normalization)
  - [分解配置 (Decomposition)](#分解配置-decomposition)
- [数据配置 (Data)](#数据配置-data)
- [训练配置 (Training)](#训练配置-training)
- [优化组合推荐](#优化组合推荐)
- [性能影响与适用场景](#性能影响与适用场景)

## 概述

本项目实现了一个多阶段优化的时间序列预测模型，集成了现代深度学习架构的最新进展：

- **第一阶段优化**：ECA/SE通道注意力、相对位置编码(ALiBi/RoPE)、局部窗口注意力、TCN门控激活
- **第二阶段优化**：InceptionTime多核分支、Conformer混合块、趋势-季节分解、RevIN归一化
- **第三阶段优化**：SSM状态空间模型、Horizon-aware解码头、Feature-as-Token变体

## 模型配置 (Model)

### CNN配置 (CNNConfig)

#### 基础参数

| 参数名 | 类型 | 默认值 | 可选值 | 功能描述 |
|--------|------|--------|--------|----------|
| `variant` | str | "standard" | standard, depthwise, dilated, inception | CNN架构变体选择 |
| `dropout` | float | 0.1 | 0.0-1.0 | Dropout概率 |
| `use_batchnorm` | bool | true | true, false | 是否使用批归一化 |

#### 通道注意力 (第一阶段新增)

| 参数名 | 类型 | 默认值 | 可选值 | 功能描述 |
|--------|------|--------|--------|----------|
| `use_channel_attention` | bool | false | true, false | 是否启用通道注意力 |
| `channel_attention_type` | str | "eca" | eca, se | 通道注意力类型 |

#### 层配置 (CNNLayerConfig)

| 参数名 | 类型 | 默认值 | 可选值 | 功能描述 |
|--------|------|--------|--------|----------|
| `out_channels` | int | 必填 | >0 | 输出通道数 |
| `kernel_size` | int | 3 | >0 | 卷积核大小 |
| `stride` | int | 1 | >0 | 步长 |
| `padding` | int/None | None | >=0 | 填充，None表示same padding |
| `dilation` | int | 1 | >0 | 膨胀率 |
| `activation` | str | "relu" | relu, gelu, silu | 激活函数 |
| `pool` | str/None | "max" | max, avg, None | 池化类型 |
| `pool_kernel_size` | int | 2 | >0 | 池化核大小 |
| `dilation_rates` | List[int]/None | None | [1,2,4] | 膨胀卷积的膨胀率列表 |

#### Inception变体专用参数 (第二阶段新增)

| 参数名 | 类型 | 默认值 | 功能描述 |
|--------|------|--------|----------|
| `inception_kernel_sizes` | List[int] | [3,5,7] | Inception分支的卷积核尺寸 |
| `inception_dilations` | List[int] | [1,2] | Inception分支的膨胀率 |

#### 使用示例

```yaml
model:
  cnn:
    variant: inception
    use_channel_attention: true
    channel_attention_type: eca
    layers:
      - out_channels: 64
        kernel_size: 3
        activation: relu
        inception_kernel_sizes: [3,5,7]
        inception_dilations: [1,2]
```

### TCN配置 (TCNConfig)

#### 基础参数

| 参数名 | 类型 | 默认值 | 可选值 | 功能描述 |
|--------|------|--------|--------|----------|
| `enabled` | bool | false | true, false | 是否启用TCN |
| `dropout` | float | 0.0 | 0.0-1.0 | Dropout概率 |
| `use_batchnorm` | bool | true | true, false | 是否使用批归一化 |

#### 层配置 (TCNLayerConfig)

| 参数名 | 类型 | 默认值 | 可选值 | 功能描述 |
|--------|------|--------|--------|----------|
| `out_channels` | int | 必填 | >0 | 输出通道数 |
| `kernel_size` | int | 3 | >0 | 卷积核大小 |
| `dilation` | int/None | None | >0 | 膨胀率，None时自动递增 |
| `activation` | str | "relu" | relu, gelu, silu, gated | 激活函数 |
| `use_weightnorm` | bool | true | true, false | 是否使用权重归一化 |

#### 门控激活 (第一阶段新增)

当 `activation: gated` 时，使用门控激活：`y = conv(x) * sigmoid(gate(x))`

#### 使用示例

```yaml
model:
  cnn:
    variant: tcn
  tcn:
    enabled: true
    layers:
      - out_channels: 64
        kernel_size: 3
        dilation: 1
        activation: gated
      - out_channels: 128
        kernel_size: 3
        dilation: 2
        activation: gated
```


### LSTM配置 (LSTMConfig)

| 参数名 | 类型 | 默认值 | 可选值 | 功能描述 |
|--------|------|--------|--------|----------|
| `rnn_type` | str | "lstm" | lstm, gru, ssm | RNN类型选择（第三阶段新增：ssm 占位）|
| `hidden_size` | int | 128 | >0 | 隐状态维度 |
| `num_layers` | int | 2 | >0 | 堆叠层数 |
| `bidirectional` | bool | true | true, false | 是否双向 |
| `dropout` | float | 0.1 | 0.0-1.0 | 层间dropout |

示例：
```yaml
model:
  lstm:
    rnn_type: gru
    hidden_size: 256
    num_layers: 2
    bidirectional: true
    dropout: 0.1
```

### 注意力配置 (AttentionConfig)

| 参数名 | 类型 | 默认值 | 可选值 | 功能描述 |
|--------|------|--------|--------|----------|
| `enabled` | bool | true | true, false | 是否启用注意力模块 |
| `variant` | str | "standard" | standard, multiscale, local, conformer | 注意力变体（第一/二阶段新增：local/conformer）|
| `num_heads` | int | 4 | >0 | 注意力头数 |
| `dropout` | float | 0.1 | 0.0-1.0 | 注意力/输出dropout |
| `add_positional_encoding` | bool | false | true, false | 是否添加绝对位置（正弦） |
| `positional_mode` | str | "none" | none, absolute, alibi, rope | 位置编码/偏置模式（第一阶段新增：alibi/rope）|
| `multiscale_scales` | List[int] | [1,2] | 整数列表 | 多尺度注意力的尺度 |
| `multiscale_fuse` | str | "sum" | sum, concat | 多尺度融合策略 |
| `local_window_size` | int | 64 | >0 | 局部窗口大小（第一阶段新增）|
| `local_dilation` | int | 1 | >=1 | 窗口膨胀步长（第一阶段新增）|

示例：
```yaml
model:
  attention:
    enabled: true
    variant: local
    num_heads: 4
    dropout: 0.1
    positional_mode: rope
    local_window_size: 64
    local_dilation: 2
```

### 模型通用参数 (ModelConfig)

| 参数名 | 类型 | 默认值 | 功能描述 |
|--------|------|--------|----------|
| `fc_hidden` | int | 128 | 最终MLP头的隐藏维度 |
| `forecast_horizon` | int | 1 | 预测步长 |
| `normalization` | dict | {revin: {enabled: false}} | 归一化设置（第二阶段新增：RevIN）|
| `decomposition` | dict | {enabled: false, method: ma, kernel: 25, seasonal_period: 24} | 趋势-残差分解（第二阶段新增）|

示例（RevIN + 分解）：
```yaml
model:
  normalization:
    revin:
      enabled: true
  decomposition:
    enabled: true
    method: ma
    kernel: 25
```

## 数据配置 (Data)

| 参数名 | 类型 | 默认值 | 可选值 | 功能描述 |
|--------|------|--------|--------|----------|
| `data_path` | str/None | None | 路径 | 数据文件路径（CSV/NPZ/NPY）|
| `sequence_length` | int | 64 | >0 | 输入序列长度 |
| `horizon` | int | 1 | >0 | 预测步长（与 model.forecast_horizon 对齐）|
| `feature_indices` | List[int]/None | None | 列索引 | 选择使用的特征列索引 |
| `target_indices` | List[int]/None | None | 列索引 | 目标列索引（默认最后一列）|
| `train_split` | float | 0.7 | (0,1) | 训练集占比 |
| `val_split` | float | 0.15 | (0,1) | 验证集占比 |
| `normalize` | str | "standard" | standard, minmax, none | 数据预处理标准化类型 |
| `batch_size` | int | 64 | >0 | 批大小 |
| `num_workers` | int | 0 | >=0 | DataLoader 并行度 |
| `shuffle_train` | bool | true | true, false | 训练集是否打乱 |
| `drop_last` | bool | false | true, false | 不满批是否丢弃 |

示例：
```yaml
data:
  data_path: data/my_dataset.csv
  sequence_length: 128
  horizon: 12
  normalize: minmax
  batch_size: 128
  num_workers: 4
```

## 训练配置 (Training)

### OptimizerConfig
| 参数名 | 类型 | 默认值 | 可选值 | 功能描述 |
|--------|------|--------|--------|----------|
| `name` | str | "adam" | adam, sgd, adamw | 优化器类型 |
| `lr` | float | 1e-3 | >0 | 学习率 |
| `weight_decay` | float | 1e-4 | >=0 | 权重衰减 |
| `momentum` | float | 0.9 | >=0 | 仅SGD使用 |
| `betas` | List[float]/None | None | [beta1, beta2] | Adam/AdamW 超参 |

### SchedulerConfig
| 参数名 | 类型 | 默认值 | 可选值 | 功能描述 |
|--------|------|--------|--------|----------|
| `name` | str/None | "cosine" | cosine, step, plateau, None | 学习率调度策略 |
| `step_size` | int | 10 | >0 | StepLR 步长 |
| `gamma` | float | 0.5 | (0,1) | 衰减因子 |
| `T_max` | int | 50 | >0 | CosineAnnealingLR 的 T_max |
| `reduce_on_plateau_mode` | str | "min" | min, max | Plateau 监控模式 |
| `reduce_on_plateau_patience` | int | 5 | >=0 | Plateau 耐心值 |

### EarlyStoppingConfig
| 参数名 | 类型 | 默认值 | 可选值 | 功能描述 |
|--------|------|--------|--------|----------|
| `enabled` | bool | true | true, false | 是否启用早停 |
| `patience` | int | 10 | >=0 | 训练轮耐心值 |
| `min_delta` | float | 0.0 | >=0 | 指标最小改善幅度阈值 |

### TrainingConfig
| 参数名 | 类型 | 默认值 | 功能描述 |
|--------|------|--------|----------|
| `epochs` | int | 50 | 训练轮数 |
| `loss` | str | "mse" | 损失函数类型（mse, mae, huber）|
| `optimizer` | OptimizerConfig | - | 优化器配置 |
| `scheduler` | SchedulerConfig | - | 学习率调度配置 |
| `early_stopping` | EarlyStoppingConfig | - | 早停配置 |
| `checkpoints` | CheckpointConfig | - | 模型保存配置 |
| `gradient_clip` | float/None | 1.0 | 梯度裁剪阈值（None禁用）|
| `mixed_precision` | bool | false | 是否启用混合精度 |
| `log_dir` | str | "runs" | TensorBoard日志目录 |
| `seed` | int | 42 | 随机种子 |
| `print_every` | int | 50 | 训练过程中打印间隔 |
| `deterministic` | bool/None | None | cudnn 确定性开关（None表示遵循PyTorch默认）|
| `cudnn_benchmark` | bool/None | None | cudnn benchmark 开关 |
| `matmul_precision` | str/None | None | 矩阵乘精度（"high","medium"）|

### CheckpointConfig
| 参数名 | 类型 | 默认值 | 功能描述 |
|--------|------|--------|----------|
| `dir` | str | "checkpoints" | 模型保存目录 |
| `save_best_only` | bool | true | 仅保存最佳模型 |
| `monitor` | str | "val_loss" | 监控指标 |
| `export_best_dir` | str/None | None | 另存最佳模型目录（可选）|

## 优化组合推荐

- 长序列（T≥1024）：
  - attention.variant: local
  - attention.positional_mode: rope 或 alibi
  - cnn.variant: dilated 或 inception
  - tcn.layers.*.activation: gated（若用TCN骨干）
  - 可启用 normalization.revin.enabled: true

- 多变量强相关：
  - cnn.use_channel_attention: true（eca 优先）
  - attention.variant: local 或 conformer
  - positional_mode: rope

- 强趋势/季节性：
  - decomposition.enabled: true（method: ma 或 ets）
  - normalization.revin.enabled: true

- 多步长预测（大 horizon）：
  - 可后续切换到更强的解码头（第三阶段计划），当前建议 fc_hidden 适当增大

## 性能影响与适用场景

- ECA/SE 通道注意力（第一阶段）：
  - 额外开销极小（ECA 更轻），一般带来 1–3% 误差改善
- RoPE/ALiBi（第一阶段）：
  - 几乎不增加参数；长序列与外推稳健性更佳
- 局部窗口注意力（第一阶段）：
  - 将 O(T^2) 降低为 O(T·W)，适合长序列；窗口太小可能丢失极远依赖
- TCN gated（第一阶段）：
  - 轻微增加计算，非平稳数据更稳；对复杂序列有效
- Inception CNN（第二阶段）：
  - 增加分支并发，参数/计算增加；多尺度/多周期数据收益明显
- Conformer（第二阶段）：
  - 引入 MHSA+DWConv 混合，表达力强；计算开销中等
- RevIN（第二阶段）：
  - 几乎无额外参数；跨域/季节性变化时显著改善
- 趋势分解（第二阶段）：
  - 少量额外算子；对趋势明显数据提升泛化
- SSM（第三阶段）：
  - 目标是替代 RNN 提升长序列效率与精度；需要选择成熟实现（当前为占位）

---

备注：第三阶段的“解码头（Horizon-aware）”与“Feature-as-Token 变体”尚未集成到主干配置中，等需求明确后可在本指南中追加对应章节与参数说明。
