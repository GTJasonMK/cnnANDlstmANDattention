## 项目神经网络架构总览

本文档系统阐述当前项目中实现的所有神经网络架构与其配置方式，涵盖主模型、CNN 变体、循环网络处理器、注意力机制，以及与配置文件的对应关系与适用建议。

---

### 1. 主要模型架构：CNNLSTMAttentionModel

文件：model_architecture.py

- 输入输出
  - 输入张量形状：(B, T, F)，分别是 batch、时间步、特征数
  - 输出：
    - 单目标：(B, horizon)
    - 多目标：(B, horizon, n_targets)

- 模块组成与数据流
  1) CNN 特征提取（可选变体）
     - 根据 `cnn_variant` 选择：
       - `standard` → CNNFeatureExtractor（cnn_feature_extractor.py）
       - 其他（`depthwise`/`dilated`）→ AdvancedCNNFeatureExtractor（advanced_cnn.py）
     - 输出形状：(B, T, C_cnn)
  2) 循环编码（LSTM/GRU）
     - 在构造后，通过 `model.rnn_type` 切换：`lstm`（默认）或 `gru`
     - 封装类：LSTMProcessor 或 GRUProcessor
     - 输出形状：(B, T, H_rnn)
  3) 注意力层（可选）
     - 由 `attn_enabled` 控制启用
     - `attn_variant`：
       - `standard` → MultiHeadSelfAttention（attention_mechanism.py）
       - `multiscale` → MultiScaleTemporalAttention（improved_attention.py）
     - 输出形状保持 (B, T, H_attn)
  4) 预测头（全连接）
     - 取最后一个时间步的隐藏状态 `last = y[:, -1, :]`
     - MLP: Linear(H→fc_hidden) + ReLU + Dropout + Linear(fc_hidden→horizon×n_targets)
     - 根据目标数 reshape 为最终输出形状

- 关键参数映射（由 main.build_model_with_data 传入）
  - `cnn_layers`、`use_batchnorm`、`cnn_dropout`、`cnn_variant`
  - `lstm_hidden`、`lstm_layers`、`bidirectional`、`lstm_dropout`
  - `attn_enabled`、`attn_heads`、`attn_dropout`、`attn_variant`、`attn_add_pos_enc`
  - `multiscale_scales`、`multiscale_fuse`
  - `fc_hidden`、`forecast_horizon`、`n_targets`

---

### 2. CNN 特征提取器的变体

#### 2.1 标准 CNN：CNNFeatureExtractor（cnn_feature_extractor.py）
- 目标：在时间维上保持长度（same padding），提取局部时序模式
- 每层结构（顺序）：
  - Conv1d(in=current_c, out=out_c, kernel=k, stride=s, padding=p, dilation=d)
  - 可选 BatchNorm1d(out_c)
  - 激活：ReLU/GELU/ELU
  - 池化（可选）：Max/Average，kernel=kpool，stride=1，带 padding 以保持长度
  - Dropout（按 cnn_dropout）
- 输出：将 (B, C, T) 转回 (B, T, C)
- out_channels 属性：回溯最后一层 Conv1d 的 out_channels

特点与适用：
- 简洁稳健，适合大多数时间序列的局部模式提取
- 计算与参数规模与 out_c、in_c、kernel 成正比

#### 2.2 高级 CNN：AdvancedCNNFeatureExtractor（advanced_cnn.py）
- 变体选择：`architecture_type` ∈ {`standard`, `depthwise`, `dilated`}
- 通用流程：按 layer_configs 逐层堆叠，输出 (B, T, C)

(1) standard 变体（_build_standard_block）
- Conv1d（same padding，支持 dilation）、BatchNorm、激活（ReLU/GELU/SiLU）、可选池化、Dropout
- 适用：与标准 CNN 相同，但支持更灵活的膨胀率与激活选择

(2) depthwise 变体（_build_depthwise_block + DepthwiseSeparableConv1d）
- DepthwiseSeparableConv1d：
  - 深度卷积：groups=in_channels 的 Conv1d（学习每个通道的时间模式）
  - 逐点卷积：1×1 Conv1d（建模通道间交互）
  - 激活：ReLU/GELU/SiLU
- 后接 BatchNorm、可选池化、Dropout
- 参数/计算优势：相比标准卷积，参数量近似从 `out_c*in_c*k` 降到 `in_c*k + in_c*out_c`
  - 粗略比值 ≈ `(k + out_c)/(out_c*k) = 1/k + 1/(k*out_c)`，当 k≥3 且 out_c 较大，降幅显著
- 适用：在显存/算力受限或追求更高吞吐时，通常能以更少参数获得相近表征能力

(3) dilated 变体（_build_dilated_block + DilatedConvBlock）
- DilatedConvBlock：
  - 多分支膨胀卷积：对 dilation_rates 中每个膨胀率建立一支 Conv1d
  - 每支输出通道数：`branch_channels = max(1, out_c // len(rates))`
  - 拼接：沿通道维 concat 所有支路输出 → `concat_channels = branch_channels * len(rates)`
  - 归一化与激活：BatchNorm1d(concat_channels) + ReLU/GELU
  - Dropout：按模块 dropout_p
  - 通道对齐（关键修复点）：通过 1×1 Conv 将 concat_channels 投影到 `out_channels`（若相等则 Identity）
  - 残差：仅当 `in_channels == out_channels` 且启用残差时，执行恒等/1×1 投影再相加
- 适用：针对不同时间尺度（短期/中期/长期）的模式同时建模，常在时序预测中带来更好的收敛与更低损失
- 计算与参数：多分支会带来一定额外开销；当 `branch_channels*out_branches ≈ out_c` 时，额外成本主要来自 BN 与（若需要的）1×1 投影

对比与选择建议：
- standard：基础稳健，便于做消融与基线
- depthwise：速度/参数效率优先；在吞吐优先的实验中推荐
- dilated：多尺度模式捕捉能力强；在追求更低 loss 的场景中推荐

---

### 3. 循环神经网络处理器

文件：lstm_processor.py、gru_processor.py

- LSTMProcessor
  - 封装 nn.LSTM，支持堆叠层与双向（bidirectional）
  - dropout：PyTorch 语义，仅在 num_layers>1 的层间生效
  - 接口：输入 (B, T, F)，输出 (B, T, H) 以及 (hn, cn)
  - 属性 `output_size = hidden_size * (2 if bidirectional else 1)`
  - return_sequence：若 False，则只返回最后一步的 (B, H)

- GRUProcessor
  - 封装 nn.GRU，接口与 LSTMProcessor 对齐，便于替换
  - 返回 (outputs, hn)，没有细胞状态 c（GRU 特性）

- 在主模型中的使用
  - CNN 输出通道作为 RNN 输入维
  - 通过 `model.rnn_type` 决定构造 GRU 或 LSTM（默认 LSTM；在 main 中根据配置写回）
  - 在构造时 `dropout` 选择：若未显式给定 lstm_dropout，则沿用 cnn_dropout，保持正则一致性

---

### 4. 注意力机制

#### 4.1 多头自注意力：MultiHeadSelfAttention（attention_mechanism.py）
- 输入/输出：x∈(B, T, D) → y∈(B, T, D)
- 过程：
  - 线性映射得到 Q/K/V（维度各为 D）
  - 拆分为 `num_heads` 个头，每头维度 `d_k = D / num_heads`
  - 可选正弦位置编码（对 Q/K 相加）
  - 点积注意力 + softmax + dropout
  - 拼回 (B, T, D) 后线性映射输出
  - 可选返回注意力权重 `(B, H, T, T)` 用于可视化
- 特点：在时间维上进行全局依赖建模，能突出关键时间步

#### 4.2 多尺度时序注意力：MultiScaleTemporalAttention（improved_attention.py）
- 思路：在多个下采样尺度上分别做自注意力，再上采样回原长度融合
- 步骤：
  1) 下采样：对 (B,T,D) 沿时间维做 avg_pool1d → (B, T/s, D)
  2) 每个尺度上独立运行 MHA → y_s
  3) 上采样：线性插值回 T 步 → y_up
  4) 融合：
     - `fuse="sum"`：同维求和
     - `fuse="concat"`：特征维拼接后线性投影回 D
  5) 输出：y∈(B, T, D)；可选将不同尺度的注意力权重粗略上采样到 (B,H,T,T) 并做平均
- 特点：同时关注短期与长期模式；常带来更稳健的性能提升

---

### 5. 模型参数与配置

配置数据类：config.py

- CNN 配置（ModelConfig.cnn: CNNConfig）
  - `variant`: "standard" | "depthwise" | "dilated"
  - `layers`: 每层字段（示例）
    - `out_channels`, `kernel_size`, `stride`, `padding|dilation`
    - `activation`（relu|gelu|elu|swish/SiLU）
    - `pool`（max|avg|None）, `pool_kernel_size`
    - `dilation_rates`（仅 dilated 变体的分支膨胀率列表）
  - `use_batchnorm`, `dropout`

- RNN 配置（ModelConfig.lstm: LSTMConfig）
  - `rnn_type`: "lstm" | "gru"
  - `hidden_size`, `num_layers`, `bidirectional`, `dropout`

- 注意力配置（ModelConfig.attention: AttentionConfig）
  - `enabled`: 是否启用
  - `variant`: "standard" | "multiscale"
  - `num_heads`, `dropout`, `add_positional_encoding`
  - `multiscale_scales`, `multiscale_fuse`（仅 multiscale 使用）

- 预测头与任务（ModelConfig）
  - `fc_hidden`: MLP 中间维度
  - `forecast_horizon`: 预测步数

- 示例（节选）
```yaml
model:
  fc_hidden: 128
  forecast_horizon: 3
  cnn:
    variant: dilated
    dropout: 0.1
    use_batchnorm: true
    layers:
      - {out_channels: 32, kernel_size: 5, activation: relu, pool: max, pool_kernel_size: 2, dilation_rates: [1,2,4]}
      - {out_channels: 64, kernel_size: 3, activation: gelu, pool: max, pool_kernel_size: 2, dilation_rates: [1,2,4]}
  lstm:
    rnn_type: lstm
    hidden_size: 128
    num_layers: 2
    bidirectional: true
    dropout: 0.1
  attention:
    enabled: true
    variant: multiscale
    num_heads: 4
    dropout: 0.1
    add_positional_encoding: false
    multiscale_scales: [1, 2]
    multiscale_fuse: sum
```

- main.py 中的 CLI 覆盖
  - `--output_dir` 会统一为：图片目录（visual_save_dir）、checkpoint 目录、日志目录增加前缀
  - `--image_dir` 可单独指定图片目录名；visualizer 动态读取 `VIS_SAVE_DIR` 确保图片保存在目标目录

---

### 6. 参数数量与计算复杂度（定性对比）

- 标准卷积（1D Conv）
  - 参数量 ~ `out_c * in_c * k`；计算量与该值及时序长度成正比
- 深度可分离卷积（Depthwise + Pointwise）
  - 参数量 ~ `in_c * k (depthwise) + in_c * out_c (pointwise)`
  - 相对标准卷积的比例 ≈ `1/k + 1/(k*out_c)`，总体更省参、速度更快
- 膨胀卷积块（多分支）
  - 多支路带来额外 BN 与（可选）1×1 投影开销；但支路通道平均分配时，concat 通道近似等于 out_c，整体在同级别上略有增加
- 注意力
  - 标准 MHA：复杂度 ~ O(T^2 * D)，对长序列开销显著
  - 多尺度注意力：在较短序列（T/s）上计算注意力，再上采样融合；在保留性能的同时降低计算与显存占用

性能经验（时间序列预测）：
- depthwise：在保持精度的同时显著提升吞吐，适合大批量/快速试验
- dilated + multiscale：多尺度信息捕捉能力强，常带来更低损失但训练更慢
- 提升 `fc_hidden`、`lstm.hidden_size`、注意力 `num_heads` 能增强表达，但会增加计算/显存

---

### 7. 设计思路与实践建议

- CNN 的职责：对原始多变量序列做局部模式抽取、对齐对每步特征
- RNN 的职责：对时间维做顺序建模，捕捉时序依赖与状态演化
- 注意力的职责：在序列的全局范围内做动态权重分配，聚焦关键时间片；多尺度注意力进一步兼顾不同时间跨度
- 预测头：以“最后一步”代表整段序列的摘要用于多步预测，简洁高效
- 实践建议：
  - 吞吐优先：`cnn.variant=depthwise`，大 batch + AMP + cudnn_benchmark
  - 低损失优先：`cnn.variant=dilated` + `attention.variant=multiscale`，适度增大 RNN/FC 宽度
  - 长序列：优先多尺度注意力或缩短 T（滑窗更短、步幅更大）以控制注意力的 O(T^2)

---

如需进一步的图示（Mermaid/结构图）或在你的数据集上给出针对性的配置推荐，请告知数据的序列长度 T、特征数 F、预测步长与对速度/精度的侧重，我可以补充更细致的对比与可视化。
