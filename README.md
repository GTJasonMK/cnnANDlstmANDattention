## 项目名称

CNN + LSTM + 多变体注意力的时间序列预测框架（含TCN/多尺度/局部/时空注意力、RevIN与趋势分解、独立评估与批量评估工具）

## 简介

本项目是一个面向多变量时间序列预测的深度学习框架，核心由 CNN/TCN 提取时序局部特征、LSTM/GRU/简化SSM 建模长期依赖，并可选接入多种注意力机制（标准多头、多尺度、局部窗口、Conformer风格、时空注意力）。
支持灵活的配置（YAML/JSON），包含训练器、可视化、独立评估脚本与批量评估脚本，便于快速实验与对比。

- 输入形状： (batch, seq_len, num_features)
- 输出形状： (batch, horizon) 或 (batch, horizon, n_targets)

## 主要特性与功能

- 模型骨干
  - CNN 特征提取（standard/depthwise/dilated/inception）
  - TCN（Temporal Convolutional Network）因果卷积、多层膨胀、残差
  - RNN 支持：LSTM（默认）/GRU/简化 SSM 占位实现
- 注意力机制（可选）
  - 标准多头自注意力（支持位置编码模式 none/absolute/alibi/rope）
  - 多尺度时间注意力（Multi-Scale Temporal Attention）：在多时间尺度下自注意并融合
  - 局部窗口自注意力（支持膨胀窗口）
  - 轻量 Conformer Block 堆叠
  - 时空注意力（时间与特征两个维度的注意）
- 数据管线
  - CSV/NPZ/NPY 加载；按时间滑窗生成 (seq_len, features) → (horizon, targets)
  - 归一化：standard/minmax/none（训练集统计）
  - 可选小波分解（PyWavelets），将多子带特征拼接
- 训练器与工具
  - 训练/验证/测试循环、AMP混合精度（CUDA）、梯度裁剪
  - 优化器（Adam/AdamW/SGD）、学习率调度（Cosine/Step/Plateau）、早停
  - Checkpoint 管理（按epoch保存与best保存），并可导出“最佳模型”到统一目录（含结构哈希与实验标签）
  - TensorBoard 日志（可选）
- 可视化
  - 损失与学习率曲线、参数量柱状图
  - 预测对比、残差直方图、多步长误差对比
  - 注意力热力图（含多头/时空）、LSTM隐状态热力图、CNN通道激活
  - 数据分析图（分布、相关性、数据划分分布、时间性能曲线）
- 评估
  - 在线评估：训练后在测试集计算 MSE/MAE/RMSE/MAPE
  - 独立评估脚本 eval/standalone_eval.py：从 checkpoint + 数据独立复现实验，输出 MSE/MAE/RMSE/MAPE/R2 与图表
  - 批量评估 batch_eval.py：对目录内多个 checkpoint 统一评估并汇总 CSV/对比图

## 技术栈与依赖

- 语言/框架
  - Python 3.9+（建议 3.10）
  - PyTorch ≥ 2.0（使用 torch.amp 与 set_float32_matmul_precision 等）
- 核心依赖（按代码实际使用）
  - torch（必需）
  - numpy（必需）
  - matplotlib（可视化，强烈建议安装）
  - pandas（读取CSV、相关性热力图，使用CSV数据时必需）
  - pyyaml（使用 YAML 配置时必需）
  - pywavelets（启用小波分解时必需，包名：PyWavelets）
  - tensorboard（启用 TensorBoard 日志时建议安装）

说明：项目未提供 requirements.txt；请按需安装上面依赖。

## 安装说明

- 环境要求
  - Python 3.9+/3.10，Windows/Linux/macOS
  - 建议 GPU + CUDA 的 PyTorch 环境（CPU 也可运行）

- 创建虚拟环境（任选其一）
  - venv
    - python -m venv .venv
    - 在 Windows: .venv\\Scripts\\activate
    - 在 Linux/macOS: source .venv/bin/activate
  - conda
    - conda create -n ts-cnnlstm-attn python=3.10 -y
    - conda activate ts-cnnlstm-attn

- 安装依赖（示例）
  - 安装 PyTorch（请参考官网选择对应 CUDA/CPU 版本）
    - pip install torch --index-url https://download.pytorch.org/whl/cu121  # 示例
  - 其余依赖
    - pip install numpy matplotlib pandas pyyaml PyWavelets tensorboard

## 使用示例

1) 准备数据
- 支持 CSV/NPZ/NPY 格式，数值矩阵形状 (N, F)。本仓库在 result/ 提供了示例 CSV（如 weather.csv/traffic.csv/elec.csv）。
- 特征列与目标列可通过配置指定（缺省：features=全部列；targets=全部列或最后一列）。

2) 编写配置（YAML 示例）
新建 configs/example.yaml：

```yaml
device: null
output_dir: outputs/exp1
visual_save_dir: image
visual_enabled: true
model:
  fc_hidden: 128
  forecast_horizon: 3
  cnn:
    variant: standard   # standard|depthwise|dilated|inception|tcn
    dropout: 0.1
    use_batchnorm: true
    layers:
      - {out_channels: 32, kernel_size: 5, activation: relu, pool: max, pool_kernel_size: 2}
      - {out_channels: 64, kernel_size: 3, activation: gelu, pool: max, pool_kernel_size: 2}
  tcn:
    enabled: false      # 若使用 TCN，请设为 true 并在 layers 中给出 dilation 等
    layers:
      - {out_channels: 64, kernel_size: 3, dilation: 1, activation: relu}
      - {out_channels: 64, kernel_size: 3, dilation: 2, activation: relu}
  lstm:
    rnn_type: lstm      # lstm|gru|ssm
    hidden_size: 128
    num_layers: 2
    bidirectional: true
    dropout: 0.1
  attention:
    enabled: true
    variant: standard   # standard|multiscale|local|conformer|spatiotemporal
    num_heads: 4
    dropout: 0.1
    add_positional_encoding: false
    positional_mode: none  # none|absolute|alibi|rope
    multiscale_scales: [1, 2]
    multiscale_fuse: sum
    local_window_size: 64
    local_dilation: 1
    st_mode: serial
    st_fuse: sum
  normalization:
    revin: {enabled: false}
  decomposition:
    enabled: false
    method: ma
    kernel: 25
    seasonal_period: 24

data:
  data_path: result/weather.csv  # 修改为你的数据路径
  sequence_length: 64
  horizon: 3
  feature_indices: null          # 例如: [0,1,2]
  target_indices: null           # 例如: [3]
  train_split: 0.7
  val_split: 0.15
  normalize: standard            # standard|minmax|none
  batch_size: 64
  num_workers: 0
  shuffle_train: true
  drop_last: false
  wavelet: {enabled: false, wavelet: db4, level: 3, mode: symmetric, take: all}

train:
  epochs: 20
  loss: mse                     # mse|mae|huber
  optimizer: {name: adam, lr: 0.001, weight_decay: 0.0001}
  scheduler: {name: cosine, T_max: 20}
  early_stopping: {enabled: true, patience: 5, min_delta: 0.0}
  checkpoints: {dir: checkpoints, save_best_only: true, export_best_dir: exports}
  gradient_clip: 1.0
  mixed_precision: true
  log_dir: runs
  seed: 42
  print_every: 50
```

3) 训练
- 命令行运行：
  - python main.py --config configs/example.yaml
- 可选参数：
  - --resume path/to/checkpoint.pt 恢复训练
  - --output_dir/--image_dir/--ckpt_dir 覆盖配置中的输出目录

4) 评估
- 在线评估：训练结束后，自动在测试集计算指标与生成图像到 image/（或覆盖后的目录）。
- 独立评估（无需 main.py 与 trainer）：
  - python eval/standalone_eval.py --checkpoint checkpoints/model_best.pt --data result/weather.csv --output_dir results/eval --batch_size 128 --sequence_length 64 --horizon 3 --normalize standard
  - 说明：该脚本会从 checkpoint 中读取训练期 cfg，严格复现模型，并输出 MSE/MAE/RMSE/MAPE/R2 与多种图表。
- 批量评估：
  - python batch_eval.py --checkpoints_dir checkpoints/ --data result/weather.csv --output_dir results/batch_eval --device cuda --batch_size 256 --sequence_length 64 --horizon 3 --normalize standard --save_summary_plot

5) 输入数据格式说明
- CSV：表头可有可无，每列应为数值；内部使用 pandas 读取（请安装 pandas）。
- NPZ：从第一个数组键读取；NPY：直接加载数组。
- 归一化统计仅基于训练切分，以避免信息泄漏。

6) 模型调用（Python 代码示例）

```python
import torch
from configs.config import load_config
from main import prepare_run, build_model_with_data, run

cfg = load_config('configs/example.yaml')
# 快速跑一轮（含训练/可视化/评估）
history, metrics = run(cfg)
print(metrics)
```

## 项目结构概览

- main.py：入口脚本；环境与目录设置、数据加载、模型构建、训练与可视化/评估流程
- configs/config.py：配置数据类与加载/严格校验/保存到checkpoint的规范化
- dataProcess/
  - data_preprocessor.py：数据加载、归一化统计（基于训练切分）、滑窗数据集与 DataLoader 构建、小波分解
- cnn/
  - cnn_feature_extractor.py：标准 1D CNN 特征提取
  - advanced_cnn.py：Depthwise/ Dilated/ Inception/通道注意力(ECA/SE)
  - tcn_feature_extractor.py：TCN 因果卷积/膨胀/残差/权重归一化
- rnn/
  - lstm_processor.py：LSTM/双向/层间dropout
  - gru_processor.py：GRU
  - ssm_processor.py：简化 SSM 占位实现（接口兼容）
- attention/
  - attention_mechanism.py：标准多头自注意力（可选位置编码 none/absolute/alibi/rope）
  - improved_attention.py：多尺度时间注意力（MSTA）
  - local_attention.py：局部窗口自注意力
  - conformer_block.py：轻量 Conformer Block
  - spatiotemporal_attention.py：时空注意力
- normalization/revin.py：可逆实例归一化 RevIN（可选）
- preprocess/decomposition.py：简单趋势-残差分解（移动平均/指数平滑近似）
- model_architecture.py：将 CNN/TCN + RNN + Attention 组装为 CNNLSTMAttentionModel，并在 forward 中支持返回注意力权重
- trainer.py：训练循环、AMP、调度/早停、TensorBoard 日志、checkpoint 保存与“最佳模型”导出
- eval/
  - evaluator.py：在线评估（训练流程内部复用）
  - standalone_eval.py：独立评估脚本（严格解析 checkpoint 内 cfg），支持可视化与 CSV/JSON 输出
- visualizer.py：可视化函数集合（损失/LR/预测/残差/注意力/LSTM隐状态/CNN特征/数据分布等）
- scripts/
  - generate_control_yamls.py：按“控制变量”思想批量生成对比 YAML（CNN/RNN/Attention/位置编码/小波等）
  - multi_console_train.py、rank_architectures.py：多实验与结构排序辅助脚本
- batch_eval.py：批量评估多个 checkpoint 并汇总
- tools/：实验结果合并与可视化辅助
- result/：示例CSV与结果汇总示例

## 配置详情（要点）

- data.*
  - data_path：CSV/NPZ/NPY 路径
  - sequence_length/horizon：滑窗长度与预测步数
  - feature_indices/target_indices：索引列表（从 0 开始）；缺省使用全部特征/最后一列为目标
  - normalize：standard|minmax|none；统计基于训练切分
  - wavelet.*：{enabled, wavelet, level, mode, take}，启用后特征将扩维为多子带拼接
- model.cnn.*
  - variant：standard|depthwise|dilated|inception|tcn（若选择 tcn，实际使用 model.tcn 配置）
  - layers：每层 {out_channels,kernel_size,stride,padding,dilation,activation,pool,pool_kernel_size}
  - use_channel_attention/channel_attention_type：ECA 或 SE
- model.tcn.*
  - enabled：true 使用 TCN
  - layers：{out_channels,kernel_size,dilation,activation,use_weightnorm}
- model.lstm.*
  - rnn_type：lstm|gru|ssm；hidden_size/num_layers/bidirectional/dropout
- model.attention.*
  - enabled/variant（standard|multiscale|local|conformer|spatiotemporal）
  - num_heads/dropout/positional_mode（none|absolute|alibi|rope）
  - multiscale_scales/multiscale_fuse；local_window_size/local_dilation；st_mode/st_fuse
- model.normalization.revin.enabled：是否启用 RevIN
- model.decomposition.*：是否使用趋势-残差两路骨干并融合
- train.*
  - epochs/loss（mse|mae|huber）/optimizer/scheduler/early_stopping/gradient_clip
  - mixed_precision（AMP）/log_dir/print_every/checkpoints（dir/export_best_dir/save_best_only）
- 顶层
  - device：'cuda' 或 'cpu'（缺省自动）
  - output_dir/visual_save_dir：统一控制输出目录（main 会确保目录存在并将图像导出到 VIS_SAVE_DIR）

## 贡献指南（可选）

- 提交前请确保：符合现有代码风格、尽量低耦合与易维护；新增功能与现有实现不重复（若同文件存在同类功能，请对比并保留更优的一种）。
- 新增模块请添加简明文档与必要的参数校验；大型改动建议先在 issue 中讨论设计。

## 许可证

仓库中未检测到 LICENSE 文件。如需开源发布，请在根目录添加相应 LICENSE，并在此处注明。

