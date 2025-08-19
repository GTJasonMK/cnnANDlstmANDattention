# 基于 PyTorch 的 CNN + LSTM + Attention 多特征时间序列预测框架

本项目提供一个模块化、可配置的 CNN + LSTM +（可选）多头自注意力 预测架构，适用于多特征时间序列的单步或多步预测任务。项目结构清晰，强调可读性、低耦合和可维护性，便于在工业/科研场景中快速落地与扩展。

---

## 1. 项目简介

- 功能：对形状为 (batch_size, sequence_length, num_features) 的多特征时序数据进行特征提取、时序建模和多步预测。
- 技术栈：PyTorch 2.0+、NumPy、Pandas、Matplotlib（可选 TensorBoard 日志）。
- 适用场景：
  - 预测未来数步（多步）连续值，如需求量、传感器读数、金融指标等；
  - 多特征输入（协变量），预测单/多目标序列；
  - 需要局部模式捕获（CNN）、长期依赖（LSTM）与动态权重分配（Self-Attention）的任务。

---

## 2. 功能特性

- CNN 特征提取：多层 1D 卷积（可选 BN、激活、池化、dropout），保持时间长度；
- LSTM 时序建模：多层、可选双向 BiLSTM，支持层间 dropout；
- 自注意力：轻量级多头自注意力，支持可选正弦位置编码；
- 输出头：全连接 MLP 支持单/多步、多目标预测；
- 数据管道：滑动窗口切片，按时间顺序划分训练/验证/测试；
- 归一化：仅基于训练片段估计统计量，防止数据泄漏；
- 可配置：卷积核、隐藏维度、注意力头数、损失函数（MSE/MAE/Huber）、优化器（Adam/SGD/AdamW）、LR 调度（Cosine/Step/Plateau）；
- 训练工具：早停、梯度裁剪、混合精度（AMP）、断点续训、模型检查点；
- 可视化：训练/验证损失、学习率、预测对比、注意力热力图、简单特征重要性。

---

## 3. 项目结构

- config.py：配置数据类与 JSON/YAML 加载器；
- cnn_feature_extractor.py：1D CNN 特征提取模块（保持时序长度）；
- lstm_processor.py：LSTM/BiLSTM 时序编码模块；
- attention_mechanism.py：多头自注意力层（可选位置编码）；
- model_architecture.py：整合 CNN → LSTM → Attention → MLP 的主模型；
- data_preprocessor.py：数据读取、滑窗切片、归一化与 DataLoader 构建；
- trainer.py：训练循环、验证、调度、早停、AMP、TensorBoard、checkpoint；
- evaluator.py：回归指标（MSE/MAE/RMSE/MAPE）；
- visualizer.py：损失与学习率曲线、预测对比、注意力热力图、特征重要性；
- main.py：主入口，读取配置、训练、可视化，支持 --resume 断点续训；
- example_config.yaml：示例配置（建议在其基础上调整）；
- e2e_example.py：纯合成数据端到端演示脚本。

---

## 4. 环境要求

- Python：3.9+（推荐 3.10/3.11）
- 必需依赖：
  - torch>=2.0（请根据 CUDA 情况从官方指引安装）
  - numpy, pandas, matplotlib, pyyaml
- 可选依赖：
  - tensorboard（如需使用训练过程的实时日志）

---

## 5. 安装步骤

1) 创建与激活虚拟环境（可选但推荐）

```bash
# conda
conda create -n ts-forecast python=3.10 -y
conda activate ts-forecast
# 或 venv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

2) 安装 PyTorch（根据你的平台与 CUDA 版本）

```bash
# 请参考 https://pytorch.org/ 获取最合适的安装命令
# 例如 CPU 版：
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

3) 安装其他依赖

```bash
pip install numpy pandas matplotlib pyyaml tensorboard
```

---

## 6. 使用方法

### 6.1 数据准备

- 文件格式：CSV/NPY/NPZ；形状为 (N, F)，按时间从上到下排序；
- F 表示特征总数；默认预测目标为最后一列（可在配置中指定 target_indices）。

示例 CSV（含表头亦可，默认读取全部列作为特征）：

```csv
ts_feature1,ts_feature2,target
1.23,4.56,7.89
...
```

### 6.2 配置文件（基于 example_config.yaml）

关键字段说明：

- model.cnn.layers：CNN 层列表，每层含 out_channels, kernel_size, activation, pool 等；
- model.lstm：hidden_size, num_layers, bidirectional, dropout；
- model.attention：enabled（是否启用）、num_heads、dropout、add_positional_encoding；
- model.fc_hidden：预测头的隐藏维度；
- model.forecast_horizon：预测步数（>1 为多步预测）；
- data.data_path：数据文件路径（CSV/NPY/NPZ）；
- data.sequence_length：输入序列长度 T；
- data.horizon：预测步长 H（通常与 model.forecast_horizon 保持一致）；
- data.feature_indices：输入特征列（默认全部）；
- data.target_indices：目标列（默认最后一列）；
- data.normalize：standard|minmax|none；仅基于训练切片估计统计量；
- train.loss：mse|mae|huber；
- train.optimizer：adam|sgd|adamw 及其参数；
- train.scheduler：cosine|step|plateau 及其参数；
- train.early_stopping：早停开关与耐心值等；
- train.checkpoints.dir：checkpoint 保存目录；save_best_only：是否仅保存更优；
- train.mixed_precision：是否启用 AMP；
- train.print_every：训练过程中打印间隔。

### 6.3 训练命令

```bash
python main.py --config example_config.yaml
```

### 6.4 断点续训

```bash
python main.py --config example_config.yaml --resume checkpoints/model_epochX.pt
```

### 6.5 预测与评估

- main.py 在训练完成后会：
  - 使用测试集进行预测并绘制预测 vs 真实值曲线；
  - 进行一次带注意力权重的前向以绘制注意力热力图；
- 如需单独评估：

```python
from evaluator import evaluate_model
metrics = evaluate_model(model, test_loader, device)
print(metrics)  # {'mse':..., 'mae':..., 'rmse':..., 'mape':...}
```

### 6.6 实时日志（可选）

- 如果安装了 tensorboard，Trainer 会写入日志到 train.log_dir（默认 runs）。

```bash
tensorboard --logdir runs
```

---

## 7. 配置参数详解（推荐值）

- CNN 层：
  - out_channels：32~256（任务复杂度越高可适当增大）；
  - kernel_size：3/5/7；
  - activation：relu|gelu|elu；
  - pool：max|avg|None，一般 kernel_size=2；
  - dropout：0.0~0.3；
- LSTM：
  - hidden_size：64~256（与数据维度、复杂度相关）；
  - num_layers：1~3；
  - bidirectional：True 通常表现更稳；
  - dropout：0.0~0.3；
- Attention：
  - enabled：True/False；
  - num_heads：2/4/8（需整除隐藏维度）；
  - dropout：0.0~0.3；
  - add_positional_encoding：True/False；
- 优化与训练：
  - loss：mse 常用；也可尝试 mae/huber；
  - optimizer：adam/adamw 常用，lr 1e-3 起；
  - scheduler：cosine（T_max=epochs）、step（每 step_size 轮 gamma 衰减）、plateau（监控 val_loss）；
  - early_stopping：patience 5~15；
  - gradient_clip：如 1.0；
  - mixed_precision：GPU 上可设 True 提升速度与显存利用；
- 数据：
  - sequence_length：32~256；
  - horizon：1（单步）或 >1（多步）；
  - normalize：standard 对大多数任务稳定；minmax 用于范围已知的数据。

---

## 8. 示例演示

- 使用示例配置：

```bash
python main.py --config example_config.yaml
```

- 合成数据端到端演示（无需外部文件）：

```bash
python e2e_example.py
```

该脚本会生成多通道正弦序列，自动构建数据加载器、训练模型并输出训练损失。

---

## 9. 可视化功能

- 训练/验证损失曲线：自动绘制；
- 学习率曲线：若启用调度器则自动绘制；
- 预测 vs 真实：在测试集上绘制多条样本对比；
- 注意力热力图：对测试集取一个 batch 的平均注意力并绘制；
- 特征重要性（简易）：基于预测头第一层权重的绝对值排序：

```python
from visualizer import plot_feature_importance
import numpy as np
w = model.head[0].weight.detach().cpu().numpy()  # (fc_hidden, H)
importance = np.mean(np.abs(w), axis=0)
plot_feature_importance(importance, feature_names=None)
```

> 注：这里是一种简单启发式方法，非严格的可解释性分析。

---

## 10. 常见问题（FAQ）

1) 无法导入 PyYAML：安装 `pip install pyyaml` 或改用 JSON 配置。
2) GPU 未使用：确保 CUDA 环境与 torch 版本匹配；否则将自动用 CPU。
3) 数据维度错误：输入需为 (N, F)；序列长度与 horizon 需满足 `N - T - H + 1 > 0`。
4) `forecast_horizon` 与 `data.horizon` 不一致：建议保持一致，避免输出维度不匹配。
5) CSV 读取失败：请确认路径与编码；需要 pandas：`pip install pandas`。
6) 数据泄漏担忧：本项目仅用训练切片拟合归一化参数，验证/测试复用该统计量。
7) 断点续训无法加载：确保传入正确的 `--resume` 路径、模型结构/配置未破坏兼容性。
8) 注意力头数报错：`d_model`（LSTM 输出维度）需能被 `num_heads` 整除。

---

## 11. 扩展开发

- 新增损失函数：在 `trainer._build_loss` 添加分支；
- 新增优化器/调度器：在 `_build_optimizer`/`_build_scheduler` 中添加；
- 更复杂注意力/Transformer：替换 `attention_mechanism.py` 或在 `model_architecture.py` 引入 Encoder 层；
- 自回归/Seq2Seq 多步解码：在 `model_architecture.py` 新增 Decoder 或循环预测逻辑；
- 新指标：在 `evaluator.py` 添加所需指标并在流程中调用；
- 自定义数据管道：继承或修改 `TimeSeriesDataset` 与 `create_dataloaders`；
- 分布式训练/多卡：在 `trainer.py` 引入 DDP/ZeRO 等（需同步适配日志与保存逻辑）。

---

如需我根据你的数据特点进一步调整网络结构、损失或训练策略，请告知需求与约束（数据规模、实时性要求、指标偏好等），我会基于此优化配置与实现。
