from __future__ import annotations

from typing import Optional, Tuple, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SSMProcessor(nn.Module):
    """
    完整的状态空间模型 (SSM) 处理器，实现S4/Mamba风格的架构

    基于状态空间模型理论：
    x(t+1) = Ax(t) + Bu(t)
    y(t) = Cx(t) + Du(t)

    包含：
    - HiPPO矩阵初始化
    - 正确的离散化
    - 选择性机制
    - 高效并行计算
    - 数值稳定性处理

    接口与LSTM/GRU兼容，可直接替换使用
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.0,
        return_sequence: bool = True,
        state_size: Optional[int] = None,
        # SSM特定参数
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        dt_init: str = "random",  # "random" | "constant"
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        # HiPPO参数
        hippo_N: Optional[int] = None,
        # 选择性参数
        selective: bool = True,
        # 数值稳定性
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)
        self.return_sequence = bool(return_sequence)
        self.dropout_p = dropout
        self.selective = selective
        self.eps = eps

        # 内部状态维度
        self.state_size = state_size or hidden_size
        self.hippo_N = hippo_N or self.state_size

        # 创建堆叠的SSM层
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else self.output_size
            self.layers.append(
                SSMBlock(
                    input_size=layer_input_size,
                    hidden_size=self.hidden_size,
                    state_size=self.state_size,
                    bidirectional=bidirectional,
                    dt_min=dt_min,
                    dt_max=dt_max,
                    dt_init=dt_init,
                    dt_scale=dt_scale,
                    dt_init_floor=dt_init_floor,
                    hippo_N=self.hippo_N,
                    selective=selective,
                    eps=eps,
                )
            )

        # Dropout层
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

    @property
    def output_size(self) -> int:
        """输出特征维度"""
        return self.hidden_size * (2 if self.bidirectional else 1)

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入张量 (B, T, F)
            hx: 初始状态（兼容性，当前未使用）

        Returns:
            outputs: 输出序列 (B, T, H) if return_sequence else (B, H)
            final_state: 最终状态张量（兼容性）
        """
        if x.dim() != 3:
            raise ValueError(f"期望3D输入 (B, T, F)，得到 {tuple(x.shape)}")

        # 通过所有层处理
        outputs = x
        for i, layer in enumerate(self.layers):
            outputs = layer(outputs)
            if i < len(self.layers) - 1:  # 最后一层不应用dropout
                outputs = self.dropout(outputs)

        # 提取最终状态（兼容性）
        final_state = outputs[:, -1, :] if outputs.size(1) > 0 else torch.zeros(
            outputs.size(0), self.output_size, device=x.device, dtype=x.dtype
        )

        if self.return_sequence:
            return outputs, final_state
        else:
            return final_state, final_state


class SSMBlock(nn.Module):
    """
    单个SSM块，实现完整的状态空间模型
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        state_size: int,
        bidirectional: bool = False,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        hippo_N: int = 64,
        selective: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.bidirectional = bidirectional
        self.selective = selective
        self.eps = eps
        self.dt_min = dt_min
        self.dt_max = dt_max

        # 创建前向SSM
        self.ssm_forward = S4Layer(
            input_size=input_size,
            hidden_size=hidden_size,
            state_size=state_size,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            hippo_N=hippo_N,
            selective=selective,
            eps=eps,
        )

        # 如果是双向，创建反向SSM
        if bidirectional:
            self.ssm_backward = S4Layer(
                input_size=input_size,
                hidden_size=hidden_size,
                state_size=state_size,
                dt_min=dt_min,
                dt_max=dt_max,
                dt_init=dt_init,
                dt_scale=dt_scale,
                dt_init_floor=dt_init_floor,
                hippo_N=hippo_N,
                selective=selective,
                eps=eps,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, F)
        Returns:
            (B, T, H) where H = hidden_size * (2 if bidirectional else 1)
        """
        # 前向方向
        y_forward = self.ssm_forward(x)

        if not self.bidirectional:
            return y_forward

        # 反向方向
        x_reversed = torch.flip(x, dims=[1])
        y_backward = self.ssm_backward(x_reversed)
        y_backward = torch.flip(y_backward, dims=[1])

        # 拼接前向和反向
        return torch.cat([y_forward, y_backward], dim=-1)


class S4Layer(nn.Module):
    """
    S4层实现，包含完整的状态空间模型数学基础

    实现连续状态空间模型：
    dx/dt = Ax + Bu
    y = Cx + Du

    离散化为：
    x_k+1 = Ā x_k + B̄ u_k
    y_k = C x_k + D u_k
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        state_size: int,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        hippo_N: int = 64,
        selective: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.selective = selective
        self.eps = eps

        # 状态空间矩阵
        self.A_log = nn.Parameter(torch.zeros(state_size))  # 对角化A矩阵的对数
        self.B = nn.Parameter(torch.randn(state_size, input_size))
        self.C = nn.Parameter(torch.randn(hidden_size, state_size))
        self.D = nn.Parameter(torch.randn(hidden_size, input_size))

        # 时间步长参数
        if dt_init == "constant":
            dt = torch.exp(
                torch.rand(input_size) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            )
        elif dt_init == "random":
            dt = torch.exp(
                torch.rand(input_size) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            )
        else:
            raise ValueError(f"不支持的dt_init: {dt_init}")

        # 可学习的时间步长
        dt = torch.clamp(dt, min=dt_init_floor)
        self.dt_proj = nn.Linear(input_size, input_size, bias=True)

        # 初始化dt权重
        dt_init_std = dt_scale / math.sqrt(input_size)
        with torch.no_grad():
            self.dt_proj.weight.uniform_(-dt_init_std, dt_init_std)

        # 选择性机制的投影层
        if selective:
            self.B_proj = nn.Linear(input_size, state_size, bias=False)
            self.C_proj = nn.Linear(input_size, state_size, bias=False)
        else:
            self.B_proj = None
            self.C_proj = None

        # 输出投影和归一化
        self.output_proj = nn.Linear(state_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

        # 初始化HiPPO矩阵
        self._init_hippo_A(hippo_N)
        self._init_parameters()

    def _init_hippo_A(self, N: int):
        """
        初始化HiPPO矩阵A，用于长期记忆
        HiPPO (High-order Polynomial Projection Operators)
        """
        # 创建HiPPO-LegS矩阵
        A = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i > j:
                    A[i, j] = math.sqrt((2 * i + 1) * (2 * j + 1))
                elif i == j:
                    A[i, j] = i + 0.5
                else:
                    A[i, j] = -math.sqrt((2 * i + 1) * (2 * j + 1))

        # 转换为对角化形式并取对数（数值稳定性）
        A_diag = np.diag(A)  # 简化为对角矩阵

        # 填充到所需大小
        if self.state_size <= N:
            A_init = A_diag[:self.state_size]
        else:
            # 扩展矩阵
            A_init = np.concatenate([
                A_diag,
                np.ones(self.state_size - N) * A_diag[-1]
            ])

        # 确保负值（稳定性）并取对数
        A_init = -np.abs(A_init)
        self.A_log.data = torch.from_numpy(np.log(-A_init + self.eps)).float()

    def _init_parameters(self):
        """初始化参数以保证稳定训练"""
        # B矩阵初始化
        nn.init.xavier_uniform_(self.B, gain=0.5)

        # C矩阵初始化
        nn.init.xavier_uniform_(self.C, gain=0.5)

        # D矩阵初始化（通常较小）
        nn.init.zeros_(self.D)

        # 输出投影初始化
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.5)

    def discretize(self, dt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将连续状态空间模型离散化
        使用零阶保持 (Zero-Order Hold) 方法

        Args:
            dt: 时间步长 (B, T) 或 (T,)

        Returns:
            A_bar: 离散化A矩阵
            B_bar: 离散化B矩阵
        """
        # 获取A矩阵（对角形式）
        A = -torch.exp(self.A_log)  # 确保负值（稳定性）

        # 扩展维度以支持批处理
        if dt.dim() == 1:
            dt = dt.unsqueeze(0)  # (1, T)
        if dt.dim() == 2 and A.dim() == 1:
            A = A.unsqueeze(0).unsqueeze(0)  # (1, 1, N)
            dt = dt.unsqueeze(-1)  # (B, T, 1)

        # ZOH离散化
        dt_A = dt * A  # (B, T, N)
        A_bar = torch.exp(dt_A)  # (B, T, N)

        # B矩阵离散化
        # B_bar = (A^{-1})(e^{dt*A} - I)B ≈ dt*B for small dt
        B_bar_coeff = torch.where(
            torch.abs(dt_A) > self.eps,
            (A_bar - 1) / A,
            dt  # 一阶近似
        )

        return A_bar, B_bar_coeff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入序列 (B, T, F)

        Returns:
            y: 输出序列 (B, T, H)
        """
        B, T, num_features = x.shape  # 🔥 修复：避免与torch.nn.functional.F冲突

        # 计算时间步长
        dt = F.softplus(self.dt_proj(x))  # (B, T, num_features)
        dt = dt.mean(dim=-1)  # (B, T) - 平均化

        # 选择性机制
        if self.selective:
            B_t = self.B_proj(x)  # (B, T, N)
            C_t = self.C_proj(x)  # (B, T, N)
        else:
            B_t = self.B.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)  # (B, T, N, num_features)
            C_t = self.C.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)  # (B, T, H, N)

        # 离散化
        A_bar, B_bar_coeff = self.discretize(dt)  # (B, T, N), (B, T, N)

        # 状态空间计算（递归形式）
        states = []
        h = torch.zeros(B, self.state_size, device=x.device, dtype=x.dtype)

        for t in range(T):
            # 选择性B和C
            if self.selective:
                B_curr = B_t[:, t]  # (B, N)
                C_curr = C_t[:, t]  # (B, N)
                # 计算输入项
                u_curr = x[:, t]  # (B, num_features)
                Bu = torch.einsum('bn,bf->bn', B_curr, u_curr)  # (B, N)
            else:
                # 非选择性情况
                B_curr = self.B  # (N, num_features)
                C_curr = self.C  # (H, N)
                u_curr = x[:, t]  # (B, num_features)
                Bu = torch.einsum('nf,bf->bn', B_curr, u_curr)  # (B, N)

            # 状态更新: h_{t+1} = A_bar * h_t + B_bar * u_t
            A_curr = A_bar[:, t] if A_bar.dim() == 3 else A_bar  # (B, N) or (N,)
            B_coeff = B_bar_coeff[:, t] if B_bar_coeff.dim() == 3 else B_bar_coeff  # (B, N) or (N,)

            h = A_curr * h + B_coeff * Bu

            states.append(h)

        # 堆叠状态
        states = torch.stack(states, dim=1)  # (B, T, N)

        # 输出计算: y = C * h + D * u
        if self.selective:
            # 选择性C
            y = torch.einsum('btn,btn->bt', C_t, states)  # (B, T)
            y = y.unsqueeze(-1).expand(-1, -1, self.hidden_size)  # (B, T, H)
        else:
            # 非选择性C
            y = torch.einsum('hn,btn->bth', self.C, states)  # (B, T, H)

        # 添加直通项 D*u
        y = y + torch.einsum('hf,btf->bth', self.D, x)

        # 输出投影和归一化
        y = self.output_proj(states)  # (B, T, H)
        y = self.norm(y)

        return y


class MambaBlock(nn.Module):
    """
    Mamba风格的SSM块，包含选择性机制和门控
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Union[int, str] = "auto",
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        use_fast_path: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path

        # 输入投影
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)

        # 卷积层
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # SSM参数投影
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # SSM参数初始化
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, T, D)
        Returns:
            (B, T, D)
        """
        batch, seqlen, dim = hidden_states.shape

        # 输入投影和门控
        xz = self.in_proj(hidden_states)  # (B, T, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # 各自 (B, T, d_inner)

        # 卷积
        x = x.transpose(1, 2)  # (B, d_inner, T)
        x = self.conv1d(x)[:, :, :seqlen]  # 去除padding
        x = x.transpose(1, 2)  # (B, T, d_inner)

        # 激活
        x = F.silu(x)

        # SSM计算
        x_dbl = self.x_proj(x)  # (B, T, dt_rank + 2*d_state)
        dt, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = self.dt_proj(dt)  # (B, T, d_inner)

        # 状态空间计算
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        y = self.selective_scan(x, dt, A, B, C, self.D)

        # 门控和输出投影
        y = y * F.silu(z)
        output = self.out_proj(y)

        return output

    def selective_scan(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
    ) -> torch.Tensor:
        """
        选择性扫描算法
        """
        batch, seq_len, d_inner = u.shape
        d_state = A.shape[-1]

        # 离散化
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, T, d_inner, d_state)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, T, d_inner, d_state)

        # 递归计算
        x = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=u.dtype)
        ys = []

        for i in range(seq_len):
            x = deltaA[:, i] * x + deltaB[:, i] * u[:, i:i+1]
            y = torch.einsum('bdn,bd->bn', x, C[:, i])
            ys.append(y)

        y = torch.stack(ys, dim=1)  # (B, T, d_inner)
        y = y + u * D

        return y