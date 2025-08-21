from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MovingAverageDecomposition(nn.Module):
    """简单的趋势-残差分解：y = x - MA(x), trend = MA(x)
    method: 'ma'（移动平均）或 'ets'（指数平滑，近似）
    """
    def __init__(self, kernel: int = 25, method: str = 'ma', alpha: float = 0.3):
        super().__init__()
        self.kernel = int(kernel)
        self.method = str(method)
        self.alpha = float(alpha)
        # MA 使用平均卷积核
        if self.method == 'ma':
            k = torch.ones(1, 1, self.kernel) / float(self.kernel)
            self.register_buffer('ma_kernel', k)
        else:
            self.register_buffer('ma_kernel', None)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B,T,D)
        if self.method == 'ma':
            y = x.transpose(1,2)
            trend = F.conv1d(y, self.ma_kernel.expand(y.size(1),1,-1), padding=self.kernel//2, groups=y.size(1))
            trend = trend.transpose(1,2)
        else:
            # 指数平滑近似（逐步更新）
            trend = []
            s = x[:,0,:]
            trend.append(s.unsqueeze(1))
            a = self.alpha
            for t in range(1, x.size(1)):
                s = a * x[:,t,:] + (1-a) * s
                trend.append(s.unsqueeze(1))
            trend = torch.cat(trend, dim=1)
        residual = x - trend
        return residual, trend

