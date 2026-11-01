import os
from typing import Tuple

import torch

from .base import BaseOP


class RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = None
        if not os.environ.get("MINISGL_TORCH_FALLBACK"):
            from flashinfer import rmsnorm

            self.rmsnorm = rmsnorm

    def _torch_rmsnorm(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        return (x * torch.rsqrt(variance + self.eps).to(x.dtype)) * self.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._torch_rmsnorm(x) if self.rmsnorm is None else self.rmsnorm(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        if self.rmsnorm is None:
            x.copy_(self._torch_rmsnorm(x))
        else:
            self.rmsnorm(x, self.weight, self.eps, out=x)


class RMSNormFused(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = None
        self.fused_add_rmsnorm = None
        if not os.environ.get("MINISGL_TORCH_FALLBACK"):
            from flashinfer import fused_add_rmsnorm, rmsnorm

            self.rmsnorm = rmsnorm
            self.fused_add_rmsnorm = fused_add_rmsnorm

    def _torch_rmsnorm(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        return (x * torch.rsqrt(variance + self.eps).to(x.dtype)) * self.weight

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self._torch_rmsnorm(x) if self.rmsnorm is None else self.rmsnorm(x, self.weight, self.eps), x
        if self.rmsnorm is None:
            residual.add_(x)
            return self._torch_rmsnorm(residual), residual
        self.fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x, residual
