from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import torch


def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    if os.environ.get("MINISGL_TORCH_FALLBACK"):
        gate, up = x.chunk(2, dim=-1)
        result = torch.nn.functional.silu(gate) * up
        if out is not None:
            out.copy_(result)
            return out
        return result
    from flashinfer import silu_and_mul

    return silu_and_mul(x, out=out)


def gelu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    if os.environ.get("MINISGL_TORCH_FALLBACK"):
        gate, up = x.chunk(2, dim=-1)
        result = torch.nn.functional.gelu(gate) * up
        if out is not None:
            out.copy_(result)
            return out
        return result
    from flashinfer import gelu_and_mul

    return gelu_and_mul(x, out=out)


__all__ = ["silu_and_mul", "gelu_and_mul"]
