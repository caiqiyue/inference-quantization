from __future__ import annotations

from pathlib import Path
from typing import List
import os
import sys

import torch
import torch.nn.functional as F
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.utils import div_ceil, div_even

from .base import BaseOP


def awq_scale_rows(in_features: int, group_size: int) -> int:
    """Match llm-awq v2's padded scale/zero layout."""
    if group_size not in (32, 64, 128):
        raise ValueError(f"Unsupported AWQ group size: {group_size}")
    if in_features % group_size:
        raise ValueError(
            f"AWQ input width {in_features} is not divisible by group size {group_size}"
        )
    multiplier = {128: 1, 64: 2, 32: 4}[group_size]
    groups = in_features // group_size
    padded_groups = div_ceil(groups, 8)
    padded_groups = div_ceil(padded_groups, multiplier) * multiplier
    return padded_groups * 8


class _LinearTPImpl(BaseOP):
    """Real implementation of a linear layer with tensor parallelism."""

    def __init__(
        self,
        full_isize: int,
        full_osize: int,
        local_isize: int,
        local_osize: int,
        has_bias: bool,
        awq: bool = False,
        awq_group_size: int = 128,
        awq_dtype: torch.dtype = torch.float16,
    ):
        self.full_input_size = full_isize
        self.full_output_size = full_osize
        self.local_input_size = local_isize
        self.local_output_size = local_osize
        self.awq = awq
        if awq:
            if get_tp_info().size != 1:
                raise NotImplementedError("AWQ smoke currently supports tp-size=1 only")
            self.group_size = awq_group_size
            self.qweight = torch.empty(local_osize // 4, local_isize, dtype=torch.int16)
            scale_shape = (awq_scale_rows(local_isize, self.group_size), local_osize)
            self.scales = torch.empty(scale_shape, dtype=awq_dtype)
            self.scaled_zeros = torch.empty(scale_shape, dtype=awq_dtype)
        else:
            self.weight = torch.empty(local_osize, local_isize)
        self.bias = torch.empty(local_osize) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.awq:
            return F.linear(x, self.weight, self.bias)
        kernel_path = os.environ.get("MINISGL_AWQ_KERNEL_PATH", str(Path(__file__).resolve().parents[4] / "llm-awq" / "awq" / "kernels"))
        if kernel_path not in sys.path:
            sys.path.insert(0, kernel_path)
        import awq_inference_engine
        rows = x.numel() // x.shape[-1]
        if rows < 8:
            y = awq_inference_engine.gemv_forward_cuda_new(x, self.qweight, self.scales, self.scaled_zeros, rows, self.full_output_size, self.full_input_size, self.group_size)
        else:
            y = awq_inference_engine.gemm_forward_cuda_new(x, self.qweight, self.scales, self.scaled_zeros)
        return y + self.bias if self.bias is not None else y


class LinearReplicated(_LinearTPImpl):
    """
    Linear layer where weights are replicated (not sharded) across all TP ranks.
    Each GPU holds the full weight matrix.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
        awq: bool = False,
        awq_group_size: int = 128,
        awq_dtype: torch.dtype = torch.float16,
    ):
        super().__init__(
            full_isize=input_size,
            full_osize=output_size,
            local_isize=input_size,
            local_osize=output_size,
            has_bias=has_bias, awq=awq, awq_group_size=awq_group_size, awq_dtype=awq_dtype,
        )


class LinearColParallelMerged(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_sizes: List[int],
        has_bias: bool,
        awq: bool = False,
        awq_group_size: int = 128,
        awq_dtype: torch.dtype = torch.float16,
    ):
        # check that all output sizes are divisible by tp_size
        tp_info = get_tp_info()
        tp_output_sizes = [div_even(size, tp_info.size) for size in output_sizes]
        output_size = sum(output_sizes)
        tp_output_size = sum(tp_output_sizes)
        super().__init__(input_size, output_size, input_size, tp_output_size, has_bias, awq, awq_group_size, awq_dtype)


class LinearQKVMerged(_LinearTPImpl):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_qo_heads: int,
        num_kv_heads: int,
        has_bias: bool,
        awq: bool = False,
        awq_group_size: int = 128,
        awq_dtype: torch.dtype = torch.float16,
    ):
        tp_info = get_tp_info()

        local_num_qo = div_even(num_qo_heads, tp_info.size)
        local_num_kv = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        full_isize = hidden_size
        full_osize = (num_qo_heads + 2 * num_kv_heads) * head_dim
        local_isize = hidden_size
        local_osize = (local_num_qo + 2 * local_num_kv) * head_dim
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias, awq, awq_group_size, awq_dtype)


class LinearOProj(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
        awq: bool = False,
        awq_group_size: int = 128,
        awq_dtype: torch.dtype = torch.float16,
    ):
        tp_info = get_tp_info()
        full_isize = input_size
        full_osize = output_size
        local_isize = div_even(input_size, tp_info.size)
        local_osize = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias, awq, awq_group_size, awq_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y


class LinearRowParallel(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
        awq: bool = False,
        awq_group_size: int = 128,
        awq_dtype: torch.dtype = torch.float16,
    ):
        tp_info = get_tp_info()
        local_input_size = div_even(input_size, tp_info.size)
        local_output_size = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(input_size, output_size, local_input_size, local_output_size, has_bias, awq, awq_group_size, awq_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y
