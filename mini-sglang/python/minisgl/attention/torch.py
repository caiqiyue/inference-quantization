"""Correctness-first CUDA attention backend without FlashInfer or FlashAttention.

This backend is intentionally for learning and smoke tests.  It materializes each
request's KV sequence and uses ordinary PyTorch CUDA tensor operations, so it is
not suitable for performance measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from minisgl.core import Batch, get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from .base import BaseAttnBackend, BaseAttnMetadata

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


@dataclass
class TorchMetadata(BaseAttnMetadata):
    reqs: list

    def get_last_indices(self, bs: int) -> torch.Tensor:
        lengths = [req.extend_len for req in self.reqs[:bs]]
        return torch.tensor(lengths, device=get_global_ctx().kv_cache.device, dtype=torch.long).cumsum(0) - 1


class TorchAttentionBackend(BaseAttnBackend):
    """Paged-KV attention implemented with regular PyTorch CUDA operations."""

    def __init__(self, config: ModelConfig) -> None:
        self.kvcache = get_global_ctx().kv_cache
        self.device = self.kvcache.device
        tp_size = get_tp_info().size
        self.num_qo_heads = div_even(config.num_qo_heads, tp_size)
        self.num_kv_heads = div_even(config.num_kv_heads, tp_size, allow_replicate=True)
        self.head_dim = config.head_dim
        assert self.num_qo_heads % self.num_kv_heads == 0
        self.scale = self.head_dim**-0.5

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        assert isinstance(metadata, TorchMetadata)

        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        flat_k = self.kvcache.k_cache(layer_id).view(-1, self.num_kv_heads, self.head_dim)
        flat_v = self.kvcache.v_cache(layer_id).view(-1, self.num_kv_heads, self.head_dim)
        # Ordinary indexed writes replace mini-sglang's JIT store kernel.
        flat_k[batch.out_loc] = k
        flat_v[batch.out_loc] = v

        outputs: list[torch.Tensor] = []
        offset = 0
        page_table = get_global_ctx().page_table
        group_size = self.num_qo_heads // self.num_kv_heads
        for req in metadata.reqs:
            q_len = req.extend_len
            q_req = q[offset : offset + q_len]
            offset += q_len
            cache_locations = page_table[req.table_idx, : req.device_len]
            k_req = flat_k[cache_locations].repeat_interleave(group_size, dim=1)
            v_req = flat_v[cache_locations].repeat_interleave(group_size, dim=1)

            q_heads = q_req.transpose(0, 1)
            k_heads = k_req.transpose(0, 1)
            v_heads = v_req.transpose(0, 1)
            scores = torch.matmul(q_heads, k_heads.transpose(-1, -2)) * self.scale
            query_positions = torch.arange(req.cached_len, req.device_len, device=self.device)
            key_positions = torch.arange(req.device_len, device=self.device)
            scores.masked_fill_(key_positions.unsqueeze(0) > query_positions.unsqueeze(1), float("-inf"))
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(v_heads.dtype)
            outputs.append(torch.matmul(probs, v_heads).transpose(0, 1))
        return torch.cat(outputs, dim=0)

    def prepare_metadata(self, batch: Batch) -> None:
        batch.attn_metadata = TorchMetadata(reqs=batch.padded_reqs)

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        raise RuntimeError("The torch attention backend requires --cuda-graph-max-bs 0")

    def prepare_for_capture(self, batch: Batch) -> None:
        raise RuntimeError("The torch attention backend does not support CUDA graphs")

    def prepare_for_replay(self, batch: Batch) -> None:
        raise RuntimeError("The torch attention backend does not support CUDA graphs")
