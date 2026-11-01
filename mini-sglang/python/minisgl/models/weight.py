from __future__ import annotations

import glob
import re
from typing import Dict, Iterator, Tuple

import safetensors
import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import cached_load_hf_config, div_ceil, download_hf_weight
from tqdm import tqdm

_SPLIT_DIM_0 = [".q_proj", ".k_proj", ".v_proj", ".gate_proj", ".up_proj"]
_SPLIT_DIM_1 = [".o_proj", ".down_proj"]

# Merge groups: individual projections -> fused projection
_MERGE_GROUPS = {
    ".q_proj": (".qkv_proj", ("q", "k", "v")),
    ".k_proj": (".qkv_proj", ("q", "k", "v")),
    ".v_proj": (".qkv_proj", ("q", "k", "v")),
    ".gate_proj": (".gate_up_proj", ("gate", "up")),
    ".up_proj": (".gate_up_proj", ("gate", "up")),
}
_SLOT_NAMES = {
    ".q_proj": "q",
    ".k_proj": "k",
    ".v_proj": "v",
    ".gate_proj": "gate",
    ".up_proj": "up",
}
_EXPERT_PATTERN = re.compile(r"^(?P<prefix>.+\.experts)\.(?P<idx>\d+)\.(?P<name>.+)$")


def _shard_tensor(key: str, value: torch.Tensor, r: int, n: int, num_kv_heads: int):
    """Extract rank r's shard from a single tensor. Returns a contiguous copy."""
    if any(key.count(sub) for sub in _SPLIT_DIM_0):
        is_kv_proj = any(key.count(sub) for sub in (".k_proj", ".v_proj"))
        if is_kv_proj and num_kv_heads is not None and num_kv_heads < n:
            head_dim = value.shape[0] // num_kv_heads
            head_idx = r * num_kv_heads // n
            return value[head_idx * head_dim : (head_idx + 1) * head_dim].clone()
        return value.chunk(n, dim=0)[r].clone()
    elif any(key.count(sub) for sub in _SPLIT_DIM_1):
        return value.chunk(n, dim=1)[r].clone()
    elif key.count("lm_head") or key.count("embed_tokens"):
        num_embeddings = value.shape[0]
        num_embeddings_per_partition = div_ceil(num_embeddings, n)
        vocab_start_idx = r * num_embeddings_per_partition
        vocab_end_idx = min((r + 1) * num_embeddings_per_partition, num_embeddings)
        return value[vocab_start_idx:vocab_end_idx, :].clone()
    else:
        return value


def _get_merge_info(key: str):
    """If key belongs to a merge group, return (merged_key, slot, all_slots). Else None."""
    for suffix, (fused_suffix, slots) in _MERGE_GROUPS.items():
        if key.count(suffix):
            return key.replace(suffix, fused_suffix), _SLOT_NAMES[suffix], slots
    return None


def _get_expert_stack_info(key: str) -> tuple[str, int] | None:
    """Map an expert-scoped checkpoint key to the packed runtime key."""
    match = _EXPERT_PATTERN.match(key)
    if match is None:
        return None

    packed_name = match.group("name")
    if packed_name.endswith(".weight"):
        packed_name = packed_name.removesuffix(".weight")
    return f"{match.group('prefix')}.{packed_name}", int(match.group("idx"))


def load_weight(model_path: str, device: torch.device) -> Iterator[Tuple[str, torch.Tensor]]:
    """Streaming weight loader. Yields (name, tensor) pairs already sharded, merged,
    and on device. Peak CPU memory: one full tensor + a small merge buffer."""
    from .config import ModelConfig

    model_folder = download_hf_weight(model_path)
    config = ModelConfig.from_hf(cached_load_hf_config(model_path))
    files = glob.glob(f"{model_folder}/*.safetensors")
    files = [f for f in files if not f.endswith("consolidated.safetensors")] or files
    tp_info = get_tp_info()

    # Buffer for merge groups: merged_key -> {slot: tensor}
    merge_buf: Dict[str, Dict[str, torch.Tensor]] = {}
    expert_buf: Dict[str, Dict[int, torch.Tensor]] = {}
    for file in tqdm(files, desc="Loading weights", disable=not tp_info.is_primary()):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for name in f.keys():
                # Strip multimodal wrapper prefix, skip vision/projector weights
                if name.startswith(("vision_tower.", "multi_modal_projector.")):
                    continue
                raw = f.get_tensor(name)
                name = name.removeprefix("language_model.")
                tensor = _shard_tensor(name, raw, tp_info.rank, tp_info.size, config.num_kv_heads)
                del raw

                if (info := _get_merge_info(name)) is None:
                    out = (name, tensor)
                else:
                    merged_key, slot, all_slots = info
                    merge_buf.setdefault(merged_key, {})[slot] = tensor
                    if not all(s in merge_buf[merged_key] for s in all_slots):
                        continue
                    parts = [merge_buf[merged_key][s] for s in all_slots]
                    del merge_buf[merged_key]
                    out = (merged_key, torch.cat(parts, dim=0))

                if config.is_moe and (expert_info := _get_expert_stack_info(out[0])) is not None:
                    packed_key, expert_idx = expert_info
                    slots = expert_buf.setdefault(packed_key, {})
                    slots[expert_idx] = out[1]
                    if len(slots) != config.num_experts:
                        continue
                    experts = [slots[idx] for idx in range(config.num_experts)]
                    del expert_buf[packed_key]
                    yield packed_key, torch.stack(experts, dim=0)
                else:  # Normal dense model
                    yield out[0], out[1]

    assert not merge_buf, f"Incomplete merge groups in checkpoint: {list(merge_buf.keys())}"
    assert not expert_buf, f"Incomplete expert tensors in checkpoint: {list(expert_buf.keys())}"


_AWQ_PACK_NUM = 8  # INT4 values per int32 word in llm-awq's v2 layout.
_AWQ_LAYOUT_SUFFIXES = ("qweight", "scales", "scaled_zeros")


def awq_scale_rows(in_features: int, group_size: int) -> int:
    """Return the first dimension of llm-awq v2 scales and zero tensors.

    llm-awq pads the number of groups to satisfy the CUDA kernel's packing
    requirements.  It is *not* simply ``in_features // group_size``.  Keeping
    this formula here makes the mini-sglang adapter validate the same physical
    layout that the AWQ exporter writes.
    """
    if group_size not in (32, 64, 128):
        raise ValueError(f"Unsupported AWQ group size: {group_size}")
    if in_features % group_size:
        raise ValueError(
            f"AWQ input width {in_features} is not divisible by group size {group_size}"
        )
    multiplier = {128: 1, 64: 2, 32: 4}[group_size]
    groups = in_features // group_size
    # llm-awq's historical ``make_divisible`` is a ceil-division helper:
    # it first counts 8-group buckets, rather than rounding group count up to
    # a multiple of eight.
    padded_groups = div_ceil(groups, _AWQ_PACK_NUM)
    padded_groups = div_ceil(padded_groups, multiplier) * multiplier
    return padded_groups * _AWQ_PACK_NUM


def _validate_awq_linear(name: str, tensors: Dict[str, torch.Tensor], group_size: int) -> None:
    missing = [suffix for suffix in _AWQ_LAYOUT_SUFFIXES if suffix not in tensors]
    if missing:
        raise ValueError(f"{name}: AWQ checkpoint is missing {', '.join(missing)}")

    qweight = tensors["qweight"]
    scales = tensors["scales"]
    scaled_zeros = tensors["scaled_zeros"]
    if qweight.dtype != torch.int16 or qweight.ndim != 2:
        raise ValueError(f"{name}.qweight: expected a 2-D int16 packed tensor, got {qweight.shape}/{qweight.dtype}")
    if qweight.shape[1] % group_size:
        raise ValueError(f"{name}.qweight: input width {qweight.shape[1]} is incompatible with group size {group_size}")
    out_features = qweight.shape[0] * 4
    expected_scale_shape = (awq_scale_rows(qweight.shape[1], group_size), out_features)
    for suffix, tensor in (("scales", scales), ("scaled_zeros", scaled_zeros)):
        if tensor.shape != expected_scale_shape:
            raise ValueError(
                f"{name}.{suffix}: expected {expected_scale_shape} for llm-awq v2, got {tuple(tensor.shape)}"
            )
        if not tensor.is_floating_point():
            raise ValueError(f"{name}.{suffix}: expected a floating-point tensor, got {tensor.dtype}")
    if scales.dtype != scaled_zeros.dtype:
        raise ValueError(f"{name}: scales dtype {scales.dtype} differs from scaled_zeros dtype {scaled_zeros.dtype}")
    if "bias" in tensors and tensors["bias"].shape != (out_features,):
        raise ValueError(f"{name}.bias: expected {(out_features,)}, got {tuple(tensors['bias'].shape)}")


def _merge_awq_projections(
    state_dict: Dict[str, torch.Tensor],
    *,
    base: str,
    parts: tuple[str, ...],
    fused: str,
    group_size: int,
) -> None:
    """Map AWQ's logical projections to mini-sglang's fused projection.

    AWQ packs every four output rows.  Concatenating qweight is valid only at
    complete four-row boundaries, so the adapter verifies each source tensor
    before forming the fused tensor.  This is deliberately AWQ-specific; it
    must not be reused for a different quantization format.
    """
    first_key = f"{base}{parts[0]}.qweight"
    if first_key not in state_dict:
        return

    source: Dict[str, Dict[str, torch.Tensor]] = {}
    for part in parts:
        name = f"{base}{part}"
        tensors: Dict[str, torch.Tensor] = {}
        for suffix in _AWQ_LAYOUT_SUFFIXES:
            key = f"{name}.{suffix}"
            if key not in state_dict:
                raise ValueError(f"{name}: incomplete AWQ projection group; missing {key}")
            tensors[suffix] = state_dict.pop(key)
        bias_key = f"{name}.bias"
        if bias_key in state_dict:
            tensors["bias"] = state_dict.pop(bias_key)
        source[part] = tensors
        _validate_awq_linear(name, tensors, group_size)

    has_bias = ["bias" in source[part] for part in parts]
    if any(has_bias) and not all(has_bias):
        raise ValueError(f"{base}: AWQ fusion group mixes biased and bias-free projections")
    input_widths = {source[part]["qweight"].shape[1] for part in parts}
    scale_rows = {source[part]["scales"].shape[0] for part in parts}
    dtypes = {source[part]["scales"].dtype for part in parts}
    if len(input_widths) != 1 or len(scale_rows) != 1 or len(dtypes) != 1:
        raise ValueError(f"{base}: AWQ fusion group has incompatible input layout or scale dtype")

    fused_base = f"{base}{fused}"
    state_dict[f"{fused_base}.qweight"] = torch.cat(
        [source[part]["qweight"] for part in parts], dim=0
    )
    for suffix in ("scales", "scaled_zeros"):
        state_dict[f"{fused_base}.{suffix}"] = torch.cat(
            [source[part][suffix] for part in parts], dim=1
        )
    if all(has_bias):
        state_dict[f"{fused_base}.bias"] = torch.cat(
            [source[part]["bias"] for part in parts], dim=0
        )

    _validate_awq_linear(
        fused_base,
        {
            suffix: state_dict[f"{fused_base}.{suffix}"]
            for suffix in _AWQ_LAYOUT_SUFFIXES
        }
        | ({"bias": state_dict[f"{fused_base}.bias"]} if all(has_bias) else {}),
        group_size,
    )


def load_awq_weight(checkpoint_path: str, group_size: int = 128) -> Iterator[Tuple[str, torch.Tensor]]:
    """Adapt a native ``llm-awq`` v2 checkpoint to mini-sglang's state dict.

    The input is intentionally the native AWQ artifact rather than a generic
    quantization file.  It validates AWQ's packed layout and then maps its
    separate Q/K/V and Gate/Up projections to the fused modules expected by
    mini-sglang.
    """
    state_dict = dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True))
    for layer in range(10_000):
        prefix = f"model.layers.{layer}."
        has_q_proj = f"{prefix}self_attn.q_proj.qweight" in state_dict
        has_gate_proj = f"{prefix}mlp.gate_proj.qweight" in state_dict
        if not has_q_proj and not has_gate_proj:
            if layer == 0:
                break
            continue
        _merge_awq_projections(
            state_dict,
            base=f"{prefix}self_attn.",
            parts=("q_proj", "k_proj", "v_proj"),
            fused="qkv_proj",
            group_size=group_size,
        )
        _merge_awq_projections(
            state_dict,
            base=f"{prefix}mlp.",
            parts=("gate_proj", "up_proj"),
            fused="gate_up_proj",
            group_size=group_size,
        )

    for name, tensor in state_dict.items():
        if name.endswith(".qweight"):
            base = name.removesuffix(".qweight")
            _validate_awq_linear(
                base,
                {
                    suffix: state_dict[f"{base}.{suffix}"]
                    for suffix in _AWQ_LAYOUT_SUFFIXES
                }
                | ({"bias": state_dict[f"{base}.bias"]} if f"{base}.bias" in state_dict else {}),
                group_size,
            )
        yield name, tensor
