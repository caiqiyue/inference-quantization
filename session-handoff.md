# Session Handoff

## Current AWQ Adapter Evidence

The verified AWQ path is intentionally specific: native `llm-awq v2` W4A16,
group 128, TP=1, then the dedicated AWQ-to-mini-sglang adapter.  It is not a
generic quantization-format interface; GPTQ, FP8/NVFP4, GGUF and AWQ tensor
parallel remain unsupported.

- ModelScope returned 404 for the Llama2-7B candidate.  The public fallback
  `/home/qiyue.cai/host_workspace/models/Qwen2.5-0.5B-Instruct` was downloaded
  instead.
- Default AWQ search (auto-scale plus MSE clipping) completed all 24 layers.
  The native search artifact and packed W4G128 checkpoint are stored outside
  Git under `/home/qiyue.cai/host_workspace/models/Qwen2.5-0.5B-Instruct-awq/`.
- `runs/20261101-063118-native-awq-65c3132f/` proves the complete checkpoint
  loads and generates through native AWQ without mini-sglang.
- `runs/20261101-062735-awq-adapter-90196dd8/` compares native `WQLinear` with
  the adapter: QKV and Gate/Up match with zero absolute error for both 1-token
  GEMV and 8-token GEMM.
- `runs/20261101-063001-ebfb4f7d/` records mini-sglang loading the same
  checkpoint and completing 2/2 Harness API requests.  Its config is
  `configs/experiments/awq-qwen2.5-0.5b-smoke.json`, marked
  `artifact.runtime_compatible=true`.

The adapter validates AWQ's actual `qweight/scales/scaled_zeros` layout and
merges Q/K/V and Gate/Up (including bias).  `--awq-group-size` is now explicit.
For 896 input features AWQ metadata has 8 rows; for 4864 input features it has
40 rows—do not substitute a naive `input_width / group_size` formula.

Quality and performance remain unverified.  The API reports zero completion
tokens, so no throughput conclusion may be drawn.  If quality becomes in
scope, first add a fixed-tokenization continuation/PPL evaluator and compare
BF16, native AWQ and mini-sglang AWQ under identical inputs.

## Verified Now

`./init.sh`、`python -m unittest tests.test_harness`、AWQ smoke 配置的 `validate` 与 `plan` 均通过；所有实验配置均以 mini-sglang 为运行时。`harness-002` 已标记为 `passing`。`sglang/` 源码与 `llm-awq/tinychat/` 已删除；保留 mini-sglang、llm-awq 算法代码及 AWQ/mini-sglang 个人 Conda 环境。

## Known Limitations

默认 FlashInfer 路径仍未完成真实生成；已验证普通 PyTorch CUDA fallback（`--attn torch`、CUDA graph disabled），并固定 1024-token KV cache。API usage 字段仍可能为 0，因此现有 run 的 tokens/s 不能用作性能或质量结论。

已验证 BF16 fallback 的 100 请求稳定性：`runs/20261031-150415-c2847c41/` 包含 100 条非空响应、0 HTTP 错误、全部正常结束；它不是正式吞吐或质量评测。

GPTQ、FP8/NVFP4、GGUF、其他 AWQ layout 与 AWQ tensor parallel 未支持。它们不能复用本专用 adapter，也不能标记为 `runtime_compatible=true`。

## Next Best Step

只有在质量进入范围后，再实现固定 tokenization 的 continuation/PPL evaluator，并在相同模型、prompt、dtype 和采样参数下比较 BF16、native AWQ 与 mini-sglang AWQ。若 native AWQ 正常而 mini-sglang 异常，先检查 tokenizer/chat template、RoPE、attention/KV cache 与 generation scheduler；不要先扩展为通用 adapter。

## Earlier Qwen3 AWQ Smoke (secondary evidence)

已完成 Qwen3-8B 的 AWQ v2 W4 group-128 产物生成，产物位于仓库外 `/home/qiyue.cai/host_workspace/models/Qwen3-8B-awq/`。mini-sglang 通过 `--awq-checkpoint` 可加载该产物，并使用单卡、torch attention、CUDA graph disabled 启动成功。最终 Harness smoke run 为 `runs/20261101-045028-133b80f6/`，2/2 请求返回且 finish_reason=stop，平均延迟 1365.62 ms。

按当前功能验收，AWQ checkpoint 已被 mini-sglang 加载并用于 OpenAI-compatible API 推理，`configs/experiments/awq.json` 已设为 `artifact.runtime_compatible=true`。当前 AWQ search 使用本地 WikiText-2 calibration，并关闭旧版 scale/clip 搜索（`correctness-first-no-scale-no-clip`）；输出质量和性能结果不在本 feature 的验收范围内。

该 Qwen3 smoke 早于 Qwen2.5 的逐层对齐证据，仅保留为历史参考；当前 AWQ 功能验收应以本文件顶部列出的 Qwen2.5 原生、adapter 与 Harness 三段证据为准。

已验证的 BF16 参考运行只使用 GPU 0：

```bash
MINISGL_TORCH_FALLBACK=1 CUDA_VISIBLE_DEVICES=0 \
  /home/qiyue.cai/.conda/envs/mini-sglang/bin/python -m minisgl \
  --model /home/qiyue.cai/host_workspace/models/Qwen3-8B \
  --host 127.0.0.1 --port 30000 --tp-size 1 --dtype bfloat16 \
  --memory-ratio 0.7 --num-pages 1024 --max-seq-len-override 1024 \
  --attn torch --cuda-graph-max-bs 0

python -m harness.cli run --config configs/experiments/qwen3-8b-wikitext2-100-torch.json --server-url http://127.0.0.1:30000
```

环境验证：

```bash
PYTHONPATH="$PWD/llm-awq/awq/kernels" /home/qiyue.cai/.conda/envs/awq/bin/python scripts/gpu_smoke.py
/home/qiyue.cai/.conda/envs/mini-sglang/bin/python scripts/gpu_smoke.py
/home/qiyue.cai/.conda/envs/mini-sglang/bin/python -m minisgl --help
```

## Commands

```bash
./init.sh
python -m harness.cli validate --config configs/experiments/baseline.json
python -m harness.cli plan --config configs/experiments/baseline.json
python -m unittest tests.test_harness
python -m harness.cli validate --config configs/experiments/awq-qwen2.5-0.5b-smoke.json
python -m harness.cli plan --config configs/experiments/awq-qwen2.5-0.5b-smoke.json
```
