# Progress Log

## Current Verified State

- 已建立以 mini-sglang + OpenAI-compatible API 为核心的 harness 骨架。
- `validate`、`plan`、配置加载、结果 manifest 和 no-GPU 单元测试已实现。
- 实验协议已显式记录 `algorithm`、`artifact`、`runtime`，要求量化产物先通过 mini-sglang loader/kernel 兼容性检查。
- qiyue.cai 容器内已创建 AWQ 与 mini-sglang 个人 Conda 环境；AWQ kernel 与 mini-sglang GPU smoke 均已通过。
- mini-sglang 已作为默认运行时集成到 Harness；它支持已验证的单卡 `llm-awq v2` W4A16 量化权重加载。GPTQ、NVFP4、GGUF 与 AWQ tensor parallel 尚未支持。
- Qwen3-8B BF16 + WikiText-2 baseline 配置已准备；模型已完成下载并通过完整性校验，真实服务运行仍待完成。
- Hugging Face 不可达后，已改用可访问的 ModelScope CDN 下载 `Qwen/Qwen3-8B` 到 `/home/qiyue.cai/host_workspace/models/Qwen3-8B`。16 个文件均通过官方 SHA-256；禁用 GPU 的 config/tokenizer smoke 确认 `model_type=qwen3`、5 个权重分片齐全。
- `/home/qiyue.cai/.conda/envs/mini-sglang` 已安装官方 mini-sglang 源码、PyTorch 2.7.1+cu128、FlashInfer 和 GPU JIT 所需依赖；`python -m minisgl --help` 与 `scripts/gpu_smoke.py` 已通过。
- 已为学习/验证加入 correctness-first 的普通 PyTorch CUDA fallback：`--attn torch`，关闭 CUDA graph，并通过 `MINISGL_TORCH_FALLBACK=1` 避开 FlashInfer/FlashAttention 的 attention、RoPE、RMSNorm 与 SwiGLU 路径。它不用于性能结论。
- 单卡 GPU 0 已在该 fallback 下成功加载 Qwen3-8B BF16、完成 1-token 真实生成，并通过两条 WikiText-2 Harness smoke。结果位于 `runs/20261031-144928-f34e77c8/`；服务 API 未报告 token usage，故其中 tokens/s=0 不能作为性能指标。所有测试进程已停止，两张 GPU 均恢复到 2 MiB 系统占用。
- 已完成 100 条 WikiText-2 可靠性测试：`runs/20261031-150415-c2847c41/` 有 100/100 条非空响应、0 HTTP 错误、全部 `finish_reason=stop`。平均客户端延迟 390.42 ms，中位数 380.28 ms；API usage 仍为 0，因此 tokens/s 不可用于性能结论。测试仅使用 GPU 0，完成后已释放两张 GPU。
- 修复 `run_benchmark`：实验配置中的相对 dataset 路径现在一律相对配置文件解析；`python -m unittest tests.test_harness` 通过（4 tests）。
- 已生成 Qwen3-8B W4 group-128 的 llm-awq v2 checkpoint（仓库外），并补充本地 JSONL calibration、Qwen3 AWQ 层识别和 Transformers 新接口兼容。
- mini-sglang 已能加载该 AWQ checkpoint、完成 qkv/gate-up fused state mapping、初始化 KV cache 并启动 API；AWQ Harness smoke 已落盘到 `runs/20261101-032713-a19e18e8/`。
- 当前 smoke 请求能完成且无 HTTP 错误。按当前功能验收，AWQ checkpoint 可由 mini-sglang 加载并用于 API 推理，`runtime_compatible=true`；最终可追溯运行是 `runs/20261101-045028-133b80f6/`（2/2 OpenAI-compatible 请求完成）。输出质量和性能有效性不在本 feature 的验收范围内，后续另行验证。
- ModelScope 上 Llama2-7B 的候选仓库下载接口返回 404，故按回退策略下载公开的 `Qwen/Qwen2.5-0.5B-Instruct`（仓库外，约 954MB）。它使用原生 `Qwen2ForCausalLM` 路径、24 层、BF16。
- Qwen2.5-0.5B 的默认 AWQ search（含 auto-scale 和 MSE clipping）已完成 24/24 层，并导出真实 W4G128 llm-awq v2 checkpoint（约 324MB）。search 与 checkpoint 均位于仓库外 `/home/qiyue.cai/host_workspace/models/Qwen2.5-0.5B-Instruct-awq/`。
- 已把 AWQ 加载收敛为专用 adapter：校验原生 `qweight/scales/scaled_zeros` packing 布局、显式传递 group size、融合 Q/K/V 和 Gate/Up 时同时融合 bias，并按 checkpoint dtype 分配量化元数据。此前 mini-sglang 对 `scales` 的形状假设不符合部分 AWQ 输入维度；现以 llm-awq 实际 ceil-division/bucket 规则修复并有无 GPU 结构测试覆盖。
- 原生 AWQ `WQLinear` 与 adapter 的 QKV/Gate-Up 输出在 GEMV（1 token）及 GEMM（8 token）两条路径上均为 0 误差，证据在 `runs/20261101-062735-awq-adapter-90196dd8/`。不经过 mini-sglang 的原生 AWQ 全模型加载和生成 smoke 在 `runs/20261101-063118-native-awq-65c3132f/` 完成。
- mini-sglang 使用同一 checkpoint 完整加载、启动 API，并由 Harness 完成 2/2 请求，证据在 `runs/20261101-063001-ebfb4f7d/`。该配置为 `configs/experiments/awq-qwen2.5-0.5b-smoke.json`，`artifact.runtime_compatible=true`。输出质量和性能仍未作结论。
- `harness-002` 的功能验收现已标记为 `passing`：基线检查、AWQ 配置 `validate` 和 `plan` 均再次通过；三份 AWQ 运行证据分别覆盖原生 checkpoint、adapter 逐层对齐和 mini-sglang API smoke。该状态不覆盖质量、精度或性能验收。

## Session Log

### Session 001 — Bootstrap

- 建立 `AGENTS.md`、状态文件、`harness/`、配置、smoke 数据和文档。
- 下一步：实现 AWQ W4A16 产物到 mini-sglang 的 loader/linear adapter，再接入统一跑分。
- 下一步：若用户要求质量验证，优先加入同 tokenization 的续写/PPL evaluator，并在相同模型/采样下比较 BF16、原生 AWQ 与 mini-sglang AWQ；不要将当前 API 的 `completion_tokens=0` 用作吞吐结论。
- 环境证据：`docs/ENVIRONMENTS.md`。
- 将默认 baseline 切换为双 RTX 5090 张量并行的 Qwen3-8B BF16，并增加可复现的 WikiText-2 prompt 生成脚本与模型下载脚本。

### Session 002 — mini-sglang 单卡冒烟

- 将 Harness 的 runtime 改为 `mini-sglang`，并将 baseline/smoke 实验配置迁移到该运行时。
- 将官方 mini-sglang 源码纳入仓库普通目录（已移除其 `.git`），建立独立 `mini-sglang` Conda 环境。
- 完成 CLI、PyTorch GPU、真实 Qwen3-8B 权重加载和 OpenAI `/v1/models` API 验证。
- 真实两条 WikiText-2 生成尚未完成：需保留 FlashInfer JIT 日志并单独定位 scheduler 子进程退出原因后再重跑。

### Session 003 — AWQ 专用 adapter 与功能链路验收

- 对 ModelScope `Qwen/Qwen2.5-0.5B-Instruct` 完成默认 AWQ search 与 W4G128 packing；Llama2-7B 下载接口返回 404 后按既定回退策略使用该公开小模型。
- 实现并验证专用 `llm-awq v2 .pt → mini-sglang` adapter：合并 QKV/Gate-Up（包括 bias），按真实 AWQ scale/zero padding 规则分配元数据，并使用原生 AWQ GEMV/GEMM kernel。
- 原生 AWQ 全模型、逐层 adapter 对齐以及 Harness API smoke 均已留存独立运行证据。下一项未完成 feature 为质量/性能对比；应先加入固定 tokenization 的 continuation/PPL evaluator，禁止从当前 API 的零 completion token 推导吞吐。
