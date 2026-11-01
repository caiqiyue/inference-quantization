# 推理量化实验 Harness

## 目标

Harness 把一次实验固定成五个输入：模型、量化算法、量化产物格式、数据集和 mini-sglang 服务参数。量化算法可以替换，模型格式也可以多样，但必须先声明并验证 mini-sglang loader/kernel 兼容性，之后才进入统一跑分。Harness 负责生成服务启动命令、向 mini-sglang 的 OpenAI 兼容接口发请求，并把配置快照、原始响应和指标写入独立 run 目录。

当前版本刻意不把“量化算法实现”塞进 runner。量化发生在独立算法环境或转换阶段，mini-sglang 只负责加载和执行。每个实验必须显式记录 `algorithm`、`artifact` 和 `runtime`。目前已实现一条专用链路：`llm-awq v2 .pt → AWQ adapter → mini-sglang`。它不是通用量化格式；GPTQ、FP8 等产物必须各自实现并验证对应适配器，不能把不兼容 checkpoint 当作性能实验输入。

## 冷启动

```bash
./init.sh
python -m harness.cli validate --config configs/experiments/baseline.json
python -m harness.cli plan --config configs/experiments/baseline.json
```

`plan` 只检查配置并打印命令，不需要 GPU，也不会启动进程。

## 当前 Qwen3-8B 基线

默认 baseline 是未量化的官方 `Qwen/Qwen3-8B` BF16 checkpoint，落在仓库外的
`/home/qiyue.cai/host_workspace/models/Qwen3-8B`，避免将权重纳入 Git。首次准备模型：

```bash
scripts/download_qwen3_8b_modelscope.sh
python scripts/prepare_wikitext2.py
```

第二条命令从 `/home/qiyue.cai/host_workspace/calibration-data/wikitext-2-raw/wiki.test.raw`
确定性地产生 32 条 JSONL prompt。它用于串行 API smoke 和延迟基线，**不是** WikiText
困惑度（PPL）评测；若要比较量化精度，后续应加入同 tokenization 的续写/PPL evaluator。

## 已验证的 BF16 真实实验

当前已验证的是普通 PyTorch CUDA fallback：它关闭 FlashInfer/FlashAttention attention 与 CUDA graph，适合功能和稳定性验证，不适合性能结论。只使用 GPU 0 启动服务：

```bash
MINISGL_TORCH_FALLBACK=1 CUDA_VISIBLE_DEVICES=0 \
  /home/qiyue.cai/.conda/envs/mini-sglang/bin/python -m minisgl \
  --model /home/qiyue.cai/host_workspace/models/Qwen3-8B \
  --host 127.0.0.1 --port 30000 --tp-size 1 --dtype bfloat16 \
  --memory-ratio 0.7 --num-pages 1024 --max-seq-len-override 1024 \
  --attn torch --cuda-graph-max-bs 0
```

另一个终端执行：

```bash
python -m harness.cli run \
  --config configs/experiments/qwen3-8b-wikitext2-100-torch.json \
  --server-url http://127.0.0.1:30000
```

输出目录形如 `runs/20261031-120000-a1b2c3d4/`，包含：

- `manifest.json`：完整配置、run_id、数据集和服务地址。
- `responses.jsonl`：每个请求的原始响应。
- `metrics.json`：请求数、平均/P50 延迟、生成 token 数和聚合 token/s。

不要把不同模型、不同 prompt 数量或不同采样参数的结果直接横向比较。吞吐实验还应固定并发度、warmup 次数、GPU 型号和温度；本版本的 runner 是串行 smoke benchmark，不能代替正式并发压测。

## 接入量化算法：当前边界

复制 `configs/experiments/baseline.json`，修改 `name`、`model`、`algorithm`、`artifact`、`server.extra_args` 与 `quantization`。`configs/experiments/awq-qwen2.5-0.5b-smoke.json` 是已验证 AWQ 示例：它使用 `llm-awq` 的原生 v2 `.pt`、W4A16、group 128、单卡 BF16 和 `--awq-group-size 128`。adapter 会在加载前校验 `qweight`、`scales`、`scaled_zeros` 的真实 AWQ packing 形状，并将 Q/K/V、Gate/Up 映射为 mini-sglang 的 fused projection。

在将 AWQ 产物标记为 `runtime_compatible=true` 前，必须完成以下证据链：

1. 原生 AWQ 完成 search 与 real packing；
2. 原生 `WQLinear` 与 mini-sglang adapter 对同一层、同一输入的 GEMV/GEMM 输出对齐；
3. mini-sglang 完整加载并由 Harness 完成 API smoke。

`scripts/validate_awq_minisgl_adapter.py` 记录第 2 步的唯一 run 目录；`scripts/smoke_native_awq.py` 可在不经过 mini-sglang 的情况下直接加载并生成，以排除 runtime 对 AWQ 产物的影响。每种量化方法需要另有转换/校准步骤，并将搜索产物、最终模型格式和转换证据写入配置。

当前这三步已对 `configs/experiments/awq-qwen2.5-0.5b-smoke.json` 验证完成：原生 AWQ 全模型 smoke 位于 `runs/20261101-063118-native-awq-65c3132f/`，逐层 adapter 对齐位于 `runs/20261101-062735-awq-adapter-90196dd8/`（QKV 与 Gate/Up 在 GEMV/GEMM 均为 0 误差），mini-sglang 的 2/2 API smoke 位于 `runs/20261101-063001-ebfb4f7d/`。这些证据只支持“可加载、可执行”的结论；必须另行运行固定 tokenization 的质量评测和独立性能压测，才可对准确率、输出质量或吞吐作结论。

标准实验链路：

```text
量化算法（AWQ/GPTQ/NVFP4/...）
        ↓
量化模型或中间 checkpoint
        ↓ 兼容性检查/格式转换
mini-sglang 已支持的模型格式
        ↓
mini-sglang 服务
        ↓
Harness 性能与质量记录
```

质量评测应使用同一组 prompt 和 reference，在独立 evaluator 中计算准确率、困惑度或任务分数；性能指标和质量指标都要落到同一个 run 的 manifest 中，避免“只报快、不报准”。

## 设计边界

本 harness 当前覆盖：配置校验、mini-sglang 命令计划、OpenAI 兼容请求、基础客户端计时、结果持久化、BF16 torch fallback 下的 100 请求稳定性验证，以及单卡 `llm-awq v2` W4A16 loader/kernel adapter。尚未覆盖：GPTQ/NVFP4 loader 与 kernel、AWQ tensor parallel、自动启动/等待/清理服务、并发压测、GPU 显存采集和准确率 evaluator。这些应作为独立 feature 逐个加入，并各自提供验证证据。
