# Inference & Quantization

推理（Inference）与模型量化（Quantization）学习记录。

本仓库同时提供一个以 mini-sglang 为统一推理运行时的实验 harness：用同一套配置运行 baseline 和不同量化模型，记录生成结果、延迟、吞吐及后续质量评测证据。当前已跑通 Qwen3-8B BF16 的普通 PyTorch CUDA 路径，以及 `llm-awq v2` W4A16 权重经专用 adapter 由 mini-sglang 加载并完成 API smoke 的链路。后者只代表结构和功能兼容，尚不代表输出质量或性能结论。内容会随着学习过程持续补充。

## 快速开始

```bash
./init.sh
python -m harness.cli validate --config configs/experiments/baseline.json
python -m harness.cli plan --config configs/experiments/baseline.json
```

`plan` 是无 GPU 的预演；真实服务启动、实验运行和结果目录说明见 [docs/HARNESS.md](docs/HARNESS.md)，配置字段见 [docs/EXPERIMENT_SCHEMA.md](docs/EXPERIMENT_SCHEMA.md)。

## 内容方向

- 深度学习模型推理基础
- 权重、激活值与 KV Cache 量化
- PTQ（训练后量化）与 QAT（量化感知训练）
- GPTQ、AWQ、SmoothQuant 等方法
- INT8、INT4、FP8 等低精度格式
- Transformers、vLLM、TensorRT-LLM 等推理生态
- 显存、吞吐、延迟与精度之间的权衡
- mini-sglang 推理服务与可复现实验编排

## 目录约定

随着内容增加，预计按以下方向组织：

```text
.
├── notes/        # 学习笔记
├── experiments/  # 实验记录与少量代码
├── references/   # 论文、文档与资料索引
└── README.md
```

## 学习原则

记录关键概念的同时，尽量保留实验环境、参数、指标和结论，方便之后复查与复现。

## 说明

这是个人学习仓库，笔记内容可能存在不完整或需要修正的地方，欢迎交流指正。
