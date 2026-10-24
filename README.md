# Inference & Quantization

推理（Inference）与模型量化（Quantization）学习记录。

本仓库主要用于整理学习笔记、实验结论、参考资料和少量可复现实验代码。内容会随着学习过程持续补充。

## 内容方向

- 深度学习模型推理基础
- 权重、激活值与 KV Cache 量化
- PTQ（训练后量化）与 QAT（量化感知训练）
- GPTQ、AWQ、SmoothQuant 等方法
- INT8、INT4、FP8 等低精度格式
- Transformers、vLLM、TensorRT-LLM 等推理生态
- 显存、吞吐、延迟与精度之间的权衡

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
