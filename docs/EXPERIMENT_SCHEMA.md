# 实验配置与结果协议

配置使用 JSON，便于无额外依赖地校验和复制。必填字段：

| 字段 | 含义 |
| --- | --- |
| `name` | 人类可读的实验名，不能复用同一 run 输出目录 |
| `model` | Hugging Face ID 或本地模型路径 |
| `dataset` | JSONL 路径，每行至少包含字符串字段 `prompt` |
| `sampling` | 传给 `/v1/chat/completions` 的采样参数 |
| `server` | mini-sglang 启动参数；`port` 必填 |
| `quantization` | 方法、权重/激活/KV cache 数据类型及算法参数 |

当前 baseline 还使用 `dataset_metadata` 记录外部 WikiText-2 源文件、生成命令和评测用途；
它不是必填字段，但凡是外部数据集都建议这样保存可追溯信息。

建议额外添加 `tags`、校准集、硬件和 git revision。任何影响结果的参数都应进入配置，而不是藏在 shell 历史里。

## AWQ v2 专用输入

当前已验证的 AWQ 输入格式为 `llm-awq-v2-pt`。实验必须在
`server.extra_args` 中同时声明 `awq_checkpoint` 与 `awq_group_size`，并让
`quantization.group_size` 保持一致。mini-sglang 的 AWQ adapter 只支持单卡 W4A16
和 group size 32/64/128；它在加载时验证 native `qweight`、`scales`、
`scaled_zeros` 形状以及 Q/K/V、Gate/Up 的融合边界。其他 AWQ 版本或其它量化方法
不能复用这条适配路径，必须先补充自己的 loader/kernel 验证。

## 结果目录

```text
runs/<run_id>/
├── manifest.json    # 实验输入快照和运行身份
├── metrics.json     # 可聚合指标
└── responses.jsonl  # 原始 API 响应
```

`run_id` 由 UTC 时间和随机后缀组成，结果目录只创建不覆盖。`artifact.runtime_compatible` 表示产物已经被当前 mini-sglang loader 与 kernel 验证可加载；`false` 的量化产物只能作为待适配输入，不能进入正式跑分。该状态只代表结构和功能兼容，不代表质量分数或性能结论；两者必须由独立 evaluator/benchmark run 记录。未来加入 GPU 采样、服务日志和 evaluator 时，沿用该目录并在 manifest 中记录版本。
