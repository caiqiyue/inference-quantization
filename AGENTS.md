# Inference & Quantization Harness

本项目是一个面向大模型推理量化的可复现实验平台：以 mini-sglang 为唯一推理运行时，对不同量化算法、产物格式和运行参数进行对比，并记录质量、延迟、吞吐和显存等证据。Agent MUST 优先保证实验可复现和结果可解释，再追求单次运行的性能。

## Startup Workflow

1. 阅读本文件、`feature_list.json` 和 `claude-progress.md`。
2. 运行 `./init.sh`，确认 harness 基线健康。
3. 选择 `feature_list.json` 中最高优先级且未完成的单个 feature（WIP=1）。
4. 先用 `python -m harness.cli validate --config ...` 校验实验，再用 `plan` 检查命令和参数。
5. 真实 GPU 实验必须保存配置快照、运行环境和原始结果，禁止只记录手工总结。

## Hard Constraints

- mini-sglang 是默认且唯一的推理运行时；新增运行时必须另行记录，不得悄悄改变基线。
- AWQ 只负责产生量化权重；量化产物必须完成 mini-sglang loader/kernel 兼容验证后才可进入跑分。
- 每次对比至少固定模型、数据集、采样参数、并发度、GPU 信息和量化配置。
- 量化算法通过配置和 registry 接入，不能在 runner 中写算法名称分支。
- `plan` 不启动服务、不访问 GPU；`run` 才执行有副作用的实验。
- 实验结果使用唯一 `run_id`，原始响应、指标和 manifest 不覆盖已有运行。
- 缺少依赖、模型或 GPU 时必须明确失败或 `[SKIP]`，禁止伪造通过。
- 不在本仓库提交模型权重、tokenizer 缓存或大体积 benchmark 输出。
- 更改实验协议时同步更新 `docs/`、feature 状态和 progress 日志。

## Repository Map

- `harness/`：配置、mini-sglang 命令生成、HTTP benchmark、结果落盘。
- `configs/experiments/`：可复制的实验配置；配置是实验输入的 source of truth。
- `data/`：小型 smoke 数据；正式数据集使用外部路径。
- `docs/HARNESS.md`：面向使用者的完整操作说明。
- `docs/EXPERIMENT_SCHEMA.md`：配置字段和结果格式。
- `tests/`：不依赖 GPU 的 harness 单元测试。
- `feature_list.json`：任务状态机；完成 feature 必须有 evidence。

## Definition of Done

- 配置可被 `validate` 接受，且 `plan` 输出的命令与参数可审查。
- 代码有对应的无 GPU 测试；涉及 mini-sglang/真实模型时补充实际运行证据。
- 结果包含 `manifest.json`、`metrics.json`，并能从 run 目录追溯配置。
- 更新 `claude-progress.md` 和 `session-handoff.md`，如有必要更新文档。

## Verification

```bash
./init.sh
python -m harness.cli validate --config configs/experiments/baseline.json
python -m harness.cli plan --config configs/experiments/baseline.json
```

真实运行示例见 `docs/HARNESS.md`。默认服务地址为 `http://127.0.0.1:30000`。

## End of Session

更新 feature 状态、验证证据和 progress；检查 `git diff`；确认 `./init.sh` 可运行，并在 handoff 中写下下一个最小步骤。

## Context Pressure Protocol

50% 时收敛范围，70% 时准备交接，85% 时只做验证和记录，95% 时停止扩展并写完整 handoff。
