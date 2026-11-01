# 容器内 AWQ 与 mini-sglang 环境

本项目运行在共享宿主机分配的 `qiyue.cai` 容器中。环境只安装到个人目录，不修改宿主机或共享 Python 环境：

```text
/home/qiyue.cai/.conda/envs/awq     # Python 3.10，AWQ 与 CUDA kernel
/home/qiyue.cai/.conda/envs/mini-sglang  # Python 3.10，mini-sglang 服务与 Harness
```

AWQ 项目固定了 PyTorch 2.3.0，而当前 RTX 5090 属于 Blackwell 架构；因此不能盲目接受 AWQ 的旧版 PyTorch pin。应在 `awq` 环境安装兼容 RTX 5090 的 PyTorch，再用 `pip install --no-deps -e llm-awq`，避免项目元数据把 PyTorch 降回 2.3.0。安装后必须编译 `awq/kernels` 并运行 GPU smoke test。

本次容器中实际验证的版本是：

```text
awq:    Python 3.10.22 / torch 2.7.1+cu128 / GPU RTX 5090 capability 12.0
mini-sglang: Python 3.10.22 / torch 2.7.1+cu128 / GPU RTX 5090 capability 12.0
```

AWQ kernel 使用 `TORCH_CUDA_ARCH_LIST=12.0` 构建。容器提供 CUDA 13.0 `nvcc`，而 AWQ 使用 cu128 PyTorch，因此构建脚本临时跳过 PyTorch 的 minor-version 检查；kernel 已通过导入和 CUDA 矩阵乘 smoke，但正式量化仍应先用小模型验证数值正确性。可重复执行：

```bash
scripts/build_awq_kernel.sh
PYTHONPATH="$PWD/llm-awq/awq/kernels" \
  /home/qiyue.cai/.conda/envs/awq/bin/python scripts/gpu_smoke.py
```

mini-sglang 验证：

```bash
/home/qiyue.cai/.conda/envs/mini-sglang/bin/python scripts/gpu_smoke.py
/home/qiyue.cai/.conda/envs/mini-sglang/bin/python -m minisgl --help
```

mini-sglang 已完成 Qwen3-8B BF16 的真实模型加载和 100 请求 GPU 0 稳定性验证。该验证使用关闭 FlashInfer/FlashAttention 的普通 PyTorch CUDA fallback；它不是性能基线。

AWQ 与 mini-sglang 不共用 Python 环境，但共享 GPU、模型缓存和实验产物目录。AWQ 只负责生成量化产物；只有在 `artifact.runtime_compatible=true` 且完成 mini-sglang 加载验证后，产物才可以进入 benchmark。目前 AWQ 的 loader/kernel 适配尚未实现。
