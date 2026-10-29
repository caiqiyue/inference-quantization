# NVFP4 基础与计算公式学习笔记

> 学习前置：IEEE 754 浮点数。目标：理解 NVFP4 的数值表示、两级缩放、量化与反量化计算，以及它与硬件推理的关系。本文采用**反量化方向（decode direction）**定义 Scale，即重建时乘 Scale。不同框架可能存储倒数（encode scale）。

## 1. 一句话理解 NVFP4

NVFP4 不是“每个权重直接用 4 bit 存储”这么简单，而是 **E2M1 FP4 数值 + 每 16 个元素共享的 FP8 E4M3 Block Scale + Tensor 级 FP32 Global Scale** 的分层缩放表示方案。

核心重建关系：

$$
\boxed{\hat x_i=q_i\,s_{b(i)}\,s_g}
$$

| 符号 | 意义 | 类型/粒度 |
|---|---|---|
| $x_i$ | 原始权重或激活值 | FP16/BF16/FP32 等 |
| $q_i$ | 量化后的数值 | FP4 E2M1，每个元素 4 bit |
| $s_{b(i)}$ | 第 $i$ 个元素所在 Block 的局部 Scale | FP8 E4M3，通常每 16 个元素一个 |
| $s_g$ | 整个 Tensor 共用的 Global Scale | FP32，每个 Tensor 一个 |
| $\hat x_i$ | 反量化重建值 | $q_i s_{b(i)}s_g$ |

> 注：张量分组轴、Scale 排列与具体算子实现有关。本文的基础推演采用一维 16 元素 Block。NVFP4 格式与实际硬件/kernel 所支持的布局要分开辨别。

## 2. E2M1：为什么 FP4 最大数值是 6？

E2M1 = 1 个符号位（S）+ 2 个指数位（E）+ 1 个尾数位（M），总计 $1+2+1=4$ bit，共 16 个 bit pattern（含正负零）。与 IEEE 754 类似，它也有正常数和次正规数，但**E2M1 的全 1 指数编码用于有限值，而不是照搬 IEEE 754 的 Inf/NaN 约定**。

非负数编码表（负数改变符号位）：

| S E E M | 数值 | 说明 |
|---|---:|---|
| 0000 | +0 | 零 |
| 0001 | 0.5 | 次正规数 |
| 0010 | 1 | 正常数 |
| 0011 | 1.5 | 正常数 |
| 0100 | 2 | 正常数 |
| 0101 | 3 | 正常数 |
| 0110 | 4 | 正常数 |
| 0111 | 6 | 正常数 |

可表示的**不同实数值**集合为：

$$
\mathcal F_4=\{0,\pm0.5,\pm1,\pm1.5,\pm2,\pm3,\pm4,\pm6\}.
$$

这里不同实数值为 15 种；正负零有不同的 bit pattern，因此总共有 16 种编码。最大绝对值 $M_4=6$。FP4 的量化函数 $Q_{E2M1}(z)$ 就是把 $z$ 映射到最接近的合法 FP4 数值（还需规定舍入、截断/饱和方式）。

## 3. 名词解释：Per-Tensor、Per-Channel、Per-Block

这些 **Per-XXX 是“多少元素共享一个 Scale”的量化粒度**，并非另一种浮点格式。

- **Per-Tensor**：一个 Tensor 共享一个 Scale。存储少，但小值可能受全局异常大值影响。
- **Per-Channel**：每个通道各自使用 Scale。对于权重矩阵，具体按行还是按列由权重布局和算子定义决定。
- **Per-Group / Per-Block**：每小组数据一个 Scale。组越小通常越能适应局部数据范围，但需要更多 Scale 元数据；并不保证任意误差指标都单调改善。
- **Block Size**：每组元素数。本文的一维 NVFP4 采用 16 个元素。
- **Amax**：Absolute Maximum，绝对值最大值；$\max_i |x_i|$。不要与普通最大值 $\max_i x_i$ 混淆。
- **Global Scale**：Tensor 级缩放基准；在 NVFP4 中与局部 Block Scale 一起作用。
- **Block Scale**：每 16 个 FP4 值共享的局部缩放系数，存为 FP8 E4M3。
- **Effective Scale**：某元素实际使用的总缩放，$s_{\mathrm{eff},b}=s_gs_b$。
- **Quantization**：原值映射到有限数值集合，$q=Q(x/s)$。
- **Dequantization**：用保存的量化值和 Scale 重建近似原值，$\hat x=qs$。
- **Clipping / Saturation**：数值超出格式可表示范围时裁剪到边界；它与就近舍入不同。

NVFP4 可以同时有 Per-Tensor **Global Scale** 和 Per-Block **Block Scale**；这两个概念不是二选一。

## 4. 为什么需要两个 Scale？

普通 FP4 若原始数据最大值为 12，而 FP4 最大值为 6，单个反量化 Scale 可以取 $s=12/6=2$，从而量化 $q=Q_{E2M1}(x/2)$。但整个 Tensor 共享一个 Scale，局部较小的数容易损失精度。

NVFP4 将有效 Scale 分解为：

$$
\boxed{s_{\mathrm{eff},b}=s_g\cdot s_b}
$$

让全局级别确定范围，局部 Block 根据各自的最大值调整。Block Scale 要存储为 FP8 E4M3，最大有限正数 $M_8=448$；FP4 E2M1 最大正数 $M_4=6$。这两个限制解释了 Global Scale 公式中的 $6\times448$。

## 5. 核心公式：Global Scale 怎样算，为什么除以 2688？

给定原始 Tensor $X=\{x_1,\ldots,x_N\}$：

**步骤 1：取整个 Tensor 的最大绝对值**

$$
\boxed{A_g=\max_i|x_i|}
$$

**步骤 2：确定 Global Decode Scale**

单独 FP4 的最大数值是 6；局部 FP8 Scale 最大值是 448。因此，在 $q=6$、$s_b=448$ 时希望重建值正好覆盖整个 Tensor 的 $A_g$：

$$
6\times448\times s_g=A_g
\quad\Longrightarrow\quad
\boxed{s_g=\frac{A_g}{6\times448}=\frac{A_g}{2688}}.
$$

例如 $A_g=12$，则 $s_g=12/2688=1/224$。**Global Scale 是把 FP4 与 FP8 Scale 的组合可表示范围映射回原始 Tensor 的范围，不是直接拿它单独缩放所有数据。**

有些实现保存量化方向的倒数：

$$
\boxed{s_g^{\mathrm{enc}}=\frac{2688}{A_g}=\frac1{s_g^{\mathrm{dec}}}}.
$$

在代码里看到 `global_scale`、`global_scale_inv`、`per_tensor_scale`、`decode_scale` 等变量时，应检查乘法/除法的位置，不要只凭名字判断。

## 6. 核心公式：Block Scale 怎么算？

设第 $b$ 个 Block 为 $B_b$：

$$
\boxed{A_b=\max_{i\in B_b}|x_i|}
$$

我们希望这个 Block 的最大值尽可能对应 FP4 最大值 6：

$$
6\,s_b^{\mathrm{ideal}}\,s_g=A_b
\quad\Longrightarrow\quad
\boxed{s_b^{\mathrm{ideal}}=\frac{A_b}{6s_g}}.
$$

因为 Block Scale 只能按 FP8 E4M3 存储，还需要执行一次 FP8 舍入：

$$
\boxed{s_b=Q_{E4M3}\!\left(\frac{A_b}{6s_g}\right)}.
$$

把 $s_g=A_g/(6\times448)$ 代入，得到：

$$
\boxed{s_b^{\mathrm{ideal}}=448\frac{A_b}{A_g}}.
$$

因为 $A_b\le A_g$，所以理想局部 Scale 不会大于 448；这正是两级缩放的巧妙之处。极小值可能遭遇 FP8 下溢，零 Block 和全零 Tensor 都需专门处理。

## 7. 完整算法：从原始数据到 FP4，再反量化

在采用反量化方向 Scale 的约定下：

1. $A_g=\max_i|x_i|$。
2. $s_g=A_g/2688$（全零 Tensor 单独处理）。
3. 每 16 个元素组成一个 Block，计算 $A_b=\max_{i\in B_b}|x_i|$。
4. $s_b=Q_{E4M3}(A_b/(6s_g))$，需要考虑数值边界。
5. $s_{\mathrm{eff},b}=s_bs_g$。
6. 对每个元素：

   $$
   \boxed{q_i=Q_{E2M1}\!\left(\frac{x_i}{s_bs_g}\right)}
   $$

7. 重建时：

   $$
   \boxed{\hat{x}_i=q_i\,s_b\,s_g}.
   $$

量化误差：

$$
e_i=\hat x_i-x_i,
\qquad
|e_i|=|\hat x_i-x_i|.
$$

误差既可能来自元素映射为 E2M1，也可能来自理想 Block Scale 舍入为 E4M3。Scale 的存在不会增加 FP4 本身可选的数值个数。

## 8. 手算：两个 16 元素 Block

为了看清数学关系，考虑两个 Block 的最大绝对值分别为 6 和 12：

$$
A_1=6,\quad A_2=12,\quad A_g=12.
$$

全局：

$$
s_g=12/2688=1/224.
$$

局部（这两个例子的 Scale 可被 E4M3 精确表示）：

$$
s_1=6/(6/224)=224,
\qquad
s_2=12/(6/224)=448.
$$

有效 Scale：

$$
s_{\mathrm{eff},1}=224/224=1,
\qquad
s_{\mathrm{eff},2}=448/224=2.
$$

| 所属 Block | 原值 $x$ | 有效 Scale | 缩放值 $x/s$ | FP4 值 $q$ | 重建值 $qs$ |
|---|---:|---:|---:|---:|---:|
| 1 | 0.5 | 1 | 0.5 | 0.5 | 0.5 |
| 1 | 3 | 1 | 3 | 3 | 3 |
| 1 | 6 | 1 | 6 | 6 | 6 |
| 2 | 2 | 2 | 1 | 1 | 2 |
| 2 | 6 | 2 | 3 | 3 | 6 |
| 2 | 8 | 2 | 4 | 4 | 8 |
| 2 | 12 | 2 | 6 | 6 | 12 |

出现误差的例子：在有效 Scale = 2 的 Block 中，$x=7.2$，则 $7.2/2=3.6$，就近映射至 FP4 的 4，重建为 $4\times2=8$，绝对误差为 $0.8$，相对误差约 11.11%。

## 9. 为什么不使用“最小值除以最小值”来计算 Scale？

**Scale 本质上在选择有限量化刻度覆盖哪个原始数值范围。** 最大值法优先确保正常数据范围可被覆盖；最小正值法虽然可能照顾小值，却容易导致大值溢出并被裁剪。

例子：原始值 $[0.5,1,2,3,6,12]$，FP4 最大正数为 6，最小正非零数为 0.5。

- 最大值法：$s=12/6=2$。12 映射到 FP4 的 6，可正常重建为 12；0.5 缩放后是 0.25，发生舍入损失。
- 最小正值法：$s=0.5/0.5=1$。0.5 可精确表示，但 12 超过 FP4 的 6，会饱和至 6，重建值只有 6，误差达 50%。

此外，若直接拿整个 Tensor 的最小绝对值，数据中出现零会导致 Scale 为零。对于含正负值的数据，更不能把代数最小值与最小绝对值混为一谈。

这并不意味着 Amax 一定是所有场景的最优 Scale：带异常值的数据可能受益于 clipping、校准、误差最优化等方法，但那是在主动权衡裁剪误差与舍入误差，不是简单“最小值除以最小值”。

## 10. 存储成本：为什么局部 Scale 用 FP8？

16 个 FP4 元素：$16\times4=64$ bit = 8 Bytes。

一个 FP8 Scale：1 Byte。

因此每个 16 元素 Block：$8+1=9$ Bytes，即平均 $9\times8/16=4.5$ bit/参数（**未计 Global Scale、padding、对齐等**）。若局部 Scale 改用 FP32，每 Block 为 $8+4=12$ Bytes，即平均 6 bit/参数。

之前调试 llama.cpp 时观测的 $64$ 个权重 + $4$ 个局部 Scale，可对应：$64\times4/8 +4=32+4=36$ Bytes。这个数字说明基础数据和局部 Scale 的开销，**不能仅凭它确认某个 GGUF 文件如何编码 Global Scale**，仍需查看对应实现源码。

## 11. 为什么推理芯片可能“只原生支持 NVFP4”？

芯片原生支持的不只是“4 bit 的数据宽度”，还有 E2M1 编码解释、FP8 Scale 的读取及配对、Block 分组、打包布局、矩阵乘法输入接口、累加路径和编译器/kernel 支持。INT4、MXFP4 等即使也有 4-bit 元素，其数值含义和 Scale 约定也可能不同，因此不能直接把相同 bit 串交给 NVFP4 的计算通路。

要和 Leader 确认“只支持 NVFP4”具体是：

1. 矩阵计算单元仅有 NVFP4 原生指令；还是
2. 整条模型量化工具、Runtime、Kernel 和推理部署链路只实现 NVFP4。

两者差别很大，不能从“芯片只支持 NVFP4”一句话判断其他通用运算能力是否存在。

**低精度存储和输入，不等于全过程以 FP4 累加。** GEMM 可以读取低精度数据，并在更高精度的累加器中执行乘加。真正的解码、缩放与乘法可能在硬件中融合，并不总是先完整生成一个 FP16/FP32 反量化张量。

## 12. 代码阅读时的四个核对点

```python
amax_global = max(abs(x))
global_decode_scale = amax_global / (6 * 448)

for block in blocks_of_16(x):
    amax_block = max(abs(block))
    ideal_block_scale = amax_block / (6 * global_decode_scale)
    block_scale = round_to_e4m3(ideal_block_scale)
    effective_scale = global_decode_scale * block_scale
    q = round_to_e2m1(block / effective_scale)
    # 保存 q 和 block_scale；global_decode_scale 每 Tensor 保存一次
```

这是**便于理解的伪代码**，省略了零值保护、极小 Scale、NaN/Inf 策略、裁剪、打包布局、实际 kernel 及具体舍入约定。若代码使用 encode scale，量化阶段对应乘以 decode scale 的倒数。

## 13. 复习卡片

- E2M1 数值最大绝对值为什么是 6？因为 4-bit 编码表定义如此，不是 4-bit 无符号整数最大值 15。
- 2688 从哪里来？$6\times448$，即 FP4 最大值与 FP8 局部 Scale 最大值的乘积。
- Global Scale 算法的目标是什么？给整个 Tensor 一个共同范围基准，让局部 Scale 在 FP8 有限表示范围内发挥作用。
- 为什么还要 Block Scale？不同局部数据范围不同，统一有效 Scale 会损失小值分辨率。
- 为什么理想 Scale 算完还要量化？因为实际 Block Scale 本身只能存 FP8。
- 两种 `global_scale` 公式为什么相反？一个用于 encode（乘倒数），一个用于 decode（乘原 Scale）。
- 为什么不用最小正值定 Scale？它可能使全局或局部的大值严重饱和截断。
- 4.5 bit/参数是不是 GGUF 文件的精确平均值？不是；它是 16 个 FP4 + 一个 FP8 Scale 的基础开销，其他元数据另计。

## 14. 后续学习路线

1. 亲手把 E2M1 的 16 个 bit pattern 全部解码，并练习就近舍入与饱和。
2. 用一个 16 元素真实 Block 手算 Amax、理想 E4M3 Scale、Scale 舍入、FP4 映射和重建误差。
3. 理解两个 NVFP4 矩阵 GEMM 时各自 Block Scale 如何参与乘积与高精度累加。
4. 对照 llama.cpp 中具体 GGUF NVFP4 的 block struct、packing 和 GEMM kernel，区分格式规范和项目实现。
5. 再学习 clipping、校准、AWQ、GPTQ 与量化误差优化。

## 参考资料

- NVIDIA Transformer Engine：NVFP4 low precision training 文档：https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html
- NVIDIA Transformer Engine：C API / recipe 文档：https://docs.nvidia.com/deeplearning/transformer-engine/api/c/recipe.html

