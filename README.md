# mkv-matmul

面向 FPGA GEMM 的最小调度编译器实验，用于学习从问题配置到硬件执行参数生成、再到软件验证的完整流程。

支持 INT8 输入和权重、INT32 累加与输出、MNK 分块、尾块处理，以及带填充的行步长。

## 编译流程

```text
YAML 配置
→ 候选枚举
→ 合法性检查
→ 成本估计
→ 选择最佳方案
→ 生成二进制描述符
→ 软件执行验证
→ 输出报告
```

## 环境安装

需要 Python 3.10 或更新版本。在仓库根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

依赖包括 NumPy、PyYAML 和 pytest。

## 快速开始

```bash
python -m mkv_matmul compile \
  --problem examples/q_proj_prefill.yaml \
  --hardware examples/zcu104_16x16.yaml \
  --output build/q_proj
```

编译器读取指定配置，枚举并评估候选方案，生成描述符，再按照描述符访问模拟内存并执行矩阵乘法。

输入数据的随机种子来自问题 YAML 中的 `seed` 字段。

成功完成后，输出目录包含：

| 文件 | 内容 |
| --- | --- |
| `candidates.csv` | 全部候选的编号、分块参数、合法性、失败原因、buffer 占用与预测成本 |
| `best_schedule.json` | 最佳方案、问题与硬件配置快照、预测成本 |
| `matmul.desc` | 64 字节、小端序的二进制执行描述符 |
| `verification.json` | 随机种子、输出比较结果、最大绝对误差与填充区检查结果 |
| `report.md` | 人可读的编译与验证摘要 |

成功读取配置后，重复编译会先清理输出目录中这五个同名旧产物。建议使用专门的编译输出目录。

## 配置约定

示例问题为：

```text
M = 16
N = 2048
K = 2048
```

矩阵形状为：

```text
A: M × K
B: K × N
C: M × N
```

主要单位约定：

- 基地址以字节计。
- 行步长以元素计。
- Buffer 容量和搬运量以字节计。
- 预测时间以周期计。
- `seed` 为非负整数。

输入、权重、输出的行步长分别至少为 `K`、`N`、`N`。填充空间不属于逻辑矩阵，不参与计算。

示例硬件配置包含：

```text
阵列：16 × 16
输入 buffer：32768 字节
权重 buffer：65536 字节
输出 buffer：32768 字节
输入与权重双缓冲：开启
```

候选集合为：

```text
tile_m: [1, 4, 8, 16]
tile_n: [16, 32, 64]
tile_k: [32, 64, 128, 256]
```

共生成 48 个候选。非法候选仍保留在候选表中，记录失败原因，成本字段留空。

## 方案选择与成本模型

合法候选按以下顺序比较：

1. 预测总周期更少；
2. 总周期相同时，PE 利用率更高；
3. 再相同时，buffer 总占用更小；
4. 最后按原始候选编号选择。

当前成本模型为 `serial_v1`：

```text
总周期 = 计算周期 + DMA 周期
```

计算与 DMA 不重叠。双缓冲计入片上容量需求，但当前模型不估计重叠带来的时间收益。

成本计算按尾块的实际尺寸估计计算量与搬运量；buffer 容量检查按完整 tile 计算。

每个 M、N、K 块组合读取对应的 A、B 块；一个输出块完成全部 K 块累加后写回一次。当前模型未考虑跨块驻留复用。

PE 利用率定义为：

```text
有效 MAC 数 /（阵列 PE 数 × 预测计算周期）
```

计算周期包含模型中的流水线开销，利用率分母不包含 DMA 周期。

## 描述符与软件验证

描述符采用固定的 64 字节小端布局，包含：

- 协议标识、版本、操作码与执行选项；
- M、N、K 和分块尺寸；
- 输入、权重、输出基地址；
- 以元素计的行步长。

验证流程按原始问题布局装载输入，使用描述符解码出的参数执行，再按原始问题约定读取输出。

执行器通过模拟内存处理地址、步长、尾块与部分和累加。参考结果由独立的 NumPy 整体矩阵乘法使用 INT64 运算得到。

验证通过要求：

- 所有输出元素与参考结果完全相等；
- 输出行间填充内容保持不变。

## 失败处理与退出码

| 情况 | 退出码 | 产物行为 |
| --- | ---: | --- |
| 编译与软件验证通过 | 0 | 生成五个产物 |
| 全部候选非法 | 1 | 保留 `candidates.csv`，不生成描述符 |
| 执行完成但结果验证失败 | 1 | 保留五个产物，报告记录失败 |
| 配置错误或执行异常 | 1 | 输出错误信息，产物取决于失败阶段 |
| 命令行参数错误 | 2 | 由 argparse 输出使用说明 |

在 shell 中可以用 `echo $?` 查看上一条命令的退出码。

## 辅助命令

仅检查配置并生成摘要：

```bash
python -m mkv_matmul check-config \
  --problem examples/q_proj_prefill.yaml \
  --hardware examples/zcu104_16x16.yaml \
  --output build/config_check
```

仅枚举原始分块候选：

```bash
python -m mkv_matmul list-candidates \
  --hardware examples/zcu104_16x16.yaml \
  --output build/raw
```

查看命令帮助：

```bash
python -m mkv_matmul --help
python -m mkv_matmul compile --help
```

根目录的 `main.py` 保留为分步实验脚本。它将权重 buffer 覆盖为 1024 字节，结果写入 `build/small_weight_buffer/`。常规使用请运行统一的 `compile` 命令。

## 测试

```bash
python -m pytest -q
```

最近一次测试结果为 **136 passed**。

测试覆盖配置检查、候选枚举、容量与布局检查、成本估计、方案选择、描述符固定字节与往返检查、模拟内存、尾块与部分和、验证报告，以及命令行成功和失败路径。

## 模块结构

| 模块 | 职责 |
| --- | --- |
| `config`、`types` | 配置检查与内部数据结构 |
| `candidates`、`legality` | 候选枚举与合法性检查 |
| `cost`、`search` | 成本估计与最佳方案选择 |
| `descriptor_layout`、`descriptor` | 描述符协议定义、编码与解码 |
| `memory`、`emulator` | 模拟内存与描述符驱动执行 |
| `reference`、`verification` | 独立参考计算与结果验证 |
| `report` | CSV、JSON 和 Markdown 产物输出 |
| `__main__` | 命令行入口与流程组织 |

## 当前边界

本项目用于学习最小调度编译流程。预测周期不是 FPGA 实测性能，软件执行器也不是周期精确模拟器。

当前尚未对接真实 RTL 或 FPGA 板卡，尚未建模计算与 DMA 重叠、DMA 总线细节及完整的硬件资源约束。一次随机输入验证通过，也不代表覆盖了所有可能输入。