# mkv-matmul

面向 FPGA GEMM 的最小调度编译器实验。支持 INT8 输入、INT32 累加与输出、MNK 分块、尾块和带填充的行步长。

## 流程

YAML 配置 → 候选枚举 → 合法性检查 → 成本估计与选择 → 64 字节描述符 → 软件执行验证。

| 模块 | 职责 |
| --- | --- |
| `config`、`types` | 配置检查与内部表示 |
| `candidates`、`legality` | 枚举 tile，检查容量、布局和字段位宽 |
| `cost`、`search` | 估计计算与 DMA 周期，选择合法方案 |
| `descriptor`、`descriptor_layout` | 小端描述符编码与解码 |
| `memory`、`emulator`、`reference`、`verification` | 按描述符执行，与独立参考比较 |
| `report` | 导出 CSV、JSON |

## 运行

在仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python main.py
```

`main.py` 使用示例问题，并将权重 buffer 改为 1024 字节。产物位于 `build/small_weight_buffer/`：

- `candidates.csv`：全部候选、失败原因及预测成本。
- `best_schedule.json`：最佳方案与配置快照。
- `matmul.desc`：64 字节二进制执行参数。
- `verification.json`：数值误差与输出填充检查结果。

配置检查和原始候选枚举也可单独运行：

```bash
python -m mkv_matmul check-config --problem examples/q_proj_prefill.yaml --hardware examples/zcu104_16x16.yaml --output build/config_check
python -m mkv_matmul list-candidates --hardware examples/zcu104_16x16.yaml --output build/raw
```

## 实验与边界

示例形状为 `M=16, N=2048, K=2048`，共 48 个候选。权重 buffer 为 1024 字节时，4 个合法，最佳 tile 为 `(16,16,32)`，预测 1,321,472 周期；32,768 个输出元素与参考结果完全一致。2026-10-18 核验：132 项测试通过。

成本模型 `serial_v1` 将计算与 DMA 周期相加，未模拟重叠；预测周期不是 FPGA 实测性能。当前验证使用随机 INT8 数据和软件执行器，尚未对接 RTL。

待完成：统一 `compile` 命令、自动生成 `report.md`、源码精简至 1000 行以内。
