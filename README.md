# mkv-matmul
A minimal FPGA-oriented matrix multiplication scheduling compiler.
面向 FPGA 通用矩阵乘法的最小调度编译器实验.

## 项目目标

问题描述 -> 候选生成 -> 合法性检查 -> 成本估计 -> 选择最佳方案 -> 生成 RTL 描述符 -> 软件执行验证

## 当前进度

- [x] 初始化仓库
- [x] 配置读取与数据结构
- [ ] 候选生成与合法性检查
- [ ] 成本模型与搜索
- [ ] 分块计算验证
- [ ] 二进制描述符生成

## 当前功能

- 读取并验证问题配置与硬件配置
- 构造内部数据对象
- 输出配置摘要
  
## 安装依赖

```bash
python -m pip install -r requirements.txt
```

## 检查配置

```bash
python -m mkv_matmul check-config \
  --problem examples/q_proj_prefill.yaml \
  --hardware examples/zcu104_16x16.yaml \
  --output build/config_check
```

## 运行配置测试

```bash
python -m pytest tests/test_config.py -q
```
