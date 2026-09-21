from dataclasses import dataclass

@dataclass(frozen = True)
class MatMulProblem:
    """ 矩阵乘法问题:基地址以字节计,行步长以元素计 """
    name : str
    M    : int
    N    : int
    K    : int
    input_base  : int
    weight_base : int
    output_base : int
    input_stride  : int
    weight_stride : int
    output_stride : int
    
@dataclass(frozen = True)
class HardwareSpec:
    """ 硬件能力: 容量以字节计,时间以周期计 """
    name : str
    
    # 计算阵列
    array_rows      : int
    array_cols      : int
    pipeline_cycles : int
    
    # 数据存储
    input_element_bytes       : int
    weight_element_bytes      : int
    accumulator_element_bytes : int
    input_buffer_bytes        : int
    weight_buffer_bytes       : int
    output_buffer_bytes       : int
    
    # 双缓冲
    double_buffer_input       : bool
    double_buffer_weight      : bool
    
    # DMA
    dma_bytes_per_cycle       : int
    dma_setup_cycles          : int
    alignment_bytes           : int
    
    # 候选集合
    tile_m_candidates         : tuple[int, ...]
    tile_n_candidates         : tuple[int, ...]
    tile_k_candidates         : tuple[int, ...]
    
    
@dataclass(frozen = True)
class Schedule:
    """ 矩阵乘法分块方案 循环顺序从外层到内层 """
    tile_m : int
    tile_n : int
    tile_k : int
    loop_order : str = "MNK" # 块循环 M -> N -> K 由外到内
    
@dataclass(frozen = True)
class Cost:
    """ 一个调度方案的预测成本 时间以周期计 传输量以字节计 """
    total_cycles   : int 
    compute_cycles : int
    dma_cycles     : int
    input_bytes    : int   # 任务的累计传输量 而非 buffer 容量
    weight_bytes   : int   # 小 buffer 可能被反复使用 所以总传输量可能大于 buffer 容量
    output_bytes   : int
    pe_utilization : float # 相较于计算阶段周期的有效 MAC 利用率