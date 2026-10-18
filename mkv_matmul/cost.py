from .types import Cost, HardwareSpec, MatMulProblem, Schedule


def ceil_div(value: int, divisor: int) -> int:
    """ 非负整数的向上取整除法，要求 divisor > 0 """
    return (value + divisor - 1) // divisor

def dma_transfer_cycles(
    byte_count: int,
    hw: HardwareSpec,
) -> int:
    """ 估计一次非空 DMA 传输的周期，包含一次启动开销 """
    dma_cycles = hw.dma_setup_cycles + ceil_div(byte_count, hw.dma_bytes_per_cycle)
    return dma_cycles
  
def estimate_cost(
    problem: MatMulProblem,
    hw: HardwareSpec,
    schedule: Schedule,
) -> Cost:
    """ 估计合法 MNK 方案的成本；计算与 DMA 不重叠 """
    # 总周期 = 计算周期 + DMA周期
    
    compute_cycles = 0
    dma_cycles = 0

    input_bytes = 0
    weight_bytes = 0
    output_bytes = 0

    for m0 in range(0, problem.M, schedule.tile_m): 
        mt = min(schedule.tile_m, problem.M - m0) 

        for n0 in range(0, problem.N, schedule.tile_n):
            nt = min(schedule.tile_n, problem.N - n0)

            for k0 in range(0, problem.K, schedule.tile_k):
                kt = min(schedule.tile_k, problem.K - k0)

                block_input_bytes = mt * kt * hw.input_element_bytes
                block_weight_bytes = kt * nt * hw.weight_element_bytes
                block_compute_cycles = (ceil_div(mt, hw.array_rows) * ceil_div(nt, hw.array_cols)) \
                                      * (kt + hw.pipeline_cycles)

                input_bytes += block_input_bytes
                weight_bytes += block_weight_bytes
                compute_cycles += block_compute_cycles

                dma_cycles += dma_transfer_cycles(block_input_bytes, hw)
                dma_cycles += dma_transfer_cycles(block_weight_bytes, hw)

            # 当前输出块完成全部 K 块的累加后，写回一次
            block_output_bytes = (mt * nt * hw.accumulator_element_bytes)
            output_bytes += block_output_bytes
            dma_cycles += dma_transfer_cycles(block_output_bytes, hw)
            
    effective_macs = problem.M * problem.N * problem.K

    pe_utilization = effective_macs / (
        hw.array_rows * hw.array_cols * compute_cycles
    )

    return Cost(
        total_cycles=compute_cycles + dma_cycles,
        compute_cycles=compute_cycles,
        dma_cycles=dma_cycles,
        input_bytes=input_bytes,
        weight_bytes=weight_bytes,
        output_bytes=output_bytes,
        pe_utilization=pe_utilization,
    )