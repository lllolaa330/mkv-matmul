import numpy as np

from .descriptor import decode_descriptor
from .legality import check_legality
from .memory import Memory, read_matrix, write_matrix
from .types import HardwareSpec


def execute_descriptor(
    data: bytes,
    hw: HardwareSpec,
    memory: Memory,
) -> None:
    """ 按描述符执行分块矩阵乘法 将 INT32 结果写回模拟内存 """
    decoded = decode_descriptor(data)
    problem = decoded.problem
    schedule = decoded.schedule

    legal, reason = check_legality(problem, hw, schedule)

    if not legal:
        raise ValueError(f"描述符不能在当前硬件上执行: {reason}")

    expected_flags = (
        int(hw.double_buffer_input)
        | (int(hw.double_buffer_weight) << 1)
    )

    if decoded.flags != expected_flags:
        raise ValueError("描述符双缓冲选项与硬件配置不一致")

    for m0 in range(0, problem.M, schedule.tile_m):
        mt = min(schedule.tile_m, problem.M - m0)

        for n0 in range(0, problem.N, schedule.tile_n):
            nt = min(schedule.tile_n, problem.N - n0)

            # 一个输出块只初始化一次累加器
            accumulator = np.zeros((mt, nt), dtype=np.int32)

            for k0 in range(0, problem.K, schedule.tile_k):
                kt = min(schedule.tile_k, problem.K - k0)

                # TODO 1：A[m0, k0] 的字节地址
                input_address = problem.input_base + \
                    (m0 * problem.input_stride + k0) * hw.input_element_bytes

                # TODO 2：B[k0, n0] 的字节地址
                weight_address = problem.weight_base + \
                    (k0 * problem.weight_stride + n0) * hw.weight_element_bytes

                A_block = read_matrix(
                    memory,
                    base=input_address,
                    rows=mt,
                    cols=kt,
                    stride=problem.input_stride,
                    dtype="i1",
                )

                B_block = read_matrix(
                    memory,
                    base=weight_address,
                    rows=kt,
                    cols=nt,
                    stride=problem.weight_stride,
                    dtype="i1",
                )

                accumulator += (
                    A_block.astype(np.int32)
                    @ B_block.astype(np.int32)
                )

            # TODO 3：C[m0, n0] 的字节地址
            output_address = problem.output_base + \
                    (m0 * problem.output_stride + n0) * hw.accumulator_element_bytes

            write_matrix(
                memory,
                base=output_address,
                stride=problem.output_stride,
                values=accumulator,
            )