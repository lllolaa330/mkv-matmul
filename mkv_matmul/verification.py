import numpy as np

from .emulator import execute_descriptor
from .memory import Memory, read_matrix, write_matrix
from .reference import generate_inputs, reference_matmul
from .types import HardwareSpec, MatMulProblem


def verify_descriptor(
    problem: MatMulProblem,
    hw: HardwareSpec,
    descriptor: bytes,
    seed: int,
) -> dict:
    """装载原始问题的数据，执行描述符并比较输出。"""
    A, B = generate_inputs(problem, seed)
    expected = reference_matmul(A, B)

    memory = Memory()

    regions = (
        (
            problem.input_base,
            problem.M * problem.input_stride
            * hw.input_element_bytes,
        ),
        (
            problem.weight_base,
            problem.K * problem.weight_stride
            * hw.weight_element_bytes,
        ),
        (
            problem.output_base,
            problem.M * problem.output_stride
            * hw.accumulator_element_bytes,
        ),
    )

    for base, size in regions:
        memory.map_region(base, size)
        memory.write(base, b"\xa5" * size)

    write_matrix(
        memory,
        problem.input_base,
        problem.input_stride,
        A,
    )
    write_matrix(
        memory,
        problem.weight_base,
        problem.weight_stride,
        B,
    )

    execute_descriptor(descriptor, hw, memory)

    actual = read_matrix(
        memory,
        base=problem.output_base,
        rows=problem.M,
        cols=problem.N,
        stride=problem.output_stride,
        dtype="<i4",
    )

    # 先扩宽再相减，避免误差统计本身溢出
    difference = actual.astype(np.int64) - expected

    # TODO 1：统计 difference 中非零元素的个数，转换成 Python int
    mismatch_count = int(np.count_nonzero(difference))

    # TODO 2：求最大绝对误差，转换成 Python int
    max_abs_error = int(np.max(np.abs(difference)))

    padding_unchanged = True
    padding_bytes = (
        problem.output_stride - problem.N
    ) * hw.accumulator_element_bytes

    if padding_bytes > 0:
        for i in range(problem.M):
            address = problem.output_base + (
                i * problem.output_stride + problem.N
            ) * hw.accumulator_element_bytes

            if memory.read(address, padding_bytes) != b"\xa5" * padding_bytes:
                padding_unchanged = False

    # TODO 3：同时检查数值正确、填充区未改变
    passed = (mismatch_count==0) and padding_unchanged

    return {
        "schema_version": 1,
        "problem_name": problem.name,
        "seed": seed,
        "shape": {
            "M": problem.M,
            "N": problem.N,
            "K": problem.K,
        },
        "checked_elements": int(actual.size),
        "mismatch_count": mismatch_count,
        "max_abs_error": max_abs_error,
        "output_padding_unchanged": padding_unchanged,
        "passed": passed,
    }