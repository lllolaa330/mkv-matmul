from pathlib import Path

import numpy as np
import pytest

from mkv_matmul.config import load_hardware
from mkv_matmul.descriptor import encode_descriptor
from mkv_matmul.emulator import execute_descriptor
from mkv_matmul.memory import Memory, read_matrix, write_matrix
from mkv_matmul.reference import generate_inputs, reference_matmul
from mkv_matmul.types import MatMulProblem, Schedule

@pytest.mark.parametrize(
    "M, N, K",
    [
        (4, 16, 32),
        (5, 19, 35),
        (1, 3, 5),
    ],
)
def test_execute_descriptor_matches_reference(M, N, K):
    root = Path(__file__).resolve().parents[1]
    hw = load_hardware(
        str(root / "examples" / "zcu104_16x16.yaml")
    )

    problem = MatMulProblem(
        name="emulator_test",
        M=M,
        N=N,
        K=K,
        input_base=0x10000000,
        weight_base=0x20000000,
        output_base=0x30000000,
        input_stride=K + 3,
        weight_stride=N + 5,
        output_stride=N + 2,
    )
    schedule = Schedule(4, 16, 32)

    A, B = generate_inputs(problem, seed=20260920)
    expected = reference_matmul(A, B)

    memory = Memory()

    regions = (
        (
            problem.input_base,
            M * problem.input_stride,
        ),
        (
            problem.weight_base,
            K * problem.weight_stride,
        ),
        (
            problem.output_base,
            M * problem.output_stride * 4,
        ),
    )

    for base, size in regions:
        memory.map_region(base, size)
        memory.write(base, b"\xa5" * size)

    # 按原始问题布局装载输入
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

    # 执行器只拿到描述符、硬件配置和模拟内存
    descriptor = encode_descriptor(problem, hw, schedule)
    execute_descriptor(descriptor, hw, memory)

    # 按原始问题约定的输出布局读取结果
    actual = read_matrix(
        memory,
        base=problem.output_base,
        rows=M,
        cols=N,
        stride=problem.output_stride,
        dtype="<i4",
    )

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.dtype("<i4")

    # 每行的输出填充空间都必须保持不变
    padding_bytes = (problem.output_stride - N) * 4

    for i in range(M):
        padding_address = (
            problem.output_base
            + (i * problem.output_stride + N) * 4
        )
        assert memory.read(
            padding_address,
            padding_bytes,
        ) == b"\xa5" * padding_bytes
        
def test_execute_descriptor_preserves_partial_sum():
    root = Path(__file__).resolve().parents[1]
    hw = load_hardware(
        str(root / "examples" / "zcu104_16x16.yaml")
    )

    problem = MatMulProblem(
        name="partial_sum_test",
        M=1,
        N=1,
        K=35,
        input_base=0x10000000,
        weight_base=0x20000000,
        output_base=0x30000000,
        input_stride=35,
        weight_stride=1,
        output_stride=1,
    )
    schedule = Schedule(1, 16, 32)

    A = np.ones((1, 35), dtype=np.int8)
    B = np.ones((35, 1), dtype=np.int8)

    A[:, 32:] = 2
    B[32:, :] = 3

    memory = Memory()
    memory.map_region(problem.input_base, 35)
    memory.map_region(problem.weight_base, 35)
    memory.map_region(problem.output_base, 4)

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

    descriptor = encode_descriptor(problem, hw, schedule)
    execute_descriptor(descriptor, hw, memory)

    actual = read_matrix(
        memory,
        base=problem.output_base,
        rows=1,
        cols=1,
        stride=problem.output_stride,
        dtype="<i4",
    )

    # TODO：填写手算得到的完整结果
    assert actual[0, 0] == 50