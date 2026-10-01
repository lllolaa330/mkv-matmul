import pytest
import numpy as np

from mkv_matmul.memory import Memory, read_matrix, write_matrix

def test_memory_stores_rows_with_padding():
    memory = Memory()
    base = 0x10000000

    memory.map_region(base, 10)

    # 第一行从偏移 0 开始
    memory.write(base, bytes([1, 2, 3]))

    # 第二行从偏移 5 开始
    memory.write(base + 5, bytes([4, 5, 6]))

    assert memory.read(base, 10) == bytes([
        1, 2, 3, 0, 0,
        4, 5, 6, 0, 0,
    ])

    assert memory.read(base, 3) == bytes([1, 2, 3])
    assert memory.read(base + 5, 3) == bytes([4, 5, 6])
    
def test_memory_rejects_overlapping_regions():
    memory = Memory()
    base = 0x10000000

    memory.map_region(base, 10)

    with pytest.raises(ValueError, match="重叠"):
        memory.map_region(base + 9, 4)

    # 首尾相接允许映射
    memory.map_region(base + 10, 4)

    memory.write(base + 10, b"\x7f")
    assert memory.read(base + 10, 1) == b"\x7f"
    
@pytest.mark.parametrize(
    "offset, size",
    [
        (-1, 1),  # 起点在区域之前
        (10, 1),  # 起点恰好在区域末尾之外
        (8, 3),   # 起点合法，但访问末尾越界
    ],
)
def test_memory_rejects_out_of_bounds_access(offset, size):
    memory = Memory()
    base = 0x10000000

    memory.map_region(base, 10)

    with pytest.raises(ValueError, match="已映射内存"):
        memory.read(base + offset, size)

    with pytest.raises(ValueError, match="已映射内存"):
        memory.write(base + offset, bytes(size))

    # 被拒绝的写操作不能改变已有内容
    assert memory.read(base, 10) == bytes(10)
    
MATRIX_CASES = [
    (
        "i1",
        [[1, -2, 3], [4, 0, -1]],
        5,
        (
            "01 fe 03 a5 a5 "
            "04 00 ff a5 a5"
        ),
    ),
    (
        "<i4",
        [[1, -2], [0x01020304, -3]],
        3,
        (
            "01 00 00 00 fe ff ff ff a5 a5 a5 a5 "
            "04 03 02 01 fd ff ff ff a5 a5 a5 a5"
        ),
    ),
]

@pytest.mark.parametrize(
    "dtype, values, stride, expected_hex",
    MATRIX_CASES,
)
def test_read_matrix_respects_layout(
    dtype,
    values,
    stride,
    expected_hex,
):
    memory = Memory()
    base = 0x10000000
    raw = bytes.fromhex(expected_hex)

    memory.map_region(base, len(raw))
    memory.write(base, raw)

    expected = np.array(values, dtype=dtype)
    rows, cols = expected.shape

    actual = read_matrix(
        memory,
        base,
        rows,
        cols,
        stride,
        dtype,
    )

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype
    
@pytest.mark.parametrize(
    "dtype, values, stride, expected_hex",
    MATRIX_CASES,
)
def test_write_matrix_preserves_padding(
    dtype,
    values,
    stride,
    expected_hex,
):
    memory = Memory()
    base = 0x10000000
    expected_raw = bytes.fromhex(expected_hex)

    memory.map_region(base, len(expected_raw))

    # 先把整个区域填成非零字节，包括有效数据区和填充区
    memory.write(base, b"\xa5" * len(expected_raw))

    matrix = np.array(values, dtype=dtype)
    write_matrix(memory, base, stride, matrix)

    actual_raw = memory.read(base, len(expected_raw))

    assert actual_raw == expected_raw
    
def test_read_submatrix_uses_parent_stride():
    memory = Memory()
    base = 0x10000000

    # 3 行，每行占据 7 个字节
    memory.map_region(base, 21)

    # 填充区使用非零值，便于发现误读
    memory.write(base, b"\xa5" * 21)

    # 直接按固定地址准备原矩阵，避免依赖矩阵写入函数
    memory.write(base, bytes([10, 11, 12, 13, 14]))
    memory.write(base + 7, bytes([20, 21, 22, 23, 24]))
    memory.write(base + 14, bytes([30, 31, 32, 33, 34]))

    # TODO 1：A[1, 1] 的地址，INT8 每个元素占 1 字节
    block_base = base + 8

    actual = read_matrix(
        memory,
        base=block_base,
        rows=2,
        cols=3,
        stride= 7,  # TODO 2：原存储区中的行步长
        dtype="i1",
    )

    expected = np.array(
        [
            [21, 22, 23],
            [31, 32, 33],
        ],
        dtype=np.int8,
    )

    np.testing.assert_array_equal(actual, expected)