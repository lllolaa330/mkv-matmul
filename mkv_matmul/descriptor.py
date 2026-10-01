from .descriptor_layout import (
    DESCRIPTOR_STRUCT,
    DESCRIPTOR_VERSION,
    MAGIC,
    MODE_MNK,
    OPCODE_MATMUL,
)
from .legality import check_legality
from .types import HardwareSpec, MatMulProblem, Schedule, DecodedDescriptor

def encode_descriptor(
    problem: MatMulProblem,
    hw: HardwareSpec,
    schedule: Schedule,
) -> bytes:
    """ 检查执行参数，并编码成 64 字节的小端描述符 """
    legal, reason = check_legality(problem, hw, schedule)

    if not legal:
        raise ValueError(f"不能编码非法方案: {reason}")

    flags = (
        int(hw.double_buffer_input)
        | (int(hw.double_buffer_weight) << 1)
    )

    return DESCRIPTOR_STRUCT.pack(
        MAGIC,
        DESCRIPTOR_VERSION,
        OPCODE_MATMUL,
        MODE_MNK,
        flags,
        problem.M,
        problem.N,
        problem.K,

        schedule.tile_m,  # TODO 1：tile_m
        schedule.tile_n,  # TODO 2：tile_n
        schedule.tile_k,  # TODO 3：tile_k

        0,  # reserved
        problem.input_base,
        problem.weight_base,
        problem.output_base,
        problem.input_stride,
        problem.weight_stride,
        problem.output_stride,
        0,  # reserved
    )

def decode_descriptor(data: bytes) -> DecodedDescriptor:
    """ 解码并检查描述符自身的格式与基本语义 """
    if len(data) != DESCRIPTOR_STRUCT.size:
        raise ValueError("描述符长度必须为 64 字节")

    (
        magic,
        version,
        opcode,
        mode,
        flags,
        M,
        N,
        K,
        tile_m,
        tile_n,
        tile_k,
        reserved0,
        input_base,
        weight_base,
        output_base,
        input_stride,
        weight_stride,
        output_stride,
        reserved1,
    ) = DESCRIPTOR_STRUCT.unpack(data)

    fixed_fields = (
        ("magic", magic, MAGIC),
        ("version", version, DESCRIPTOR_VERSION),
        ("opcode", opcode, OPCODE_MATMUL),
        ("mode", mode, MODE_MNK),
    )

    for name, actual, expected in fixed_fields:
        if actual != expected:
            raise ValueError(f"描述符 {name} 不受支持")

    if flags & ~0b11:
        raise ValueError("描述符 flags 包含未知位")

    if reserved0 != 0 or reserved1 != 0:
        raise ValueError("描述符 reserved 字段必须为 0")

    positive_fields = (
        ("M", M),
        ("N", N),
        ("K", K),
        ("tile_m", tile_m),
        ("tile_n", tile_n),
        ("tile_k", tile_k),
    )

    for name, value in positive_fields:
        if value == 0:
            raise ValueError(f"描述符 {name} 必须大于 0")

    stride_fields = (
        ("input_stride", input_stride, K),
        ("weight_stride", weight_stride, N),
        ("output_stride", output_stride, N),
    )

    for name, value, minimum in stride_fields:
        if value < minimum:
            raise ValueError(f"描述符 {name} 小于矩阵行宽")

    problem = MatMulProblem(
        name="decoded_matmul",
        M=M,
        N=N,
        K=K,
        input_base=input_base,
        weight_base=weight_base,
        output_base=output_base,
        input_stride=input_stride,
        weight_stride=weight_stride,
        output_stride=output_stride,
    )

    # TODO：用解包得到的三个 tile 值构造 Schedule
    schedule = Schedule(tile_m, tile_n, tile_k)

    # TODO：返回包含 problem、schedule 和 flags 的解码结果
    return DecodedDescriptor(problem, schedule, flags)
