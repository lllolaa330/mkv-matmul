from .descriptor_layout import (
    DESCRIPTOR_STRUCT,
    DESCRIPTOR_VERSION,
    MAGIC,
    MODE_MNK,
    OPCODE_MATMUL,
    DESCRIPTOR_FIELDS,
    PROBLEM_FIELDS,
    TILE_FIELDS
)
from .legality import check_legality
from .types import HardwareSpec, MatMulProblem, Schedule, DecodedDescriptor
from dataclasses import asdict

def encode_descriptor(
    problem: MatMulProblem,
    hw: HardwareSpec,
    schedule: Schedule,
) -> bytes:
    """ 检查执行参数，并按协议字段顺序编码 """
    legal, reason = check_legality(problem, hw, schedule)

    if not legal:
        raise ValueError(f"不能编码非法方案: {reason}")

    values = {
      **asdict(problem),
      **asdict(schedule),
      "magic": MAGIC,
      "version": DESCRIPTOR_VERSION,
      "opcode": OPCODE_MATMUL,
      "mode": MODE_MNK,
      "flags": (
          int(hw.double_buffer_input)
          | int(hw.double_buffer_weight) << 1),
      "reserved0": 0,
      "reserved1": 0,
    }
    
    ordered_values = [values[name] for name in DESCRIPTOR_FIELDS]

    return DESCRIPTOR_STRUCT.pack(*ordered_values)

def decode_descriptor(data: bytes) -> DecodedDescriptor:
    """ 按协议字段名解码,并检查格式与基本语义 """
    if len(data) != DESCRIPTOR_STRUCT.size:
        raise ValueError("描述符长度必须为 64 字节")
      
    values = dict(zip(
        DESCRIPTOR_FIELDS,
        DESCRIPTOR_STRUCT.unpack(data),
        strict=True,
    ))
    
    fixed_fields = (
        ("magic", MAGIC),
        ("version", DESCRIPTOR_VERSION),
        ("opcode", OPCODE_MATMUL),
        ("mode", MODE_MNK),
    )
    
    for name, expected in fixed_fields:
        if values[name] != expected:
            raise ValueError(f"描述符 {name} 不受支持")

    if values["flags"] & ~0b11:
        raise ValueError("描述符 flags 包含未知位")

    if values["reserved0"] != 0 or values["reserved1"] != 0:
        raise ValueError("描述符 reserved 字段必须为 0")
    
    for name in ("M", "N", "K") + TILE_FIELDS:
        if values[name] == 0:
            raise ValueError(f"描述符 {name} 必须大于 0 ")

    for stride_name, width_name in (
        ("input_stride", "K"),
        ("weight_stride", "N"),
        ("output_stride", "N"),
    ):
        if values[stride_name] < values[width_name]:
            raise ValueError(f"描述符 {stride_name} 小于矩阵行宽")

    problem_values = {name: values[name] for name in PROBLEM_FIELDS}
    tile_values = {name: values[name]for name in TILE_FIELDS}
    
    problem = MatMulProblem(name = "decoded_matmul", **problem_values)
    schedule = Schedule(**tile_values)
    

    # TODO：返回包含 problem、schedule 和 flags 的解码结果
    return DecodedDescriptor(problem, schedule, values["flags"])
