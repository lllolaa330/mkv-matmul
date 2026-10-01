import struct

DIM_BITS = 32
TILE_BITS = 16
BASE_BITS = 64
STRIDE_BITS = 16

MAGIC = 0x4D4B5631
DESCRIPTOR_VERSION = 1
OPCODE_MATMUL = 1
MODE_MNK = 0

DESCRIPTOR_STRUCT = struct.Struct(
    "<IHBBIIIIHHHHQQQHHHH"
)
# < 小端序 固定标准宽度 不自动插入对齐填充
# B 1字节无符号整数
# H 2字节无符号整数
# I 4字节无符号整数
# Q 8字节无符号整数
# IHBB 8 + IIII 16 + HHHH 8 + QQQ 24 + HHHH 8 = 64 字节

def fits_unsigned(value: int, bits: int) -> bool:
    """ 检查数值能否表示为指定位宽的无符号整数 """
    
    return (type(value) == int) and (value >= 0) and (value < 2**bits)
