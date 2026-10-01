DIM_BITS = 32
TILE_BITS = 16
BASE_BITS = 64
STRIDE_BITS = 16

def fits_unsigned(value: int, bits: int) -> bool:
    """ 检查数值能否表示为指定位宽的无符号整数 """
    
    return (type(value) == int) and (value >= 0) and (value < 2**bits)

