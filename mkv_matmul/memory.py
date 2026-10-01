import numpy as np

class Memory:
    """ 按独立地址区间保存字节，拒绝重叠映射和越界访问 """

    def __init__(self) -> None:
        self._regions: list[tuple[int, bytearray]] = []

    def map_region(self, base: int, size: int) -> None:
        if base < 0 or size <= 0:
            raise ValueError("映射要求基地址非负、大小为正")

        end = base + size

        if end > (1 << 64):
            raise ValueError("映射超出 64 位地址空间")

        for other_base, buffer in self._regions:
            other_end = other_base + len(buffer)

            if base < other_end and other_base < end:
                raise ValueError("内存映射区间重叠")

        # TODO 1：创建 size 个字节的存储区，初始内容为 0
        buffer = bytearray(size)

        self._regions.append((base, buffer))

    def _locate(
        self,
        address: int,
        size: int,
    ) -> tuple[bytearray, int]:
        """ 找到包含整个访问范围的存储区，返回存储区与内部偏移 """
        if size <= 0:
            raise ValueError("访问大小必须为正")

        for base, buffer in self._regions:
            end = base + len(buffer)

            if base <= address and address + size <= end: # 左闭右开 不重叠 整个访问范围必须包含在当前映射区间内
                # TODO 2：计算 address 相对于该存储区起点的偏移
                offset = address - base
                return buffer, offset

        raise ValueError("访问范围不在已映射内存中")

    def read(self, address: int, size: int) -> bytes:
        buffer, offset = self._locate(address, size)

        # TODO 3：取出对应范围，并转换为 bytes 返回
        return bytes(buffer[offset:offset + size])

    def write(self, address: int, data: bytes) -> None:
        buffer, offset = self._locate(address, len(data))
        buffer[offset:offset + len(data)] = data

def read_matrix(
    memory: Memory,
    base: int,
    rows: int,
    cols: int,
    stride: int,
    dtype: str,
) -> np.ndarray:
    """ 按元素行步长读取矩阵，支持 INT8 和小端 INT32 """
    element_type = np.dtype(dtype)

    if element_type not in (np.dtype("i1"), np.dtype("<i4")):
        raise ValueError("只支持 INT8 和小端 INT32")

    if rows <= 0 or cols <= 0 or stride < cols:
        raise ValueError("矩阵尺寸或行步长无效")

    element_bytes = element_type.itemsize
    result = np.empty((rows, cols), dtype=element_type)

    for i in range(rows):
        # TODO 1：第 i 行有效数据的起始字节地址
        address = base + stride * i * element_bytes

        # TODO 2：本行需要读取的有效字节数
        raw = memory.read(address, cols * element_bytes)

        result[i, :] = np.frombuffer( # 解释字节
            raw,
            dtype=element_type,
            count=cols,
        )

    return result
  
def write_matrix(
    memory: Memory,
    base: int,
    stride: int,
    values: np.ndarray,
) -> None:
    """ 将 INT8 或 INT32 矩阵按行写入，内存中使用小端表示 """
    if values.ndim != 2:
        raise ValueError("写入数据必须是二维矩阵")

    if values.dtype.kind != "i" or values.dtype.itemsize not in (1, 4):
        raise ValueError("只支持 INT8 和 INT32")

    rows, cols = values.shape

    if rows <= 0 or cols <= 0 or stride < cols:
        raise ValueError("矩阵尺寸或行步长无效")

    element_type = values.dtype.newbyteorder("<")
    element_bytes = element_type.itemsize

    for i in range(rows):
        # TODO 3：第 i 行的起始字节地址
        address = base + stride * i * element_bytes

        row = values[i, :].astype(element_type, copy=False)
        memory.write(address, row.tobytes())