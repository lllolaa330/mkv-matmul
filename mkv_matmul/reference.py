import numpy as np

from .types import MatMulProblem


def generate_inputs(
    problem: MatMulProblem,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """ 生成可重复的 INT8 输入矩阵 """
    rng = np.random.default_rng(seed)

    # TODO 1：填写 A 的形状
    A = rng.integers(
        low=-8,
        high=8,
        size=(problem.M, problem.K),
        dtype=np.int8,
    )

    # TODO 2：填写 B 的形状
    B = rng.integers(
        low=-8,
        high=8,
        size=(problem.K, problem.N),
        dtype=np.int8,
    )

    return A, B
  
def reference_matmul(
    A: np.ndarray,
    B: np.ndarray,
) -> np.ndarray:
    """ 使用 INT64 运算生成独立参考结果 """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("参考计算要求两个二维矩阵")

    if A.dtype != np.int8 or B.dtype != np.int8:
        raise ValueError("参考计算要求 INT8 输入")

    if A.shape[1] != B.shape[0]:
        raise ValueError("A 的列数必须等于 B 的行数")

    # TODO 3：先将两边转换为 INT64，再进行矩阵乘法
    return A.astype(np.int64) @ B.astype(np.int64)
    # (A @ B).astype(np.int64) 不对 可能会在计算时溢出