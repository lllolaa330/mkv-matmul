import numpy as np

from mkv_matmul.reference import generate_inputs, reference_matmul
from mkv_matmul.types import MatMulProblem

def test_reference_matmul_matches_hand_calculation():
    A = np.array(
        [
            [1, -2, 3],
            [4, 0, -1],
        ],
        dtype=np.int8,
    )

    B = np.array(
        [
            [2, 1],
            [-3, 4],
            [5, -2],
        ],
        dtype=np.int8,
    )

    expected = np.array(
        [
            [23, -13],
            [3, 6],
        ],
        dtype=np.int64,
    )

    actual = reference_matmul(A, B)

    # assert_array_equal 逐元素检查
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.int64
    
def test_reference_matmul_widens_before_computing():
    A = np.array(
        [[100, 100, 100, 100]],
        dtype=np.int8,
    )
    B = np.array(
        [[100], [100], [100], [100]],
        dtype=np.int8,
    )

    actual = reference_matmul(A, B)

    assert actual.shape == (1, 1)
    assert actual.dtype == np.int64
    assert actual[0, 0] == 40000
    
def test_generate_inputs_is_reproducible():
    problem = MatMulProblem(
        name="tiny",
        M=2,
        N=4,
        K=3,
        input_base=0x10000000,
        weight_base=0x20000000,
        output_base=0x30000000,
        input_stride=3,
        weight_stride=4,
        output_stride=4,
    )

    A1, B1 = generate_inputs(problem, seed=20260920)
    A2, B2 = generate_inputs(problem, seed=20260920)

    assert A1.shape == (2, 3)
    assert B1.shape == (3, 4)

    assert A1.dtype == np.int8
    assert B1.dtype == np.int8

    assert np.all((A1 >= -8) & (A1 < 8))
    assert np.all((B1 >= -8) & (B1 < 8))

    np.testing.assert_array_equal(A1, A2)
    np.testing.assert_array_equal(B1, B2)