from dataclasses import replace
from pathlib import Path
import pytest

from mkv_matmul.config import load_hardware, load_problem
from mkv_matmul.legality import calculate_buffer_usage, check_schedule_legality, check_legality
from mkv_matmul.types import BufferUsage, Schedule


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_example_hardware():
    path = PROJECT_ROOT / "examples" / "zcu104_16x16.yaml"
    return load_hardware(str(path))


def test_buffer_usage_with_double_buffering():
    hw = load_example_hardware()
    schedule = Schedule(16, 64, 256)

    usage = calculate_buffer_usage(hw, schedule)

    assert usage == BufferUsage(
        input_bytes=8192,
        weight_bytes=32768,
        output_bytes=4096,
    )
    
def test_buffer_usage_without_input_double_buffering():
    original_hw = load_example_hardware()

    hw = replace(
        original_hw,
        double_buffer_input=False,
    )

    schedule = Schedule(16, 64, 256)
    usage = calculate_buffer_usage(hw, schedule)

    assert usage == BufferUsage(
        input_bytes=4096,
        weight_bytes=32768,
        output_bytes=4096,
    )
    
@pytest.mark.parametrize(
    "field, capacity, expected_legal, expected_reason",
    [
        ("input_buffer_bytes", 8192, True, ""),
        ("input_buffer_bytes", 8191, False, "INPUT_BUFFER_OVERFLOW"),
        ("weight_buffer_bytes", 32768, True, ""),
        ("weight_buffer_bytes", 32767, False, "WEIGHT_BUFFER_OVERFLOW"),
        ("output_buffer_bytes", 4096, True, ""),
        ("output_buffer_bytes", 4095, False, "OUTPUT_BUFFER_OVERFLOW"),
    ],
)
def test_schedule_buffer_boundary(
    field,
    capacity,
    expected_legal,
    expected_reason,
):
    original_hw = load_example_hardware()
    hw = replace(original_hw, **{field: capacity})

    schedule = Schedule(16, 64, 256)

    result = check_schedule_legality(hw, schedule)

    assert result == (expected_legal, expected_reason)
    
@pytest.mark.parametrize(
    "schedule, expected_reason",
    [
        (Schedule(0, 16, 32), "INVALID_TILE"),
        (Schedule(True, 16, 32), "INVALID_TILE"),
        (Schedule(1, 24, 32), "UNSUPPORTED_TILE"),
        (
            Schedule(1, 16, 32, loop_order="KNM"),
            "UNSUPPORTED_LOOP_ORDER",
        ),
    ],
)
def test_schedule_rejects_invalid_parameters(
    schedule,
    expected_reason,
):
    hw = load_example_hardware()

    # 调用检查函数
    result = check_schedule_legality(hw, schedule)
    
    # 断言结果为 (False, expected_reason)
    assert result == (False, expected_reason)
    
def load_example_problem():
    path = PROJECT_ROOT / "examples" / "q_proj_prefill.yaml"
    return load_problem(str(path))

def test_check_legality_accepts_default_case():
    problem = load_example_problem()
    hw = load_example_hardware()
    schedule = Schedule(16, 64, 256)

    assert check_legality(problem, hw, schedule) == (True, "")
    
@pytest.mark.parametrize(
    "field, expected_reason",
    [
        ("input_base", "UNALIGNED_INPUT_BASE"),
        ("weight_base", "UNALIGNED_WEIGHT_BASE"),
        ("output_base", "UNALIGNED_OUTPUT_BASE"),
    ],
)
def test_check_legality_rejects_unaligned_base(
    field,
    expected_reason,
):
    original_problem = load_example_problem()
    hw = load_example_hardware()
    schedule = Schedule(16, 64, 256)

    original_base = getattr(original_problem, field)

    problem = replace(
        original_problem,
        **{field: original_base + 1},
    )

    assert check_legality(problem, hw, schedule) == (
        False,
        expected_reason,
    )
    
@pytest.mark.parametrize(
    "field, width_field",
    [
        ("input_stride", "K"),
        ("weight_stride", "N"),
        ("output_stride", "N"),
    ],
)
def test_check_legality_rejects_small_stride(field, width_field):
    original_problem = load_example_problem()
    hw = load_example_hardware()
    schedule = Schedule(16, 64, 256)

    minimum_width = getattr(original_problem, width_field)

    # 创建新 problem，把对应 stride 改成 minimum_width - 1
    problem = replace(original_problem, **{field:minimum_width - 1})

    # 调用 check_legality
    assert check_legality(problem, hw, schedule) == (False, "INVALID_STRIDE")
    # 断言结果是 (False, "INVALID_STRIDE")
    
def test_check_legality_accepts_padded_output_stride():
    original_problem = load_example_problem()
    hw = load_example_hardware()
    schedule = Schedule(16, 64, 256)

    # 创建新 problem，把 output_stride 改成 N + 4
    N = getattr(original_problem, "N")
    problem = replace(original_problem, **{"output_stride":N+4})
    
    # 断言 check_legality 返回 (True, "")
    assert check_legality(problem, hw, schedule) == (True, "")
    
def test_check_legality_rejects_stride_field_overflow():
    original_problem = load_example_problem()
    hw = load_example_hardware()
    schedule = Schedule(16, 64, 256)

    # 创建新 problem，将 output_stride 改成 65536
    problem = replace(original_problem, **{"output_stride":65536})
   
    # 调用统一入口 check_legality()
    assert check_legality(problem, hw, schedule) ==(False, "DESCRIPTOR_FIELD_OUT_OF_RANGE:output_stride")
