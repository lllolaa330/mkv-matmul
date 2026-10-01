from dataclasses import replace
from pathlib import Path

import pytest

from mkv_matmul.config import load_hardware, load_problem
from mkv_matmul.cost import estimate_cost
from mkv_matmul.legality import check_legality
from mkv_matmul.types import Schedule


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    (
        "K, expected_input_bytes, expected_weight_bytes, "
        "expected_compute_cycles, expected_dma_cycles, "
        "expected_total_cycles"
    ),
    [
        (32, 128, 512, 56, 116, 172),
        (33, 132, 528, 81, 158, 239),
    ],
)
def test_estimate_cost_matches_hand_calculation(
    K,
    expected_input_bytes,
    expected_weight_bytes,
    expected_compute_cycles,
    expected_dma_cycles,
    expected_total_cycles,
):
    original_problem = load_problem(
        str(PROJECT_ROOT / "examples" / "q_proj_prefill.yaml")
    )
    hw = load_hardware(
        str(PROJECT_ROOT / "examples" / "zcu104_16x16.yaml")
    )

    problem = replace(
        original_problem,
        M=4,
        N=16,
        K=K,
        input_stride=K,
        weight_stride=16,
        output_stride=16,
    )
    schedule = Schedule(4, 16, 32)

    assert check_legality(problem, hw, schedule) == (True, "")

    cost = estimate_cost(problem, hw, schedule)

    assert cost.input_bytes == expected_input_bytes
    assert cost.weight_bytes == expected_weight_bytes
    assert cost.output_bytes == 256

    assert cost.compute_cycles == expected_compute_cycles
    assert cost.dma_cycles == expected_dma_cycles
    assert cost.total_cycles == expected_total_cycles

    expected_utilization = (
        4 * 16 * K
        / (16 * 16 * expected_compute_cycles)
    )
    assert cost.pe_utilization == pytest.approx(expected_utilization)