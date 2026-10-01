from dataclasses import replace
from pathlib import Path

import pytest

from mkv_matmul.config import load_problem, load_hardware
from mkv_matmul.search import evaluate_candidates


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "capacity, expected_legal_ids",
    [
        (65536, list(range(48))),
        (1024, [0, 12, 24, 36]),
        (512, []),
    ],
)
def test_evaluate_candidates_preserves_all_candidates(
    capacity,
    expected_legal_ids,
):
    problem = load_problem(
        str(PROJECT_ROOT / "examples" / "q_proj_prefill.yaml")
    )
    original_hw = load_hardware(
        str(PROJECT_ROOT / "examples" / "zcu104_16x16.yaml")
    )
    hw = replace(original_hw, weight_buffer_bytes=capacity)

    results = evaluate_candidates(problem, hw)

    # 请补齐下面三项检查：
    # 1. 结果数量始终是 48
    assert len(results) == 48
    # 2. 全部结果的编号依次为 0～47
    assert [result.candidate_id for result in results] == list(range(48))
    # 3. 合法候选编号等于 expected_legal_ids
    assert [
        result.candidate_id
        for result in results
        if result.legal
    ] == expected_legal_ids
    