from dataclasses import replace
from pathlib import Path
import csv
import pytest
from dataclasses import replace
from pathlib import Path

from mkv_matmul.config import load_hardware, load_problem
from mkv_matmul.report import write_candidates
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
    
    for result in results:
        if result.legal:
            assert result.cost is not None
            assert result.cost.total_cycles > 0
            assert result.cost.total_cycles == (
                result.cost.compute_cycles
                + result.cost.dma_cycles
            )
        else:
            assert result.cost is None
    
def test_write_candidates_leaves_illegal_costs_empty(tmp_path):
    project_root = Path(__file__).resolve().parents[1]

    problem = load_problem(
        str(project_root / "examples" / "q_proj_prefill.yaml")
    )
    original_hw = load_hardware(
        str(project_root / "examples" / "zcu104_16x16.yaml")
    )
    hw = replace(original_hw, weight_buffer_bytes=1024)

    results = evaluate_candidates(problem, hw)

    path = tmp_path / "candidates.csv"
    write_candidates(results, str(path))

    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    legal_rows = [row for row in rows if row["legal"] == "true"]
    illegal_rows = [row for row in rows if row["legal"] == "false"]

    assert len(legal_rows) == 4
    assert len(illegal_rows) == 44

    cost_fields = [
        "total_cycles",
        "compute_cycles",
        "dma_cycles",
        "input_bytes",
        "weight_bytes",
        "output_bytes",
        "pe_utilization",
    ]

    for row in legal_rows:
        assert all(row[field] != "" for field in cost_fields)
        assert int(row["total_cycles"]) > 0

    for row in illegal_rows:
        assert all(row[field] == "" for field in cost_fields)
        assert row["illegal_reason"] == "WEIGHT_BUFFER_OVERFLOW"