import csv
import pytest
from dataclasses import replace
from pathlib import Path

from mkv_matmul.config import load_hardware, load_problem
from mkv_matmul.report import write_candidates
from mkv_matmul.search import evaluate_candidates, select_best_candidate  
from mkv_matmul.types import (
    BufferUsage,
    CandidateResult,
    Cost,
    Schedule,
)

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
        
def make_candidate(
    candidate_id: int,
    *, # * 表示，后面的参数必须写名字 例:make_candidate(7, cycles=120)
    cycles: int = 100,
    utilization: float = 0.5,
    input_buffer_bytes: int = 1024,
    legal: bool = True,
) -> CandidateResult:
    cost = None

    if legal:
        cost = Cost(
            total_cycles=cycles,
            compute_cycles=cycles - 20,
            dma_cycles=20,
            input_bytes=128,
            weight_bytes=512,
            output_bytes=256,
            pe_utilization=utilization,
        )

    return CandidateResult(
        candidate_id=candidate_id,
        schedule=Schedule(4, 16, 32),
        buffer_usage=BufferUsage(
            input_bytes=input_buffer_bytes,
            weight_bytes=1024,
            output_bytes=256,
        ),
        legal=legal,
        illegal_reason="" if legal else "INPUT_BUFFER_OVERFLOW",
        cost=cost,
    )
    
@pytest.mark.parametrize(
    "first, second, expected_id",
    [
        # 周期优先，即使另一个候选利用率更高
        (
            make_candidate(8, cycles=90, utilization=0.3),
            make_candidate(3, cycles=100, utilization=0.8),
            8,
        ),

        # 周期相同，利用率更高的优先
        (
            make_candidate(8, utilization=0.8),
            make_candidate(3, utilization=0.5),
            8,
        ),

        # 周期、利用率相同，buffer 总占用更小的优先
        (
            make_candidate(8, input_buffer_bytes=512),
            make_candidate(3, input_buffer_bytes=1024),
            8,
        ),

        # 前三项相同，原始编号更小的优先
        (
            make_candidate(8),
            make_candidate(3),
            3,
        ),
    ],
)
def test_select_best_candidate_uses_ordered_rules(
    first,
    second,
    expected_id,
):
    illegal = make_candidate(0, legal=False)

    best = select_best_candidate([illegal, first, second])
    assert best.candidate_id == expected_id

    # 改变记录排列顺序，不应该改变选择结果
    reversed_best = select_best_candidate([second, first, illegal])
    assert reversed_best.candidate_id == expected_id
    
@pytest.mark.parametrize(
    "results",
    [
        [],
        [make_candidate(0, legal=False)],
    ],
)
def test_select_best_candidate_rejects_no_legal_candidates(results):
    with pytest.raises(ValueError, match="没有合法候选"):
        select_best_candidate(results)
        
def test_select_best_candidate_rejects_missing_cost():
    incomplete = replace(make_candidate(0), cost=None)

    with pytest.raises(ValueError, match="缺少成本"):
        select_best_candidate([incomplete])