import csv
import json
import pytest
from dataclasses import replace
from pathlib import Path

from mkv_matmul.config import load_hardware, load_problem
from mkv_matmul.report import write_raw_candidates, write_best_schedule
from mkv_matmul.types import Schedule
from mkv_matmul.search import evaluate_candidates, select_best_candidate


def test_write_raw_candidates_content(tmp_path):
    schedules = [
        Schedule(4, 32, 64),
        Schedule(1, 16, 32),
    ]

    path = tmp_path / "nested" / "raw_candidates.csv"
    write_raw_candidates(schedules, str(path))

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        assert reader.fieldnames == [
            "candidate_id",
            "tile_m",
            "tile_n",
            "tile_k",
            "loop_order",
        ]

        rows = list(reader)

    assert rows == [
        {
            "candidate_id": "0",
            "tile_m": "4",
            "tile_n": "32",
            "tile_k": "64",
            "loop_order": "MNK",
        },
        {
            "candidate_id": "1",
            "tile_m": "1",
            "tile_n": "16",
            "tile_k": "32",
            "loop_order": "MNK",
        },
    ]
    
def test_write_raw_candidates_is_reproducible(tmp_path):
    schedules = [
        Schedule(4, 32, 64),
        Schedule(1, 16, 32),
    ]
    
    path_a = tmp_path / "a.csv"
    path_b = tmp_path / "b.csv"
    
    write_raw_candidates(schedules, str(path_a))
    write_raw_candidates(schedules, str(path_b))
    
    assert path_a.read_bytes() == path_b.read_bytes()

@pytest.fixture
def selected_context():
    project_root = Path(__file__).resolve().parents[1]

    problem = load_problem(
        str(project_root / "examples" / "q_proj_prefill.yaml")
    )
    original_hw = load_hardware(
        str(project_root / "examples" / "zcu104_16x16.yaml")
    )
    hw = replace(original_hw, weight_buffer_bytes=1024)

    results = evaluate_candidates(problem, hw)
    best = select_best_candidate(results)

    return problem, hw, best
  
def test_write_best_schedule_round_trip(
    selected_context,
    tmp_path,
):
    problem, hw, best = selected_context
    path = tmp_path / "nested" / "best_schedule.json"

    write_best_schedule(problem, hw, best, str(path))

    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["schema_version"] == 1
    assert data["cost_model"] == "serial_v1"
    assert data["candidate_id"] == 36

    assert data["schedule"] == {
        "tile_m": 16,
        "tile_n": 16,
        "tile_k": 32,
        "loop_order": "MNK",
    }

    assert data["buffer_usage"] == {
        "input_bytes": 1024,
        "weight_bytes": 1024,
        "output_bytes": 1024,
    }

    assert data["cost"]["total_cycles"] == 1321472
    assert data["cost"]["compute_cycles"] == 458752
    assert data["cost"]["dma_cycles"] == 862720

    assert data["problem"]["M"] == 16
    assert data["hardware"]["weight_buffer_bytes"] == 1024

    # Python 中的 tuple 写入 JSON 后成为数组，读回来是 list
    assert data["hardware"]["tile_m_candidates"] == [1, 4, 8, 16]
  
@pytest.mark.parametrize(
    "changes, expected_message",
    [
        (
            {"legal": False, "illegal_reason": "INPUT_BUFFER_OVERFLOW"},
            "非法候选",
        ),
        (
            {"cost": None},
            "缺少成本",
        ),
    ],
)
def test_write_best_schedule_rejects_invalid_result(
    selected_context,
    tmp_path,
    changes,
    expected_message,
):
    problem, hw, best = selected_context
    invalid_best = replace(best, **changes)
    path = tmp_path / "best_schedule.json"

    with pytest.raises(ValueError, match=expected_message):
        write_best_schedule(
            problem,
            hw,
            invalid_best,
            str(path),
        )

    assert not path.exists()