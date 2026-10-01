import csv

from mkv_matmul.report import write_raw_candidates
from mkv_matmul.types import Schedule


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