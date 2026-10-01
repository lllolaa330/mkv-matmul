import csv 
from collections.abc import Iterator
from pathlib import Path

from .types import Schedule, CandidateResult

RAW_CANDIDATE_FIELDS = [
    "candidate_id",
    "tile_m",
    "tile_n",
    "tile_k",
    "loop_order",
]

CANDIDATE_FIELDS = [
    "candidate_id",
    "tile_m",
    "tile_n",
    "tile_k",
    "loop_order",
    "legal",
    "buffer_a_bytes",
    "buffer_b_bytes",
    "buffer_c_bytes",
    "illegal_reason",
]

def write_raw_candidates(
    schedules: Iterable[Schedule],
    path: str,
) -> None:
    """ 按原始顺序为候选编号,并写入csv """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open(
        "w",
        encoding = "utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
          file, 
          fieldnames=RAW_CANDIDATE_FIELDS
        )
        writer.writeheader()
        
        for candidate_id, schedule in enumerate(schedules):
            writer.writerow({
                "candidate_id": candidate_id,
                "tile_m": schedule.tile_m,
                "tile_n": schedule.tile_n,
                "tile_k": schedule.tile_k,
                "loop_order": schedule.loop_order,
            })
            
def write_candidates(
    results: Iterable[CandidateResult],
    path: str,
) -> None:
    """ 写出全部候选的评价结果 """

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=CANDIDATE_FIELDS)
        writer.writeheader()

        for result in results:
            writer.writerow({
                "candidate_id": result.candidate_id,
                "tile_m": result.schedule.tile_m,
                "tile_n": result.schedule.tile_n,
                "tile_k": result.schedule.tile_k,
                "loop_order": result.schedule.loop_order,
                "legal": "true" if result.legal else "false",
                "buffer_a_bytes": result.buffer_usage.input_bytes,
                "buffer_b_bytes": result.buffer_usage.weight_bytes,
                "buffer_c_bytes": result.buffer_usage.output_bytes,
                "illegal_reason": result.illegal_reason,  
            })