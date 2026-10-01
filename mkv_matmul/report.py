import csv 
from collections.abc import Iterator
from pathlib import Path
from dataclasses import asdict

from .types import Schedule, CandidateResult

RAW_CANDIDATE_FIELDS = [
    "candidate_id",
    "tile_m",
    "tile_n",
    "tile_k",
    "loop_order",
]

COST_FIELDS = [
    "total_cycles",
    "compute_cycles",
    "dma_cycles",
    "input_bytes",
    "weight_bytes",
    "output_bytes",
    "pe_utilization",
]

CANDIDATE_FIELDS = [
    "candidate_id",
    "tile_m",
    "tile_n",
    "tile_k",
    "loop_order",
    "legal",
    "buffer_a_bytes",  # 输入数据需要的片上 buffer 容量
    "buffer_b_bytes",
    "buffer_c_bytes",
    "illegal_reason",
] + COST_FIELDS

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
            row = {
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
            }
            
            for field in COST_FIELDS:
                row[field] = ""
            
            if result.cost is not None:
                row.update(asdict(result.cost)) # asdict: 把 dataclass 对象转换成字典
            
            writer.writerow(row)