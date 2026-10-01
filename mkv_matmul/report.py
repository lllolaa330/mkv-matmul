import csv 
import json
from collections.abc import Iterable
from pathlib import Path
from dataclasses import asdict

from .types import Schedule, CandidateResult, HardwareSpec, MatMulProblem

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
    rows = []
    
    for candidate_id, schedule in enumerate(schedules):
        rows.append({
            "candidate_id" : candidate_id,
            **asdict(schedule),
        })

    write_csv(rows, RAW_CANDIDATE_FIELDS, path)
    
            
def write_candidates(
    results: Iterable[CandidateResult],
    path: str,
) -> None:
    """ 写出全部候选的评价结果 """
    rows = []
    
    for result in results:
        row = {
            "candidate_id": result.candidate_id,
            **asdict(result.schedule),
            "legal": "true" if result.legal else "false",
            "buffer_a_bytes": result.buffer_usage.input_bytes,
            "buffer_b_bytes": result.buffer_usage.weight_bytes,
            "buffer_c_bytes": result.buffer_usage.output_bytes,
            "illegal_reason": result.illegal_reason,
        }
        
        for field in COST_FIELDS:
            row[field] = ""

        if result.cost is not None:
            row.update(asdict(result.cost))
    
        rows.append(row)
    
    # TODO 2：调用公共 CSV 写出函数，使用完整候选表的列定义
    write_csv(rows, CANDIDATE_FIELDS, path)
        
            
def write_best_schedule(
    problem: MatMulProblem,
    hw: HardwareSpec,
    best: CandidateResult,
    path: str,
) -> None:
    """ 保存最佳方案、配置快照与预测成本 """
    if not best.legal:
        raise ValueError("不能保存非法候选为最佳方案")

    if best.cost is None:
        raise ValueError("最佳候选缺少成本")

    payload = {
        "schema_version": 1,
        "cost_model": "serial_v1",
        "problem": asdict(problem),
        "hardware": asdict(hw),
        "candidate_id": best.candidate_id,

        # TODO：将对应的 dataclass 对象转换成字典
        "schedule": asdict(best.schedule),
        "buffer_usage": asdict(best.buffer_usage),
        "cost": asdict(best.cost),
    }

    write_json(payload, path)
    
def write_verification(result: dict, path: str) -> None:
    """ 保存软件执行验证结果 """
    write_json(result, path)
    
def write_json(payload: dict, path: str) -> None:
    """ 将字典写成格式稳定的 JSON 文件 """
    text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text + "\n", encoding="utf-8")


def write_csv(
    rows: Iterable[dict],
    fields: list[str],
    path: str,
) -> None:
    """ 按指定列顺序写出 CSV 表头和数据行 """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        # TODO 1：一次写出 rows 中的全部记录
        writer.writerows(rows)