from .candidates import enumerate_schedules
from .legality import calculate_buffer_usage, check_legality
from .types import CandidateResult, HardwareSpec, MatMulProblem
from .cost import estimate_cost

def evaluate_candidates(
    problem: MatMulProblem,
    hw: HardwareSpec,
) -> list[CandidateResult]:
    """ 评价全部候选,保留原始编号和非法方案 """
    results = []
    for candidate_id, schedule in enumerate(enumerate_schedules(hw)):
        legal, reason = check_legality(problem, hw, schedule)
        usage = calculate_buffer_usage(hw, schedule)
        
        cost = None
        
        if legal:
            cost = estimate_cost(problem, hw, schedule)
            
        results.append(
            CandidateResult(
              candidate_id = candidate_id,
              schedule = schedule,
              buffer_usage = usage,
              legal = legal,
              illegal_reason = reason,
              cost = cost,
            )
        )
        
    return results
  
def candidate_sort_key(
    result: CandidateResult,
) -> tuple[int, float, int, int]:
    """ 生成比较键: 周期、负利用率、buffer 总占用、候选编号 """
    cost = result.cost

    if cost is None:
        raise ValueError(
            f"候选 {result.candidate_id}: 缺少成本，无法比较"
        )

    usage = result.buffer_usage

    return (
        cost.total_cycles,  # TODO 1：总周期
        - cost.pe_utilization,  # TODO 2：利用率的负数
        (usage.input_bytes + usage.weight_bytes + usage.output_bytes),  # TODO 3：三个 buffer 的总占用
        result.candidate_id,  # TODO 4：原始候选编号
    )
    
def select_best_candidate(
    results: list[CandidateResult],
) -> CandidateResult:
    """ 从合法候选中选择最佳方案 没有合法候选时报错 """
    legal_results = [
        result
        for result in results
        if result.legal
    ]

    if not legal_results:
        raise ValueError("没有合法候选，无法选择最佳方案")

    return min(legal_results, key=candidate_sort_key) # 对每个候选计算比较键 返回整个CandidateResult