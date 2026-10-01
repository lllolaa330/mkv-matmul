from .candidates import enumerate_schedules
from .legality import calculate_buffer_usage, check_legality
from .types import CandidateResult, HardwareSpec, MatMulProblem

def evaluate_candidates(
    problem: MatMulProblem,
    hw: HardwareSpec,
) -> list[CandidateResult]:
    """ 评价全部候选,保留原始编号和非法方案 """
    results = []
    for candidate_id, schedule in enumerate(enumerate_schedules(hw)):
        legal, reason = check_legality(problem, hw, schedule)
        usage = calculate_buffer_usage(hw, schedule)
        
        results.append(
            CandidateResult(
              candidate_id = candidate_id,
              schedule = schedule,
              buffer_usage = usage,
              legal = legal,
              illegal_reason = reason,
            )
        )
        
    return results