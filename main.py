from dataclasses import replace

from mkv_matmul.config import load_hardware, load_problem
from mkv_matmul.report import write_candidates, write_best_schedule
from mkv_matmul.search import evaluate_candidates, select_best_candidate

problem = load_problem("examples/q_proj_prefill.yaml")
hw = load_hardware("examples/zcu104_16x16.yaml")
hw = replace(hw, weight_buffer_bytes=1024)

results = evaluate_candidates(problem, hw)


output_path = "build/small_weight_buffer/candidates.csv"
write_candidates(results, output_path)

best = select_best_candidate(results)

print("候选 CSV =", output_path)
print("最佳候选编号 =", best.candidate_id)
print("最佳分块方案 =", best.schedule)
print("预测成本 =", best.cost)
print("Buffer 占用 =", best.buffer_usage)

output_path = "build/small_weight_buffer/best_schedule.json"

write_best_schedule(
    problem,
    hw,
    best,
    output_path,
)

print("最佳方案 JSON =", output_path)
