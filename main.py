from dataclasses import replace
from pathlib import Path

from mkv_matmul.descriptor import encode_descriptor, decode_descriptor
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

descriptor = encode_descriptor(problem, hw, best.schedule)

descriptor_path = Path("build/small_weight_buffer/matmul.desc")
descriptor_path.parent.mkdir(parents=True, exist_ok=True)
descriptor_path.write_bytes(descriptor)

print("描述符字节数 =", len(descriptor))
print("描述符十六进制 =", descriptor.hex(" "))

loaded_bytes = descriptor_path.read_bytes()
decoded = decode_descriptor(loaded_bytes)

print("解码后的矩阵尺寸 =", (
    decoded.problem.M,
    decoded.problem.N,
    decoded.problem.K,
))
print("解码后的分块方案 =", decoded.schedule)
print("解码后的 flags =", bin(decoded.flags))

assert decoded.schedule == best.schedule

output_path = "build/small_weight_buffer/best_schedule.json"

write_best_schedule(
    problem,
    hw,
    best,
    output_path,
)

print("最佳方案 JSON =", output_path)
