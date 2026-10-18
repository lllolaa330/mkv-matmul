import argparse
import sys
from dataclasses import asdict
from pathlib import Path

from .config import load_problem, load_hardware, load_yaml, require_int
from .candidates import enumerate_schedules
from .report import write_best_schedule,write_candidates,write_json,write_raw_candidates,write_report,write_verification
from .descriptor import encode_descriptor
from .search import evaluate_candidates, select_best_candidate
from .verification import verify_descriptor

def build_parser():
    parser = argparse.ArgumentParser(
        description="面向 FPGA 的矩阵乘法调度编译器"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    check_parser = subparsers.add_parser(
        "check-config",
        help="读取、检查问题与硬件配置，并生成摘要",
    )
    
    candidates_parser = subparsers.add_parser(
        "list-candidates",
        help="枚举硬件配置中的全部分块候选,并输出CSV",
    )

    check_parser.add_argument("--problem", required=True)
    check_parser.add_argument("--hardware", required=True)
    check_parser.add_argument("--output", required=True)
    
    candidates_parser.add_argument("--hardware", required=True)
    candidates_parser.add_argument("--output", required=True)
    
    compile_parser = subparsers.add_parser("compile", help="搜索分块方案、生成描述符并执行软件验证")
    compile_parser.add_argument("--problem", required=True)
    compile_parser.add_argument("--hardware", required=True)
    compile_parser.add_argument("--output", required=True)
    
    return parser
  
def run_check_config(args):
    problem = load_problem(args.problem)
    hardware = load_hardware(args.hardware)

    summary = {
        "schema_version": 1,
        "problem": asdict(problem),
        "hardware": asdict(hardware),
    }

    summary_path = Path(args.output) / "config_summary.json"
    write_json(summary, str(summary_path))

    print("配置检查通过")
    print(f"问题：{problem.name}")
    print(f"矩阵尺寸：M={problem.M}, N={problem.N}, K={problem.K}")
    print(f"硬件：{hardware.name}")
    print(f"摘要已写入：{summary_path}")
    
def run_list_candidates(args):
    hardware = load_hardware(args.hardware)
    schedules = list(enumerate_schedules(hardware))

    output_path = Path(args.output) / "raw_candidates.csv"

    write_raw_candidates(schedules, str(output_path))

    print(f"硬件：{hardware.name}")
    print(f"原始候选数量：{len(schedules)}")
    print(f"候选表已写入：{output_path}")
    
OUTPUT_FILES = (
    "candidates.csv",
    "best_schedule.json",
    "matmul.desc",
    "verification.json",
    "report.md",
)

def run_compile(args) -> int:
    """ 执行完整编译流程，成功返回 0 验证失败返回 1 """
    problem = load_problem(args.problem)
    hw = load_hardware(args.hardware)

    seed = require_int(
        load_yaml(args.problem),
        "seed",
        "problem",
        minimum=0,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 刷新这五个产物，避免本次失败时混入上次的结果
    for filename in OUTPUT_FILES:
        (output_dir / filename).unlink(missing_ok=True)

    results = evaluate_candidates(problem, hw)

    # 先保存全部候选；即使没有合法方案，也保留诊断信息
    write_candidates(
        results,
        str(output_dir / "candidates.csv"),
    )

    best = select_best_candidate(results)

    write_best_schedule(
        problem,
        hw,
        best,
        str(output_dir / "best_schedule.json"),
    )

    descriptor = encode_descriptor(problem, hw, best.schedule)

    descriptor_path = output_dir / "matmul.desc"
    descriptor_path.write_bytes(descriptor)

    verification = verify_descriptor(
        problem,
        hw,
        descriptor_path.read_bytes(),
        seed=seed,
    )

    write_verification(
        verification,
        str(output_dir / "verification.json"),
    )

    write_report(
        problem,
        hw,
        results,
        best,
        verification,
        str(output_dir / "report.md"),
    )

    legal_count = sum(result.legal for result in results)
    status = "通过" if verification["passed"] else "失败"

    print(f"候选数量：{len(results)}，合法数量：{legal_count}")
    print(f"最佳候选：{best.candidate_id}，{best.schedule}")
    print(f"软件执行验证：{status}")
    print(f"产物目录：{output_dir.resolve()}")

    # TODO 2：验证通过返回 0，否则返回 1
    return 0 if verification["passed"] else 1
    
def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "compile":
            return run_compile(args)
        if args.command == "check-config":
            run_check_config(args)
        elif args.command == "list-candidates":
            run_list_candidates(args)
    except (OSError, UnicodeError, ValueError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())