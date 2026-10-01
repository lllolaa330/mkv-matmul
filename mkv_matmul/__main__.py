import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .config import load_problem, load_hardware
from .candidates import enumerate_schedules
from .report import write_json, write_raw_candidates


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
    
def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
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