import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from mkv_matmul import __main__ as cli


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_FILES = {
    "candidates.csv",
    "best_schedule.json",
    "matmul.desc",
    "verification.json",
    "report.md",
}


@pytest.fixture
def compile_case(tmp_path):
    problem_data = yaml.safe_load(
        (PROJECT_ROOT / "examples" / "q_proj_prefill.yaml")
        .read_text(encoding="utf-8")
    )
    hardware_data = yaml.safe_load(
        (PROJECT_ROOT / "examples" / "zcu104_16x16.yaml")
        .read_text(encoding="utf-8")
    )

    problem_data["shape"] = {"M": 5, "N": 19, "K": 35}
    problem_data["strides"] = {
        "input": 38,
        "weight": 24,
        "output": 21,
    }
    problem_data["seed"] = 12345

    problem_path = tmp_path / "problem.yaml"
    hardware_path = tmp_path / "hardware.yaml"
    output_dir = tmp_path / "output"

    problem_path.write_text(
        yaml.safe_dump(problem_data, sort_keys=False),
        encoding="utf-8",
    )
    hardware_path.write_text(
        yaml.safe_dump(hardware_data, sort_keys=False),
        encoding="utf-8",
    )

    return problem_path, hardware_path, output_dir
  
def invoke_compile(problem_path, hardware_path, output_dir):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "mkv_matmul",
            "compile",
            "--problem",
            str(problem_path),
            "--hardware",
            str(hardware_path),
            "--output",
            str(output_dir),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    
@pytest.mark.parametrize(
    "weight_capacity, expected_code",
    [
        (65536, 0),
        (512, 1),
    ],
)
def test_compile_command_outputs(
    compile_case,
    weight_capacity,
    expected_code,
):
    problem_path, hardware_path, output_dir = compile_case

    hardware_data = yaml.safe_load(
        hardware_path.read_text(encoding="utf-8")
    )
    hardware_data["buffers"]["weight_bytes"] = weight_capacity
    hardware_path.write_text(
        yaml.safe_dump(hardware_data, sort_keys=False),
        encoding="utf-8",
    )

    # 模拟输出目录中残留上次的产物
    output_dir.mkdir()
    for name in EXPECTED_FILES:
        (output_dir / name).write_text("stale", encoding="utf-8")

    process = invoke_compile(
        problem_path,
        hardware_path,
        output_dir,
    )

    assert process.returncode == expected_code, (
        process.stdout + process.stderr
    )

    actual_files = {path.name for path in output_dir.iterdir()}

    with (output_dir / "candidates.csv").open(
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 48

    if expected_code == 0:
        assert actual_files == EXPECTED_FILES
        assert (output_dir / "matmul.desc").stat().st_size == 64

        verification = json.loads(
            (output_dir / "verification.json")
            .read_text(encoding="utf-8")
        )

        assert verification["passed"] is True
        assert verification["seed"] == 12345
        assert verification["checked_elements"] == 5 * 19
        assert verification["mismatch_count"] == 0
        assert verification["max_abs_error"] == 0

        report = (output_dir / "report.md").read_text(
            encoding="utf-8"
        )
        assert "serial_v1" in report

    else:
        assert actual_files == {"candidates.csv"}
        assert "没有合法候选" in process.stderr
        assert all(row["legal"] == "false" for row in rows)
        assert all(
            row["illegal_reason"] == "WEIGHT_BUFFER_OVERFLOW"
            for row in rows
        )
        
def test_compile_returns_failure_when_verification_fails(
    compile_case,
    monkeypatch,
):
    problem_path, hardware_path, output_dir = compile_case

    real_verify = cli.verify_descriptor

    def verify_with_failure(problem, hw, descriptor, seed):
        result = real_verify(problem, hw, descriptor, seed)

        # 模拟验证模块报告数值错误
        result["passed"] = False
        result["mismatch_count"] = 1
        result["max_abs_error"] = 7
        return result

    monkeypatch.setattr(
        cli,
        "verify_descriptor",
        verify_with_failure,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mkv_matmul",
            "compile",
            "--problem",
            str(problem_path),
            "--hardware",
            str(hardware_path),
            "--output",
            str(output_dir),
        ],
    )

    assert cli.main() == 1

    assert {path.name for path in output_dir.iterdir()} == EXPECTED_FILES

    verification = json.loads(
        (output_dir / "verification.json").read_text(encoding="utf-8")
    )
    assert verification["passed"] is False
    assert verification["mismatch_count"] == 1
    assert verification["max_abs_error"] == 7

    report = (output_dir / "report.md").read_text(encoding="utf-8")
    assert "结果：失败" in report