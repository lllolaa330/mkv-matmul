import json
from pathlib import Path
from dataclasses import replace
import pytest

from mkv_matmul.descriptor import encode_descriptor
from mkv_matmul.config import load_hardware
from mkv_matmul.report import write_verification
from mkv_matmul.types import MatMulProblem, Schedule
from mkv_matmul.verification import verify_descriptor
from mkv_matmul.emulator import (
    execute_descriptor as real_execute_descriptor,
)

@pytest.fixture
def verification_case():
    root = Path(__file__).resolve().parents[1]
    hw = load_hardware(
        str(root / "examples" / "zcu104_16x16.yaml")
    )

    problem = MatMulProblem(
        name="verification_test",
        M=1,
        N=1,
        K=35,
        input_base=0x10000000,
        weight_base=0x20000000,
        output_base=0x30000000,
        input_stride=38,
        weight_stride=3,
        output_stride=3,
    )
    descriptor = encode_descriptor(
        problem,
        hw,
        Schedule(1, 16, 32),
    )

    return problem, hw, descriptor
  
def test_verification_passes_and_writes_json(
    verification_case,
    tmp_path,
):
    problem, hw, descriptor = verification_case

    result = verify_descriptor(
        problem,
        hw,
        descriptor,
        seed=20260920,
    )

    assert result["passed"] is True
    assert result["checked_elements"] == 1
    assert result["mismatch_count"] == 0
    assert result["max_abs_error"] == 0
    assert result["output_padding_unchanged"] is True

    path = tmp_path / "verification.json"
    write_verification(result, str(path))

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == result
    
def test_verification_detects_missing_output(
    verification_case,
    monkeypatch,
):
    problem, hw, descriptor = verification_case

    def do_nothing(data, hw, memory):
        pass

    monkeypatch.setattr(
        "mkv_matmul.verification.execute_descriptor",
        do_nothing,
    )

    result = verify_descriptor(
        problem,
        hw,
        descriptor,
        seed=20260920,
    )

    assert result["passed"] is False
    assert result["mismatch_count"] == 1
    assert result["max_abs_error"] > 0
    assert result["output_padding_unchanged"] is True
    
def test_verification_reports_error_outside_first_row(
    verification_case,
    monkeypatch,
):
    original_problem, hw, _ = verification_case

    # 原 fixture 只有一行，这里扩展到两行
    problem = replace(original_problem, M=2)

    # 问题发生变化，必须重新编码描述符
    descriptor = encode_descriptor(
        problem,
        hw,
        Schedule(1, 16, 32),
    )

    def execute_with_second_row_error(data, hw, memory):
        # 先执行正确计算
        real_execute_descriptor(data, hw, memory)

        # C[1, 0]：第二行第一个输出元素
        address = (
            problem.output_base
            + problem.output_stride * hw.accumulator_element_bytes
        )

        original_value = int.from_bytes(
            memory.read(address, 4),
            byteorder="little",
            signed=True,
        )

        # 故意把这一个输出元素增加 7
        memory.write(
            address,
            (original_value + 7).to_bytes(
                4,
                byteorder="little",
                signed=True,
            ),
        )

    monkeypatch.setattr(
        "mkv_matmul.verification.execute_descriptor",
        execute_with_second_row_error,
    )

    result = verify_descriptor(
        problem,
        hw,
        descriptor,
        seed=20260920,
    )

    assert result["checked_elements"] == 2
    assert result["mismatch_count"] == 1
    assert result["max_abs_error"] == 7
    assert result["output_padding_unchanged"] is True
    assert result["passed"] is False