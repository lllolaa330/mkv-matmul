from pathlib import Path
import pytest
import yaml
from mkv_matmul.config import (
    load_problem,
    load_hardware,
    require_bool,
    require_int_tuple,
    load_yaml,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = PROJECT_ROOT / "examples" /"q_proj_prefill.yaml"

@pytest.fixture
def problem_data():
    """ 每个测试获得一份独立的问题配置字典 """
    with EXAMPLE_PATH.open("r", encoding = "utf-8") as file:
        return yaml.safe_load(file)
      
def write_problem(tmp_path, data):
    """ 把测试配置写入临时文件, 并返回路径字符串 """
    path = tmp_path / "problem.yaml"
    path.write_text(
        yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )
    return str(path)

@pytest.fixture
def hardware_data():
    """ 每个测试获得一份独立的硬件配置 """
    path = PROJECT_ROOT / "examples" / "zcu104_16x16.yaml"
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def write_hardware(tmp_path, data):
    """ 把硬件测试配置写入临时文件 """
    path = tmp_path / "hardware.yaml"
    path.write_text(
        yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )
    return str(path)

def test_load_problem_valid(problem_data, tmp_path):
    path = write_problem(tmp_path, problem_data)
    problem = load_problem(path)
    
    assert problem.name == "q_proj_prefill"
    assert (problem.M, problem.N, problem.K) == (16, 2048, 2048)
    
    assert problem.input_base == 0x10000000
    assert problem.weight_base == 0x20000000
    assert problem.output_base == 0x30000000

    assert problem.input_stride == 2048
    assert problem.weight_stride == 2048
    assert problem.output_stride == 2048

@pytest.mark.parametrize("bad_M", [0, -1, True, 16.0, "16"])
def test_load_problem_rejects_invalid_M(
    problem_data,
    tmp_path,
    bad_M,
):
    problem_data["shape"]["M"] = bad_M
    path = write_problem(tmp_path, problem_data)
    
    with pytest.raises(ValueError) as exc_info:
        load_problem(path)
    
    assert "problem.shape.M" in str(exc_info.value)
    
def test_load_problem_rejects_unsupported_dtype(
    problem_data,
    tmp_path
):
    problem_data["dtype"]["input"] = "float32"
    path = write_problem(tmp_path, problem_data)
    with pytest.raises(ValueError) as exc_info:
        load_problem(path)
    assert "problem.dtype.input" in str(exc_info.value)
    
def test_load_problem_accepts_padded_output_stride(
    problem_data,
    tmp_path
):
    problem_data["strides"]["output"] = problem_data["shape"]["N"] + 4
    path = write_problem(tmp_path, problem_data)
    problem = load_problem(path)
    assert problem.output_stride == problem_data["shape"]["N"] + 4
    
def test_require_bool_valid():
    assert require_bool({"enabled": True}, "enabled", "test") is True
    assert require_bool({"enabled": False}, "enabled", "test") is False

def test_require_int_tuple_valid():
    result = require_int_tuple(
        {"tile_m": [1, 4, 8, 16]},
        "tile_m",
        "hardware.candidates",
    )

    assert result == (1, 4, 8, 16)
    assert isinstance(result, tuple)
    
@pytest.mark.parametrize("bad_value", [0, 1, "true", "false", None])
def test_require_bool_rejects_invalid_value(bad_value):
    with pytest.raises(ValueError):
        require_bool({"enabled": bad_value}, "enabled", "test")

@pytest.mark.parametrize(
    "bad_value",
    [
        16,
        "1,4,8,16",
        [],
        [1, 0, 16],
        [1, -4, 16],
        [1, True, 16],
        [1, 4.0, 16],
        [1, "4", 16],
        [1, 4, 4, 16],
    ],
)
def test_require_int_tuple_rejects_invalid_value(bad_value):
    with pytest.raises(ValueError):
        require_int_tuple(
            {"tile_m": bad_value},
            "tile_m",
            "hardware.candidates",
        )
        
def test_load_hardware_valid():
    path = PROJECT_ROOT / "examples" / "zcu104_16x16.yaml"
    hw = load_hardware(str(path))

    assert hw.name == "zcu104_16x16"
    assert (hw.array_rows, hw.array_cols) == (16, 16)
    assert hw.pipeline_cycles == 24

    assert hw.input_element_bytes == 1
    assert hw.weight_element_bytes == 1
    assert hw.accumulator_element_bytes == 4

    assert hw.input_buffer_bytes == 32768
    assert hw.weight_buffer_bytes == 65536
    assert hw.output_buffer_bytes == 32768

    assert hw.double_buffer_input is True
    assert hw.double_buffer_weight is True

    assert hw.dma_bytes_per_cycle == 16
    assert hw.dma_setup_cycles == 20
    assert hw.alignment_bytes == 64

    assert hw.tile_m_candidates == (1, 4, 8, 16)
    assert hw.tile_n_candidates == (16, 32, 64)
    assert hw.tile_k_candidates == (32, 64, 128, 256)
    
@pytest.mark.parametrize(
    "section, key, bad_value",
    [
        ("array", "rows", 0),
        ("array", "pipeline_cycles", -1),
        ("data", "input_bytes", 2),
        ("buffers", "weight_bytes", -1024),
        ("buffers", "double_buffer_input", 1),
        ("dma", "bytes_per_cycle", 0),
        ("candidates", "tile_m", []),
        ("candidates", "tile_m", [1, True, 16]),
    ],
)
def test_load_hardware_rejects_invalid_field(
    hardware_data,
    tmp_path,
    section,
    key,
    bad_value,
):
    hardware_data[section][key] = bad_value
    path = write_hardware(tmp_path, hardware_data)

    with pytest.raises(ValueError) as exc_info:
        load_hardware(path)

    expected_path = f"hardware.{section}.{key}"
    assert expected_path in str(exc_info.value)

def test_load_hardware_accepts_zero_overhead(
    hardware_data,
    tmp_path
):
    hardware_data["array"]["pipeline_cycles"] = 0
    hardware_data["dma"]["setup_cycles"] = 0
    
    path = write_hardware(tmp_path, hardware_data)
    hw = load_hardware(path)
    
    assert hw.pipeline_cycles == 0
    assert hw.dma_setup_cycles == 0
  
def test_load_hardware_accepts_small_weight_buffer(
    hardware_data,
    tmp_path,
):
    hardware_data["buffers"]["weight_bytes"] = 1024

    path = write_hardware(tmp_path, hardware_data)
    hw = load_hardware(path)
    
    assert hw.weight_buffer_bytes == 1024
    assert hw.tile_n_candidates == (16, 32, 64)
    assert hw.tile_k_candidates == (32, 64, 128, 256)
    
def test_load_yaml_rejects_missing_file(tmp_path):
    path = tmp_path / "not_created.yaml"

    with pytest.raises(FileNotFoundError):
        load_yaml(str(path))
        
def test_load_yaml_rejects_invalid_syntax(tmp_path):
    path = tmp_path / "broken.yaml"

    path.write_text(
        "shape: [16, 2048\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        load_yaml(str(path))

    assert "YAML 语法错误" in str(exc_info.value)
    assert str(path) in str(exc_info.value)
    
@pytest.mark.parametrize(
    "content",
    [
        "",
        "[1, 2, 3]",
        "16",
        "true",
        "null",
    ],
)
def test_load_yaml_rejects_non_mapping_root(tmp_path, content):
    path = tmp_path / "wrong_root.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        load_yaml(str(path))

    assert "YAML 顶层必须是映射" in str(exc_info.value)
    
@pytest.mark.parametrize(
    "section, key",
    [
        (None, "shape"),
        ("shape", "M"),
        ("addresses", "input_base"),
    ],
)
def test_load_problem_rejects_missing_field(
    problem_data,
    tmp_path,
    section,
    key,
):
    if section is None:
        del problem_data[key]
        expected_path = f"problem.{key}"
    else:
        del problem_data[section][key]
        expected_path = f"problem.{section}.{key}"

    path = write_problem(tmp_path, problem_data)

    with pytest.raises(ValueError) as exc_info:
        load_problem(path)

    message = str(exc_info.value)
    assert expected_path in message
    assert "缺少必需字段" in message
    
@pytest.mark.parametrize(
    "section, key",
    [
        (None, "array"),
        ("buffers", "double_buffer_input"),
        ("candidates", "tile_k"),
    ],
)
def test_load_hardware_rejects_missing_field(
    hardware_data,
    tmp_path,
    section,
    key,
):
    if section is None:
        del hardware_data[key]
        expected_path = f"hardware.{key}"
    else:
        del hardware_data[section][key]
        expected_path = f"hardware.{section}.{key}"
        
    path = write_hardware(tmp_path, hardware_data)
    
    with pytest.raises(ValueError) as exc_info:
        load_hardware(path)
        
    message = str(exc_info.value)
    assert expected_path in message
    assert "缺少必需字段" in message