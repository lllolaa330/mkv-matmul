from pathlib import Path
from .types import MatMulProblem, HardwareSpec

import yaml

def load_yaml(path: str) -> dict:
    """ 读取 YAML, 并要求文件顶层是一个映射 """
    config_path = Path(path) # 路径字符串 -> 路径对象
    
    with config_path.open("r", encoding="utf-8") as file:
        try:
            data = yaml.safe_load(file) # 将 yaml 解析成字典、列表、整数等基础对象
        except yaml.YAMLError as error:
            raise ValueError(f"{config_path}: YAML 语法错误") from error
        
    if not isinstance(data, dict):
        raise ValueError(
            f"{config_path}: YAML 顶层必须是映射"
        )
        
    return data
  
def require_mapping(data: dict, key: str, location: str) -> dict:
    """ 取得必需的字典字段 否则报错 """
    field_path = f"{location}.{key}"
    
    if key not in data:
        raise ValueError(f"{field_path}: 缺少必需字段")
      
    value = data[key]
    
    if not isinstance(value, dict):
        raise ValueError(f"{field_path}: 必须是映射")
      
    return value
  
def require_int(data: dict,
                key: str,
                location: str,
                minimum: int,
                ) -> int:
    """ 取得必需的整数字段 并检查最小值 """
    field_path = f"{location}.{key}"
    
    if key not in data:
        raise ValueError(f"{field_path}: 缺少必需字段")
      
    value = data[key]
    
    if type(value) is not int:
        raise ValueError(f"{field_path}: 必须是整数 不能是布尔值")
      
    if value < minimum:
        raise ValueError(f"{field_path}: 必须大于等于 {minimum}")
    
    return value
  
def require_string(data: dict, key: str, location: str) -> str:
    """ 取得必需的非空字符串 """
    field_path = f"{location}.{key}"
    
    if key not in data:
        raise ValueError(f"{field_path}: 缺少必需字段")
      
    value = data[key]
    
    if not isinstance(value, str):
        raise ValueError(f"{field_path}: 必须是字符串")
      
    if not value.strip():
        raise ValueError(f"{field_path}: 不能为空或只包含空白")
      
    return value
  
def require_bool(data: dict, key: str, location: str) -> bool:
    """ 取得必需的 bool 字段 """
    field_path = f"{location}.{key}"
    
    if key not in data:
            raise ValueError(f"{field_path}: 缺少必需字段")
          
    value = data[key]
    
    if type(value) is not bool:
        raise ValueError(f"{field_path}: 必须是布尔值")
    
    return value
    
def require_int_tuple(data: dict, key: str, location: str) -> tuple[int, ...]:
    """ 读取非空、无重复的正整数列表 list, 并转换为元组 tuple """
    field_path = f"{location}.{key}"
        
    if key not in data:
                raise ValueError(f"{field_path}: 缺少必需字段")
              
    values = data[key]
    
    if not isinstance(values, list):
        raise ValueError(f"{field_path}: 必须是 list")
      
    if not values:
        raise ValueError(f"{field_path}: 列表不能为空")
    
    for index, value in enumerate(values):
        if type(value) is not int:
            raise ValueError(f"{field_path}[{index}]: 每个元素必须是严格的整数")
        
        if value <= 0:
            raise ValueError(f"{field_path}[{index}]: 每个元素必须大于零")
    
    if len(set(values)) != len(values):
        raise ValueError(f"{field_path}: 列表不能包含重复值")
    
    return tuple(values)
    
  
def load_problem(path: str) -> MatMulProblem:
    """ 读取并验证矩阵乘法问题配置 """
    data = load_yaml(path)
    
    name = require_string(data, "name", "problem")
    operator = require_string(data, "operator", "problem")
    
    if operator != "matmul":
        raise ValueError("problem.operator: 当前只支持matmul")
      
    dtype = require_mapping(data, "dtype", "problem")
    
    expected_dtypes = {
        "input" : "int8",
        "weight" : "int8",
        "accumulator" : "int32",
    }
    
    for role, expected in expected_dtypes.items():
        actual = require_string(dtype, role, "problem.dtype")
        
        if actual != expected:
            raise ValueError(f"problem.dtype.{role}: 当前只支持{expected}")
        
    shape = require_mapping(data, "shape", "problem")
    
    M = require_int(shape, "M", "problem.shape", minimum=1)
    N = require_int(shape, "N", "problem.shape", minimum=1)
    K = require_int(shape, "K", "problem.shape", minimum=1)
    
    addresses = require_mapping(data, "addresses", "problem")
    strides = require_mapping(data, "strides", "problem")
    
    return MatMulProblem(
      name = name,
      M=M, N=N, K=K,
      input_base = require_int(addresses, "input_base", "problem.addresses", minimum=0),
      weight_base = require_int(addresses, "weight_base", "problem.addresses", minimum=0),
      output_base = require_int(addresses, "output_base", "problem.addresses", minimum=0),
      input_stride = require_int(strides, "input", "problem.strides", minimum=K),
      weight_stride = require_int(strides, "weight", "problem.strides", minimum=N),
      output_stride = require_int(strides, "output", "problem.strides", minimum=N),
    )
    
def load_hardware(path: str) -> HardwareSpec:
    """ 读取并验证硬件配置 """
    config = load_yaml(path)
    name = require_string(config, "name", "hardware")
    array = require_mapping(config, "array", "hardware")
    element_data = require_mapping(config, "data", "hardware")
    buffers = require_mapping(config, "buffers", "hardware")
    dma = require_mapping(config, "dma", "hardware")
    candidates = require_mapping(config, "candidates", "hardware")
    
    expected_element_sizes = {
        "input_bytes": 1,
        "weight_bytes": 1,
        "accumulator_bytes": 4,
    }
    element_sizes = {}
    for key, expected in expected_element_sizes.items():
        actual = require_int(element_data, key, "hardware.data", minimum=1)
        if actual != expected:
            raise ValueError(f"hardware.data.{key}: 当前只支持 {expected} 字节")
        element_sizes[key] = actual  
        
    return HardwareSpec(
        name=name,
        array_rows=require_int(array, "rows", "hardware.array", minimum=1),
        array_cols=require_int(array, "cols", "hardware.array", minimum=1),
        pipeline_cycles=require_int(array, "pipeline_cycles", "hardware.array", minimum=0),
        
        input_element_bytes=element_sizes["input_bytes"],
        weight_element_bytes=element_sizes["weight_bytes"],
        accumulator_element_bytes=element_sizes["accumulator_bytes"],
        
        input_buffer_bytes=require_int(buffers, "input_bytes", "hardware.buffers", minimum=1),
        weight_buffer_bytes=require_int(buffers, "weight_bytes", "hardware.buffers", minimum=1),
        output_buffer_bytes=require_int(buffers, "output_bytes", "hardware.buffers", minimum=1),
        double_buffer_input=require_bool(buffers, "double_buffer_input", "hardware.buffers"),
        double_buffer_weight=require_bool(buffers, "double_buffer_weight", "hardware.buffers"),

        dma_bytes_per_cycle=require_int(dma, "bytes_per_cycle", "hardware.dma", minimum=1),
        dma_setup_cycles=require_int(dma, "setup_cycles", "hardware.dma", minimum=0),
        alignment_bytes=require_int(dma, "alignment_bytes", "hardware.dma", minimum=1),

        tile_m_candidates=require_int_tuple(candidates, "tile_m", "hardware.candidates"),
        tile_n_candidates=require_int_tuple(candidates, "tile_n", "hardware.candidates"),
        tile_k_candidates =require_int_tuple(candidates, "tile_k", "hardware.candidates"),
        
    )