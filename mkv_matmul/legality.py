from .types import (
    BufferUsage, 
    HardwareSpec,
    MatMulProblem,
    Schedule,
)
from .descriptor import(
    BASE_BITS,
    DIM_BITS,
    STRIDE_BITS,
    TILE_BITS,
    fits_unsigned,
)

def calculate_buffer_usage(
    hw: HardwareSpec,
    schedule: Schedule,
) -> BufferUsage:
    """ 按完整 tile 计算物理 buffer 需求 """
    input_copies = 2 if hw.double_buffer_input else 1
    weight_copies = 2 if hw.double_buffer_weight else 1
    
    input_bytes = (
        input_copies
        * schedule.tile_m
        * schedule.tile_k
        * hw.input_element_bytes
    )
    
    weight_bytes = (
        weight_copies
        * schedule.tile_n
        * schedule.tile_k
        * hw.weight_element_bytes
    )
    
    output_bytes = (
        schedule.tile_m
        * schedule.tile_n
        * hw.accumulator_element_bytes
    )
    
    return BufferUsage(
        input_bytes=input_bytes,
        weight_bytes=weight_bytes,
        output_bytes=output_bytes,   
    )
    
def check_schedule_legality(
    hw: HardwareSpec,
    schedule: Schedule
) -> tuple[bool, str]:
    """ 检查分块参数与 buffer 容量, 返回第一个失败原因 """
    if schedule.loop_order != "MNK" :
        return False, "UNSUPPORTED_LOOP_ORDER"
      
    tile_options = (
        (schedule.tile_m, hw.tile_m_candidates),
        (schedule.tile_n, hw.tile_n_candidates),
        (schedule.tile_k, hw.tile_k_candidates),
    )
    
    for value, supported_values in tile_options:
        if type(value) is not int or value <= 0:
            return False, "INVALID_TILE"
          
        if value not in supported_values:
            return False, "UNSUPPORTED_TILE"
          
    usage = calculate_buffer_usage(hw, schedule)
    
    if usage.input_bytes > hw.input_buffer_bytes:
        return False, "INPUT_BUFFER_OVERFLOW"
    
    if usage.weight_bytes > hw.weight_buffer_bytes:
        return False, "WEIGHT_BUFFER_OVERFLOW"
      
    if usage.output_bytes > hw.output_buffer_bytes:
        return False, "OUTPUT_BUFFER_OVERFLOW"
    
    return True, ""
  
def check_problem_layout(
    problem: MatMulProblem,
    hw: HardwareSpec,
) -> tuple[bool, str]:
    """ 检查非负基地址、基地址对齐与行步长 """
    addresses = (
        (problem.input_base, "UNALIGNED_INPUT_BASE"),
        (problem.weight_base, "UNALIGNED_WEIGHT_BASE"),
        (problem.output_base, "UNALIGNED_OUTPUT_BASE"),
    )
    
    for base, alignment_reason in addresses:
        if type(base) is not int or base < 0:
            return False, "INVALID_BASE_ADDRESS"
        
        if base % hw.alignment_bytes != 0:
            return False, alignment_reason
    
    strides = (
        (problem.input_stride, problem.K),
        (problem.weight_stride, problem.N),
        (problem.output_stride, problem.N),
    )
    
    for stride, minimum_width in strides:
        if type(stride) is not int:
            return False, "INVALID_STRIDE"
        if stride < minimum_width:
            return False, "INVALID_STRIDE"
      
    return True, ""
    
def check_descriptor_fields(
    problem: MatMulProblem,
    schedule: Schedule,
) -> tuple[bool, str]:
    """ 检查变量字段能否编码进描述符 """
    fields = (
        ("M", problem.M, DIM_BITS),
        ("N", problem.N, DIM_BITS),
        ("K", problem.K, DIM_BITS),
        
        ("tile_m", schedule.tile_m, TILE_BITS),
        ("tile_n", schedule.tile_n, TILE_BITS),
        ("tile_k", schedule.tile_k, TILE_BITS),
        
        ("input_base", problem.input_base, BASE_BITS),
        ("weight_base", problem.weight_base, BASE_BITS),
        ("output_base", problem.output_base, BASE_BITS),
        
        ("input_stride", problem.input_stride, STRIDE_BITS),
        ("weight_stride", problem.weight_stride, STRIDE_BITS),
        ("output_stride", problem.output_stride, STRIDE_BITS),
    )
    
    for name, value, bits in fields:
        if not fits_unsigned(value, bits):
            return False, f"DESCRIPTOR_FIELD_OUT_OF_RANGE:{name}"
    
    return True, ""
  
def check_legality(
    problem: MatMulProblem,
    hw: HardwareSpec,
    schedule: Schedule,
) -> tuple[bool, str]:
    """ 组合问题布局、分块、容量检查和描述符字段检查 """
    legal, reason = check_problem_layout(problem, hw)
    if not legal:
        return False, reason
    
    legal, reason = check_schedule_legality(hw, schedule)
    if not legal:
        return False, reason
      
    return check_descriptor_fields(problem, schedule)

  
