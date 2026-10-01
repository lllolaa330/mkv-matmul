from dataclasses import replace
from pathlib import Path

from mkv_matmul.candidates import enumerate_schedules
from mkv_matmul.config import load_hardware
from mkv_matmul.types import Schedule

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def load_example_hardware():
    path = PROJECT_ROOT / "examples" / "zcu104_16x16.yaml"
    return load_hardware(str(path))
  
def test_enumerate_default_candidates():
    hw = load_example_hardware()
    schedules = list(enumerate_schedules(hw))
    
    assert len(schedules) == 48
    assert len(set(schedules)) == 48

    assert schedules[0] == Schedule(1, 16, 32)
    assert schedules[-1] == Schedule(16, 64, 256)

    assert all(s.loop_order == "MNK" for s in schedules)

def test_enumerate_custom_candidates_in_order():
    original_hw = load_example_hardware()
    hw = replace(
        original_hw,
        tile_m_candidates=(4, 1),
        tile_n_candidates=(32, 16),
        tile_k_candidates=(64, 32),
    )

    schedules = list(enumerate_schedules(hw))
    
    expected = [Schedule(tile_m=4, tile_n=32, tile_k=64, loop_order='MNK'),
                Schedule(tile_m=4, tile_n=32, tile_k=32, loop_order='MNK'),
                Schedule(tile_m=4, tile_n=16, tile_k=64, loop_order='MNK'),
                Schedule(tile_m=4, tile_n=16, tile_k=32, loop_order='MNK'),
                Schedule(tile_m=1, tile_n=32, tile_k=64, loop_order='MNK'),
                Schedule(tile_m=1, tile_n=32, tile_k=32, loop_order='MNK'),
                Schedule(tile_m=1, tile_n=16, tile_k=64, loop_order='MNK'),
                Schedule(tile_m=1, tile_n=16, tile_k=32, loop_order='MNK')
                ]
    
    assert schedules == expected
    
    assert original_hw.tile_m_candidates == (1, 4, 8, 16)
    