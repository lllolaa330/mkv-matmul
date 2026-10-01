from collections.abc import Iterator

from .types import HardwareSpec, Schedule

def enumerate_schedules(hw: HardwareSpec) -> Iterator[Schedule]:
    """ 按配置顺序枚举全部分块组合,不进行合法性过滤 """
    for tile_m in hw.tile_m_candidates:
        for tile_n in hw.tile_n_candidates:
            for tile_k in hw.tile_k_candidates:
                yield Schedule(tile_m=tile_m, tile_n=tile_n, tile_k=tile_k)
                