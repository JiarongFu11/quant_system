import numpy as np
from numba import njit
from functools import wraps
from typing import Callable, TypeAlias

# 定义核心滚动函数的类型提示
CoreRollingFunc: TypeAlias = Callable[[np.ndarray], np.ndarray]

def njit_rolling(core_func: CoreRollingFunc) -> Callable[[np.ndarray, int], np.ndarray]:
    """
    一个纯Numpy的装饰器，将一个处理小窗口的Numba核心函数，
    转换成一个能在整个2D Numpy数组上高效滚动的函数。

    Args:
        core_func: 一个用 @njit 装饰的函数，接收 (window, n_features) 的
                   numpy数组，返回 (n_features,) 的结果数组。

    Returns:
        一个新函数，接收一个2D Numpy数组和window大小，返回滚动计算后的2D Numpy数组。
    """
    
    # 确保核心函数已经被Numba编译
    if not hasattr(core_func, 'py_func'):
        core_func = njit(core_func)
    
    # 这个主循环是整个机制的核心，必须被Numba JIT编译
    @njit
    def _rolling_loop(data_array: np.ndarray, window: int) -> np.ndarray:
        n_rows = data_array.shape[0]
        # 使用np.nan初始化，这是浮点数的标准做法
        output_array = np.full((n_rows), fill_value=np.nan, dtype=np.float64)

        # 从第一个可以形成完整窗口的位置开始循环
        for i in range(window - 1, n_rows):
            start_idx = i - window + 1
            end_idx = i + 1
            current_window_slice = data_array[start_idx:end_idx]
            output_array[i] = core_func(current_window_slice)
        return output_array

    # 这是最终返回给用户的、可以直接使用的函数
    @wraps(core_func)
    def rolling_applier(data_array: np.ndarray, window: int) -> np.ndarray:
        if not isinstance(data_array, np.ndarray) or data_array.ndim != 1:
            raise TypeError("Input must be a 1D numpy array.")
        
        if window <= 1:
            # 窗口大小为1或更小时，滚动没有意义
            output_array = data_array.astype(np.float64, copy=True)
            output_array[:window-1] = np.nan
            return output_array
        if window > data_array.shape[0]:
            raise ValueError("Window size cannot be larger than the number of rows.")

        # 直接调用Numba加速的循环函数进行计算
        return _rolling_loop(data_array, window)

    return rolling_applier