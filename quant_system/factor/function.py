import numpy as np
from abc import ABC, abstractmethod
import os
import sys
from functools import reduce
from numba import njit

# 确保能找到accelerators.py (根据您的项目结构调整)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
# 假设 accelerators.py 在 factor/ 目录下
from factor.accelerators import njit_rolling
from factor.base import Function 

class TemporalFunction(Function):
    """时序函数的基类，需要一个回看周期。"""
    def __init__(self, name: str):
        # 名字中自动加入周期，方便识别，如 ts_mean_10
        super().__init__(f"{name}", 1) # arity 总是 1
    
# -----------------------------------------------
# 核心滚动计算逻辑 (用njit装饰)
# -----------------------------------------------

def _core_max(window: np.ndarray) -> np.ndarray: 
    return np.max(window)


def _core_min(window: np.ndarray) -> np.ndarray: 
    return np.min(window)

def _core_sum(window: np.ndarray) -> np.ndarray: 
    return np.sum(window)

def _core_mean(window: np.ndarray) -> np.ndarray: 
    return np.mean(window)

def _core_std(window: np.ndarray) -> np.ndarray: 
    return np.std(window)

def _core_delta(window: np.ndarray) -> np.ndarray: 
    return window[-1] - window[0]

def _core_decay_linear(window: np.ndarray) -> np.ndarray:
    n = len(window)
    weights = np.arange(1.0, n + 1.0)
    total_weight = np.sum(weights)
    
    # 确保窗口数据是浮点数类型
    window_float = window.astype(np.float64)
    
    return np.dot(window_float, weights) / total_weight

# -----------------------------------------------
# 使用njit_rolling装饰器包装核心逻辑
# -----------------------------------------------
ts_max = njit_rolling(_core_max)
ts_min = njit_rolling(_core_min)
ts_sum = njit_rolling(_core_sum)
ts_mean = njit_rolling(_core_mean)
ts_std = njit_rolling(_core_std)
ts_delta = njit_rolling(_core_delta)
ts_decay_linear = njit_rolling(_core_decay_linear)


# -----------------------------------------------------------
# 1. 时序聚合函数类 (纯Numpy版本)
# -----------------------------------------------------------
class TemporalMax(TemporalFunction):
    def __init__(self): super().__init__("ts_max")
    def __call__(self, lookback_period: int, x: np.ndarray) -> np.ndarray: return ts_max(x, lookback_period)

class TemporalMin(TemporalFunction):
    def __init__(self): super().__init__("ts_min")
    def __call__(self, lookback_period: int, x: np.ndarray) -> np.ndarray: return ts_min(x, lookback_period)

class TemporalSum(TemporalFunction):
    def __init__(self): super().__init__("ts_sum")
    def __call__(self, lookback_period: int, x: np.ndarray) -> np.ndarray: return ts_sum(x, lookback_period)

class TemporalMean(TemporalFunction):
    def __init__(self): super().__init__("ts_mean")
    def __call__(self, lookback_period: int, x: np.ndarray) -> np.ndarray: return ts_mean(x, lookback_period)

class TemporalStd(TemporalFunction):
    def __init__(self): super().__init__("ts_std")
    def __call__(self, lookback_period: int, x: np.ndarray) -> np.ndarray: return ts_std(x, lookback_period)

class TemporalDelta(TemporalFunction):
    def __init__(self): super().__init__("ts_delta")
    def __call__(self, lookback_period: int, x: np.ndarray) -> np.ndarray: return ts_delta(x, lookback_period)

class TemporalDecayLinear(TemporalFunction):
    def __init__(self): super().__init__("ts_decay_linear")
    def __call__(self, lookback_period: int, x: np.ndarray) -> np.ndarray: return ts_decay_linear(x, lookback_period)

# -----------------------------------------------------------
# 2. 非时序一元函数
# -----------------------------------------------------------
class Abs(Function):
    def __init__(self): super().__init__("abs", 1)
    def __call__(self, x: np.ndarray) -> np.ndarray: return np.abs(x)

class Negative(Function):
    def __init__(self): super().__init__("negative", 1)
    def __call__(self, x: np.ndarray) -> np.ndarray: return -x

class Sign(Function):
    def __init__(self): super().__init__("sign", 1)
    def __call__(self, x: np.ndarray) -> np.ndarray: return np.sign(x)

class Sqrt(Function):
    def __init__(self): super().__init__("sqrt", 1)
    def __call__(self, x: np.ndarray) -> np.ndarray:
        with np.errstate(invalid='ignore'): return np.sqrt(np.abs(x))

class Square(Function):
    def __init__(self): super().__init__("square", 1)
    def __call__(self, x: np.ndarray) -> np.ndarray: return np.square(x) # type: ignore

class Cube(Function):
    def __init__(self): super().__init__("cube", 1)
    def __call__(self, x: np.ndarray) -> np.ndarray: return np.power(x, 3)

class Cbrt(Function):
    def __init__(self): super().__init__("cbrt", 1)
    def __call__(self, x: np.ndarray) -> np.ndarray: return np.cbrt(x)

class Log(Function):
    def __init__(self): super().__init__("log", 1)
    def __call__(self, x: np.ndarray) -> np.ndarray:
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.log(np.abs(x) + 1e-9)

class Inv(Function):
    def __init__(self): super().__init__("inv", 1)
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (x + 1e-9)

# -----------------------------------------------------------
# 3. 多元函数
# -----------------------------------------------------------
class Add(Function):
    def __init__(self, arity: int): super().__init__("add", arity)
    def __call__(self, *args: np.ndarray) -> np.ndarray:
        if len(args) != self.arity:
            raise ValueError(f"Add (arity={self.arity}) expected {self.arity} arguments, got {len(args)}")
        return reduce(np.add, args)

class Subtract(Function):
    def __init__(self, arity: int): super().__init__("subtract", arity)
    def __call__(self, *args: np.ndarray) -> np.ndarray:
        if len(args) != self.arity:
            raise ValueError(f"Subtract (arity={self.arity}) expected {self.arity} arguments, got {len(args)}")
        return reduce(np.subtract, args)

class Multiply(Function):
    def __init__(self, arity: int): super().__init__("multiply", arity)
    def __call__(self, *args: np.ndarray) -> np.ndarray:
        if len(args) != self.arity:
            raise ValueError(f"Multiply (arity={self.arity}) expected {self.arity} arguments, got {len(args)}")
        return reduce(np.multiply, args)

class Divide(Function):
    def __init__(self, arity: int): super().__init__("divide", arity)
    def __call__(self, *args: np.ndarray) -> np.ndarray:
        if len(args) != self.arity:
            raise ValueError(f"Divide (arity={self.arity}) expected {self.arity} arguments, got {len(args)}")
        
        def safe_divide(x, y):
            # 仅在y上加极小值，避免影响x
            return x / (y + 1e-9)
        return reduce(safe_divide, args)

# ----------------------------------------------------
# Collection of all function instances for genetic algo
# ----------------------------------------------------

def get_temporal_functions() -> list:
    """根据给定的周期列表，动态创建所有时序函数实例。"""
    return [
                TemporalMax(),
                TemporalMin(),
                TemporalSum(),
                TemporalMean(),
                TemporalStd(),
                TemporalDelta(),
                TemporalDecayLinear(),
            ]

def get_unary_functions() -> list:
    """获取所有非时序的一元函数实例。"""
    return [
        Abs(), Negative(), Sign(), Sqrt(), Square(), Cube(), Cbrt(), Log(), Inv(),
    ]

def get_binary_functions() -> list:
    """获取所有二元函数实例。"""
    return [
        Add(2), Subtract(2), Multiply(2), Divide(2),
    ]

def get_ternary_functions() -> list:
    """获取所有三元函数实例。"""
    return [
        Add(3), Subtract(3), Multiply(3), Divide(3),
    ]

def get_quaternary_functions() -> list:
    """获取所有四元函数实例。"""
    return [
        Add(4), Subtract(4), Multiply(4), Divide(4),
    ]

def build_function_set() -> dict:
    """
    构建一个包含所有函数类型的字典，方便遗传算法使用。

    Args:
        lookback_periods (list, optional):
            用于生成时序函数的回看周期列表。Defaults to None.

    Returns:
        dict: 一个字典，键为 'unary', 'binary', 'ternary', 'quaternary'，
              值为对应的函数实例列表。
    """
        
    unary_functions = get_unary_functions() + get_temporal_functions()
    
    function_set = {
        'unary': unary_functions,
        'binary': get_binary_functions(),
        'ternary': get_ternary_functions(),
        'quaternary': get_quaternary_functions(),
    }
    
    # 将所有函数合并到一个列表中，方便随机选择
    all_funcs = []
    for funcs in function_set.values():
        all_funcs.extend(funcs)
    function_set['all'] = all_funcs
    
    return function_set
