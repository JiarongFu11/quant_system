import numpy as np
from abc import ABC, abstractmethod

class IFactorGenerator:
    @abstractmethod
    def generate_factors(self):
        ...

class Function(ABC):
    """所有算子的抽象基类。"""
    def __init__(self, name: str, arity: int):
        self.name = name
        self.arity = arity
    
    @abstractmethod
    def __call__(self, *args: np.ndarray) -> np.ndarray:
        raise NotImplementedError
        
    def __repr__(self): return f"Func:{self.name}(arity={self.arity})"
    def __str__(self): return self.name


