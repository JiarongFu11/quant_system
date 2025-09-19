import pandas as pd
import numpy as np
import random
import re
import os
import sys
# 添加项目根目录到 PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from factor.base import Function
from factor.function import TemporalFunction
from typing import List, Union

def is_number(s):
    pattern = r'^[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$'
    return re.fullmatch(pattern, s.strip()) is not None

class FactorCalculator:
     def calculate_factors(self, factors: List[List[Function | str]], data: pd.DataFrame) -> pd.DataFrame:
        """
        计算多个因子
        """
        factor_dataframe = pd.DataFrame()
        for factor_nodes in factors:
            # --- 这是被修复的行 ---
            # 错误：factor_value = self.calculate_factor(factor, data)
            # 正确：
            factor = ExpressionFactor(factor_nodes)
            factor_value = factor.calculate_factor_value(data)
            # --- 修复结束 ---
            factor_dataframe = pd.concat([factor_dataframe, factor_value], axis=1)
        
        return factor_dataframe

class ExpressionFactor():
    DEFAULT_LOOKBACK_PERIODS: int = 20
    def __init__(self, nodes: List[Union[Function, str]]):
        if not isinstance(nodes, list):
            raise ValueError("nodes must be a list")
        
        self.nodes = nodes
    
    def __len__(self):
        return len(self.nodes)
    
    def __repr__(self):
        node_strs = [node 
                     if isinstance(node, str)
                     else node.name
                     for node in self.nodes]
        
        return f"ExpressionTree({', '.join(node_strs)})"
    
    @classmethod
    def generate_random_factor(cls, 
                               functions: List[Function],
                               basic_factors: List[str], 
                               max_depth: int,
                               lookback_period: int = None): 
        """
        从表达式树中随机提取子树
        
        返回三元组: (左子树, 子树, 右子树)
        """
        if lookback_period is None:
            lookback_period = cls.DEFAULT_LOOKBACK_PERIODS 
        aritys = []
        factor = []
        single_function = random.choice(functions)####
        factor.append(single_function)
        if isinstance(single_function, TemporalFunction):
            period = str(random.randint(2, lookback_period))
            factor.append(period)
        aritys.append(single_function.arity)

        '通过均匀分布选择正态分布的均值和sigma，以此决定树的复杂度'
        value = random.uniform(0, 1)
        if value < 0.33:
            mu = 0.35
            sigma = 0.1
        elif value > 0.33 and value < 0.66:
            mu = 0.5
            sigma = 0.1
        elif value > 0.66:
            mu = 0.65
            sigma = 0.1

        depth = 1

        while aritys:
            pro = random.gauss(mu, sigma)
            
            if pro < 0.5 and depth < max_depth:
                single_function = random.choice(functions)
                factor.append(single_function)
                if isinstance(single_function, TemporalFunction):
                    period = str(random.randint(2, lookback_period))
                    factor.append(period)
                aritys[-1] -= 1
                if aritys[-1] == 0:
                    aritys.pop()
                aritys.append(single_function.arity)
                '计算depth' 
                if len(aritys) > 1:
                    depth += 1
                elif len(aritys) == 1:
                    '左边节点构建完重置深度'
                    depth = 1

            elif pro > 0.5:
                basic_factor = random.choice(basic_factors)
                factor.append(basic_factor)
                aritys[-1] -= 1

                if aritys[-1] == 0:
                    aritys.pop() 
                
                if len(aritys) == 1:
                    depth = 1
        return cls(factor)

    def get_subfactor(self):
        """获取子树"""
        # 寻找一个有效的子树起点（必须是函数）
        possible_start_indices = [i for i, node in enumerate(self.nodes) if isinstance(node, Function)]
        if not possible_start_indices:
            return None # 如果树中没有函数，则无法提取子树

        start_index = random.choice(possible_start_indices)
        start_child_index = None
        end_child_index = None
        
        aritys = []
        # 从选定的起点开始扫
        for node_index in range(start_index, len(self.nodes)):
            node = self.nodes[node_index]
            if isinstance(node, Function):
                if not aritys and start_child_index is None: 
                    pro = random.uniform(0, 1)
                    '随机选择子树开始节点'
                    if pro > 0.5:
                        aritys.append(node.arity)
                        start_child_index = node_index
                        if isinstance(node, TemporalFunction):
                            aritys[-1] += 1
                elif start_child_index:
                    '顺着子树迭代'
                    aritys[-1] -= 1
                    if aritys[-1] == 0:
                        aritys.pop()
                    aritys.append(node.arity)
                    if isinstance(node, TemporalFunction):
                        aritys[-1] += 1
            else:
                if start_child_index:
                    '顺着子树迭代寻找结束节点'
                    aritys[-1] -= 1
                    if aritys[-1] == 0:
                        aritys.pop()
                    if len(aritys) == 0:
                        end_child_index = node_index
                        left_part = self.nodes[:start_child_index]
                        subfactor = self.nodes[start_child_index:end_child_index + 1]
                        right_part = self.nodes[end_child_index + 1:]
                        return left_part, subfactor, right_part
                    
        return []
                    
    def mutate(self, 
               mutation_prob: float, 
               all_functions: List[Function],
               basic_factors: List[str]) -> 'ExpressionFactor':
        """
        对树中的一个随机节点进行突变，并返回一个新的 ExpressionFactor 实例。
        对应原 GeneAlgo._mutate_one_tree 的部分逻辑。
        """
        new_nodes = self.nodes[:] # 创建副本以保证不变性
        if not new_nodes:
            return ExpressionFactor(new_nodes)

        # 随机选择一个节点进行突变
        mutation_index = random.randint(0, len(new_nodes) - 1)
        node_to_mutate = new_nodes[mutation_index]

        if random.random() < mutation_prob:
            if isinstance(node_to_mutate, Function):
                # 如果是函数，用另一个相同元数(arity)的函数替换
                if not isinstance(node_to_mutate, TemporalFunction):
                    same_arity_functions = [f 
                                            for f in all_functions 
                                            if (f.arity == node_to_mutate.arity and not isinstance(f, TemporalFunction)) 
                                            and f != node_to_mutate]
                    new_nodes[mutation_index] = random.choice(same_arity_functions)
                elif isinstance(node_to_mutate, TemporalFunction):
                    same_arity_functions = [f 
                                            for f in all_functions 
                                            if (f.arity == node_to_mutate.arity and isinstance(f, TemporalFunction)) 
                                            and f != node_to_mutate]
                    new_nodes[mutation_index] = random.choice(same_arity_functions)
            elif node_to_mutate in basic_factors:
                # 如果是基础因子，用另一个基础因子替换
                other_factors = [f for f in basic_factors if f != node_to_mutate]
                if other_factors:
                    new_nodes[mutation_index] = random.choice(other_factors)
            elif isinstance(node_to_mutate, str):
                if is_number(node_to_mutate):
                    new_nodes[mutation_index] = str(random.randint(2, 20))
        
        return ExpressionFactor(new_nodes)
         
    def calculate_factor_value(self, data: pd.DataFrame) -> pd.Series:
        """
        计算单个因子
        :param data: 数据
        :return: 因子结果
        """
        stack = []
        for node in reversed(self.nodes):
            if isinstance(node, str):
                if is_number(node):
                    # 将数字转换为浮点数
                    stack.append(float(node))
                else:
                    # 确保取出的是一维数组
                    stack.append(data[node].values.astype(np.float64))
            elif isinstance(node, Function):
                if isinstance(node, TemporalFunction):
                    if len(stack) < node.arity + 1:
                        raise ValueError(f"Not enough arguments for {node.name}")
                    
                    # 获取回看周期参数
                    period = stack.pop()
                    # 获取数据参数
                    args = [stack.pop() for _ in range(node.arity)]
                    result = node(period, *args)
                else:
                    # 普通函数
                    if len(stack) < node.arity:
                        raise ValueError(f"Not enough arguments for {node.name}")
                    
                    args = [stack.pop() for _ in range(node.arity)]
                    result = node(*args)
                # 确保结果是一维数组
                if isinstance(result, np.ndarray) and result.ndim > 1:
                    result = result.flatten()
                stack.append(result)
        
        if len(stack) == 1:
            result = pd.Series(stack[0].flatten())
            if isinstance(result, pd.Series) and result.name is None:
                # 尝试生成一个名字
                 result.name = repr(self)
            return result
        else:
            # 提供更多调试信息
            print(f"Expression evaluation failed. Stack has {len(stack)} items:")
            for i, item in enumerate(stack):
                if isinstance(item, np.ndarray):
                    print(f"  Item {i}: array of shape {item.shape}")
                else:
                    print(f"  Item {i}: {type(item)} - {item}")
            raise ValueError("Expression evaluation failed, stack has multiple items.")

if __name__ == "__main__":
    from factor.function import TemporalMean, TemporalStd, Abs, Sqrt, Add, Divide
    from factor.factor_unit import ExpressionFactor

        # 构造表达式树节点
    nodes = [
        Sqrt(),                              # sqrt(...)
        Abs(),                              # abs(...)
        Add(2), 'close',                        # ... + 3.14
        Divide(2),                           # ts_mean(...) / ts_std(...)
        TemporalStd(), '3', "volume",        # ts_std(volume, 5)
        TemporalMean(), '3',"close"        # ts_mean(close, 10)
    ]

    factor = ExpressionFactor(nodes)
    import pandas as pd
    import numpy as np

    # 构造模拟数据
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100)
    data = pd.DataFrame({
        "close": np.random.randn(100).cumsum() + 100,
        "volume": np.random.randint(1e6, 1e7, 100),
    }, index=dates)

    result = factor.calculate_factor_value(data)
    print(result.tail())
    print(data.tail())
    data.tail().to_csv('/Users/fu/Desktop/test.csv')
    
    nodes = [
        Divide(2),                           # ts_mean(...) / ts_std(...)
        TemporalStd(), '3', "volume",        # ts_std(volume, 5)
        TemporalMean(), '3',"close"        # ts_mean(close, 10)
    ]
    factor = ExpressionFactor(nodes)
    result = factor.calculate_factor_value(data)
    print(result.tail())