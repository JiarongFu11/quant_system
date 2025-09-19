import os
import sys
import re
import random
import numpy as np
import pandas as pd

from itertools import combinations
from abc import abstractmethod
# 添加项目根目录到 PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from factor.base import IFactorGenerator
from factor.factor_unit import ExpressionFactor
from factor.function import build_function_set
from tqdm import tqdm
from typing import List

class IFitnessCalculator:
    @abstractmethod
    def calculate_fitness(self) -> float:
        pass

    @abstractmethod
    def select_by_fitness(self, fitness_list: List[float]) -> List:
        """
        input: fitness_list: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] 
        output: reserved_list: [False, True], True 表示保留 False 表示不保留   
        """
        pass

class ICorrelationCalculator:
    def calculate_factor_correlation(self, factor1, factor2):
        """
        抽象方法：用于计算两个因子之间的相关性
        子类可以自定义相关性计算方式，如皮尔逊、斯皮尔曼等
        返回值应为 [0, 1] 的相关系数
        """
        pass

class GeneAlgo(IFactorGenerator):
    DEFAULT_POPULATION_SIZE = 100000
    DEFAULT_MAX_DEPTH = 10
    DEFAULT_CROSS_RATE = 0.5
    DEFAULT_MUTATION_RATE = 0.2
    DEFAULT_ROUNDS = 10
    DEFAULT_CORR_THRESHOLD = 0.8

    def __init__(self, 
                 data: pd.DataFrame = None,
                 correlation_calculator: ICorrelationCalculator = None,
                 FitnessCalculator: IFitnessCalculator = None,
                 population_size=DEFAULT_POPULATION_SIZE, 
                 basic_factors: List[str] = None, 
                 max_depth=DEFAULT_MAX_DEPTH,
                 crossover_prob=DEFAULT_CROSS_RATE,
                 mutation_prob=DEFAULT_MUTATION_RATE,
                 rounds=DEFAULT_ROUNDS,
                 corr_threshold = DEFAULT_CORR_THRESHOLD):
        
        self.data = data
        self.correlation_calculator = correlation_calculator
        self.fitness_calculator = FitnessCalculator
        self.population_size = population_size
        self.all_factors = []
        self.corr_threshold = corr_threshold
        self.functions = build_function_set()

        self.max_depth = max_depth
        if basic_factors is not None:
            self.basic_factors = basic_factors
        else:
            self.basic_factors = self.data.columns.to_list()
        
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.rounds = rounds

    def initiate_trees(self):
        """初始化种群中的所有表达式树"""
        for population_index in tqdm(range(self.population_size), 
                                     desc="Initiating population",
                                      total=self.population_size):
            
            self.all_factors.append(ExpressionFactor.generate_random_factor(functions=self.functions['all'], 
                                                                      basic_factors=self.basic_factors, 
                                                                      max_depth=self.max_depth))

    def mutate(self):
        """随机突变"""
        factors_to_mutate = list(self.all_factors)
        for factor_obj in tqdm(factors_to_mutate, 
                                     desc="Mutating population",
                                      total=self.population_size):
            
            mutation_factor = factor_obj.mutate(mutation_prob=self.mutation_prob,
                                                all_functions=self.functions['all'],
                                                basic_factors=self.basic_factors)
            self.all_factors.append(mutation_factor)

    def crossover(self):
        """交叉变换"""
        num_factors = len(self.all_factors)
        for factor_index in tqdm(range(num_factors),
                                desc="Crossover",
                                total=num_factors):
            pro = random.uniform(0, 1)
            if pro < self.crossover_prob:
                first_factor = self.all_factors[factor_index]
                sub_info = first_factor.get_subfactor()
                if len(sub_info) == 0:
                    continue
                first_left_factor, first_subfactor, first_right_factor = sub_info

                for factor_index_2 in range(factor_index + 1, num_factors):
                    pro = random.uniform(0, 1)
                    if pro < self.crossover_prob:
                        second_factor = self.all_factors[factor_index_2]
                        sec_sub_info = second_factor.get_subfactor()
                        if len(sec_sub_info) == 0:
                            continue
                        second_left_factor, second_subfactor, second_right_factor = sec_sub_info

                        new_tree_1 = first_left_factor + second_subfactor + first_right_factor
                        new_tree_2 = second_left_factor + first_subfactor + second_right_factor
                        self.all_factors.append(ExpressionFactor(new_tree_1))
                        self.all_factors.append(ExpressionFactor(new_tree_2))

    def get_high_fitness_factors(self):
        fitness_list = []
        for factor_index in tqdm(range(len(self.all_factors)),
                           desc="Getting high fitness factors",
                           total=len(self.all_factors)):
            factor = self.all_factors[factor_index]
            factor_data = factor.calculate_factor_value(self.data)
            fitness = self.fitness_calculator.calculate_fitness(factor_data)
            fitness_list.append(fitness)

        reserved_factors = self.fitness_calculator.select_by_fitness(fitness_list)
        high_fitness_factors = [self.all_factors[reserved_index] 
                             for reserved_index ,reserved_or_drop in enumerate(reserved_factors) 
                             if reserved_or_drop == True]
        
        self.all_factors = high_fitness_factors

    def remove_high_correlation_factors(self):
        """
        移除高度相关的因子，只保留代表性的因子。
        
        参数:
            threshold (float): 相关系数阈值，超过此值则认为相关性过高
        """
        n = len(self.all_factors)
        if n <= 1:
            return

        # 先计算每个因子的数据
        factor_data_list = []
        for factor in self.all_factors:
            try:
                data = factor.calculate_factor_value(self.data)
                factor_data_list.append(data.values)  # 只取数值部分
            except Exception as e:
                print(f"Error calculating factor: {e}")
                factor_data_list.append(np.zeros(len(self.data)))

        # 构建相关矩阵
        corr_matrix = np.zeros((n, n))
        for i, j in combinations(range(n), 2):
            corr = self.correlation_calculator.calculate_factor_correlation(factor_data_list[i], factor_data_list[j])
            corr_matrix[i, j] = abs(corr)
            corr_matrix[j, i] = abs(corr)  # 对称填充

        # 贪心选择保留的因子索引
        to_keep = list(range(n))
        for i in tqdm(range(n), desc="Processing factors", total=n):  # 添加进度条
            for j in range(i + 1, n):
                if corr_matrix[i, j] > self.corr_threshold:
                    # 如果 i 和 j 相关性太高，删除 j
                    if j in to_keep:
                        to_keep.remove(j)

        # 更新 all_trees
        self.all_factors = [self.all_factors[i] for i in to_keep]
        print(f"Removed highly correlated factors. Remaining: {len(to_keep)}")

    def generate_factors(self):
        self.initiate_trees()
        for round in range(self.rounds):
            self.mutate()
            self.crossover()
            self.get_high_fitness_factors()
            self.remove_high_correlation_factors()
        return self.all_factors

if __name__ == '__main__':
    gene_algo = GeneAlgo()
    gene_algo.initiate_trees()
