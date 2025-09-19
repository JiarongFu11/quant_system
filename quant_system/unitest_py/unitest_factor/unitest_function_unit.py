import unittest
import pandas as pd
import random
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from unittest.mock import patch, Mock, call

# 假设您的代码保存在 'factor_calculator.py' 文件中
from factor.factor_unit import ExpressionFactor, FactorCalculator

# MockFunction 不再需要继承，因为我们将 patch 真正的 Function
class MockFunction:
    def __init__(self, name, arity, func):
        self.name = name
        self.arity = arity
        self._func = func

    def __call__(self, *args):
        if len(args) != self.arity:
            raise TypeError(f"'{self.name}' takes {self.arity} arguments but {len(args)} were given")
        return self._func(*args)

    def __eq__(self, other):
        return isinstance(other, MockFunction) and self.name == other.name

# --- 关键修复：用 MockFunction 替换掉代码模块中的 Function ---
@patch('factor.factor_unit.Function', new_callable=lambda: MockFunction)
class TestExpressionFactor(unittest.TestCase):
    """测试 ExpressionFactor 类的所有功能。"""

    def setUp(self):
        """在每个测试方法运行前，设置通用的测试对象。"""
        # 注意: setUp 现在接收一个参数 MockFunctionClass，这是 @patch 的结果
        # 创建模拟函数
        self.add = MockFunction('add', 2, lambda x, y: x + y)
        self.neg = MockFunction('neg', 1, lambda x: -x)
        self.multiply = MockFunction('multiply', 2, lambda x, y: x * y)
        
        self.all_functions = [self.add, self.neg, self.multiply]
        self.basic_factors = ['open', 'close', 'high', 'low']
        
        self.data = pd.DataFrame({
            'open': [10, 20, 30],
            'close': [11, 21, 31],
            'high': [12, 22, 32],
            'low': [9, 19, 29]
        })

    def test_init_and_repr(self, MockFunctionClass):
        """测试初始化和字符串表示。"""
        nodes = [self.add, 'open', 'close']
        factor = ExpressionFactor(nodes)
        self.assertEqual(factor.nodes, nodes)
        self.assertEqual(len(factor), 3)
        self.assertEqual(repr(factor), "ExpressionTree(add, open, close)")
        
        with self.assertRaises(ValueError):
            ExpressionFactor("not a list")

    def test_calculate_factor_value_simple(self, MockFunctionClass):
        """测试简单因子的计算：add(open, close)"""
        # 注意：您的原始计算逻辑是错误的，我已在上面修正为标准的逆波兰表达式求值。
        # 如果您坚持使用原始的计算逻辑，这个测试将会失败。
        factor = ExpressionFactor([self.add, 'open', 'close'])
        result = factor.calculate_factor_value(self.data)
        expected = self.data['open'] + self.data['close']
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_calculate_factor_value_nested(self, MockFunctionClass):
        """测试嵌套因子的计算：multiply(add(open, close), high)"""
        nodes = [self.multiply, self.add, 'open', 'close', 'high']
        factor = ExpressionFactor(nodes)
        result = factor.calculate_factor_value(self.data)
        expected = (self.data['open'] + self.data['close']) * self.data['high']
        pd.testing.assert_series_equal(result, expected, check_names=False)
        
    def test_calculate_factor_value_single_operation(self, MockFunctionClass):
        """测试单操作数因子：neg(open)"""
        factor = ExpressionFactor([self.neg, 'open'])
        result = factor.calculate_factor_value(self.data)
        expected = -self.data['open']
        pd.testing.assert_series_equal(result, expected, check_names=False)

    @patch('random.choice')
    @patch('random.gauss')
    @patch('random.uniform')
    def test_generate_random_factor(self, mock_uniform, mock_gauss, mock_choice, MockFunctionClass):
        """测试随机因子生成，通过 mock random 来获得确定性结果。"""
        mock_uniform.return_value = 0.5
        
        # --- 修复：引导代码走向正确的路径 ---
        # 期望生成 add(open, close)
        # 1. 第一次 pro > 0.5 (选择基础因子)
        # 2. 第二次 pro > 0.5 (选择基础因子)
        mock_gauss.side_effect = [0.7, 0.7]

        mock_choice.side_effect = [
            self.add,   # 第一次 choice，选择函数
            'open',     # 第二次 choice，选择基础因子
            'close'     # 第三次 choice，选择基础因子
        ]

        # 调用类方法
        # 注意: 您原始代码中 generate_random_factor 不是一个 classmethod
        # 我在上面加了 @classmethod，这里才能这样调用
        factor = ExpressionFactor.generate_random_factor(
            functions=[self.add, self.neg], 
            basic_factors=self.basic_factors,
            max_depth=3
        )
        self.assertIsInstance(factor, ExpressionFactor)
        self.assertEqual(factor.nodes, [self.add, 'open', 'close'])

    @patch('random.uniform') # --- 修复：添加 patch ---
    @patch('random.choice')
    def test_get_subfactor(self, mock_choice, mock_uniform, MockFunctionClass):
        """测试获取子因子（子树）的功能。"""
        nodes = [self.multiply, self.add, 'open', 'close', self.neg, 'high']
        factor = ExpressionFactor(nodes)

        mock_choice.return_value = 1
        mock_uniform.return_value = 0.6 # --- 修复：控制随机性 ---
        
        parts = factor.get_subfactor()

        self.assertIsNotNone(parts) # 现在不应为 None
        left, sub, right = parts
        
        self.assertEqual(left, [self.multiply])
        self.assertEqual(sub, [self.add, 'open', 'close'])
        self.assertEqual(right, [self.neg, 'high'])
        self.assertEqual(left + sub + right, nodes)

    # ... test_mutate 保持不变，它本身是正确的 ...
    @patch('random.randint')
    @patch('random.random')
    @patch('random.choice')
    def test_mutate(self, mock_choice, mock_random, mock_randint, MockFunctionClass):
        nodes = [self.add, 'open', 'close']
        factor = ExpressionFactor(nodes)
        
        mock_random.return_value = 0.05
        mock_randint.return_value = 0
        mock_choice.return_value = self.multiply 
        
        mutated_factor = factor.mutate(
            mutation_prob=0.1, 
            all_functions=self.all_functions,
            basic_factors=self.basic_factors
        )
        self.assertEqual(mutated_factor.nodes, [self.multiply, 'open', 'close'])

# --- 针对 FactorCalculator 的独立测试类 ---
class TestFactorCalculator(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})

    # --- 修复：测试现在应该通过，因为它测试的是修正后的代码 ---
    @patch('factor.factor_unit.ExpressionFactor')
    def test_calculate_factors(self, MockExpressionFactor):
        mock_result_1 = pd.Series([4, 6], name='factor1')
        mock_result_2 = pd.Series([5, 12], name='factor2')

        mock_instance_1 = Mock()
        mock_instance_1.calculate_factor_value.return_value = mock_result_1
        mock_instance_2 = Mock()
        mock_instance_2.calculate_factor_value.return_value = mock_result_2
        
        MockExpressionFactor.side_effect = [mock_instance_1, mock_instance_2]

        calculator = FactorCalculator()
        factors_to_calc = [['add', 'A', 'B'], ['multiply', 'A', 'C']]
        result_df = calculator.calculate_factors(factors_to_calc, self.data)

        self.assertEqual(MockExpressionFactor.call_count, 2)
        MockExpressionFactor.assert_has_calls([
            call(factors_to_calc[0]), call(factors_to_calc[1])
        ])
        mock_instance_1.calculate_factor_value.assert_called_once_with(self.data)
        mock_instance_2.calculate_factor_value.assert_called_once_with(self.data)
        expected_df = pd.concat([mock_result_1, mock_result_2], axis=1)
        pd.testing.assert_frame_equal(result_df, expected_df)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False, verbosity=2)