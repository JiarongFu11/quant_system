import unittest
import numpy as np
import pandas as pd
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from factor.function import (
    TemporalMax, TemporalMin, TemporalSum, TemporalMean, TemporalStd,
    TemporalDelta, TemporalDecayLinear, Abs, Negative, Sign, Sqrt, Square,
    Cube, Cbrt, Log, Inv, Add, Subtract, Multiply, Divide
)
from data.data_storage import DataStorageFacade

class TestFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """从HDF5存储中加载BTCUSDT在6月24日的数据"""
        # 初始化数据存储
        data_facade = DataStorageFacade()
        hdf5_storage = data_facade.add_file_storage('hdf5', 'data_factory')
        
        # 加载2024年6月24日的数据
        cls.data = hdf5_storage.load_data(
            table_name='2025-06-20',
            start_date='2025-06-19',
            end_date='2025-06-21',
            as_dataframe=True
        )
        
        # 确保数据存在
        if cls.data.empty:
            raise ValueError("No data found for BTCUSDT on 2024-06-24")
        
        # 提取收盘价作为测试数据
        cls.close_prices = cls.data['close'].to_numpy().astype(np.float64).reshape(-1, 1)
        # 打印基本信息
        print(f"Loaded {len(cls.close_prices)} data points for BTCUSDT on 2024-06-24")
        print(f"Sample close prices: {cls.close_prices[:5]}")

    def test_temporal_functions(self):
        """测试所有时序函数"""
        lookback = 5  # 回看周期
        
        # 测试TemporalMax
        ts_max = TemporalMax()(self.close_prices, lookback)
        self.assertEqual(len(ts_max), len(self.close_prices))
        self.assertTrue(np.isnan(ts_max[:lookback-1]).all())  # 前lookback-1个应为NaN
        self.assertFalse(np.isnan(ts_max[lookback-1:]).any())  # 之后的不应有NaN
        
        # 手动计算验证最后几个值
        for i in range(lookback-1, len(self.close_prices)):
            window = self.close_prices[i-lookback+1:i+1]
            self.assertAlmostEqual(ts_max[i], np.max(window), places=6)
        
        # 测试TemporalMin
        ts_min = TemporalMin()(self.close_prices, lookback)
        for i in range(lookback-1, len(self.close_prices)):
            window = self.close_prices[i-lookback+1:i+1]
            self.assertAlmostEqual(ts_min[i], np.min(window), places=6)
        
        # 测试TemporalMean
        ts_mean = TemporalMean()(self.close_prices, lookback)
        for i in range(lookback-1, len(self.close_prices)):
            window = self.close_prices[i-lookback+1:i+1]
            self.assertAlmostEqual(ts_mean[i], np.mean(window), places=6)
        
        # 测试TemporalSum
        ts_sum = TemporalSum()(self.close_prices, lookback)
        for i in range(lookback-1, len(self.close_prices)):
            window = self.close_prices[i-lookback+1:i+1]
            self.assertAlmostEqual(ts_sum[i], np.sum(window), places=6)
        
        # 测试TemporalStd
        ts_std = TemporalStd()(self.close_prices, lookback)
        for i in range(lookback-1, len(self.close_prices)):
            window = self.close_prices[i-lookback+1:i+1]
            self.assertAlmostEqual(ts_std[i], np.std(window), places=6)
        
        # 测试TemporalDelta
        ts_delta = TemporalDelta()(self.close_prices, lookback)
        for i in range(lookback-1, len(self.close_prices)):
            window = self.close_prices[i-lookback+1:i+1]
            expected = window[-1] - window[0]
            self.assertAlmostEqual(ts_delta[i], expected, places=6)
        
        # 测试TemporalDecayLinear
        ts_decay = TemporalDecayLinear()(self.close_prices, lookback)
        for i in range(lookback-1, len(self.close_prices)):
            window = self.close_prices[i-lookback+1:i+1]
            weights = np.arange(1.0, lookback + 1.0)
            print(weights, window)
            weighted_sum = np.dot(window.flatten(), weights)
            expected = weighted_sum / np.sum(weights)
            self.assertAlmostEqual(ts_decay[i], expected, places=6)

    def test_unary_functions(self):
        """测试所有一元函数"""
        # 测试Abs
        abs_vals = Abs()(self.close_prices)
        np.testing.assert_array_almost_equal(abs_vals, np.abs(self.close_prices))
        
        # 测试Negative
        neg_vals = Negative()(self.close_prices)
        np.testing.assert_array_almost_equal(neg_vals, -self.close_prices)
        
        # 测试Sign
        sign_vals = Sign()(self.close_prices)
        np.testing.assert_array_almost_equal(sign_vals, np.sign(self.close_prices))
        
        # 测试Sqrt
        sqrt_vals = Sqrt()(self.close_prices)
        np.testing.assert_array_almost_equal(sqrt_vals, np.sqrt(np.abs(self.close_prices)))
        
        # 测试Square
        square_vals = Square()(self.close_prices)
        np.testing.assert_array_almost_equal(square_vals, np.square(self.close_prices))
        
        # 测试Cube
        cube_vals = Cube()(self.close_prices)
        np.testing.assert_array_almost_equal(cube_vals, np.power(self.close_prices, 3))
        
        # 测试Cbrt
        cbrt_vals = Cbrt()(self.close_prices)
        np.testing.assert_array_almost_equal(cbrt_vals, np.cbrt(self.close_prices))
        
        # 测试Log
        log_vals = Log()(self.close_prices)
        expected = np.log(np.abs(self.close_prices) + 1e-9)
        np.testing.assert_array_almost_equal(log_vals, expected)
        
        # 测试Inv
        inv_vals = Inv()(self.close_prices)
        expected = 1.0 / (self.close_prices + 1e-9)
        np.testing.assert_array_almost_equal(inv_vals, expected)

    def test_binary_functions(self):
        """测试二元函数"""
        # 使用收盘价和开盘价进行测试
        open_prices = self.data['open'].to_numpy().astype(np.float64)
        
        # 测试Add
        add_result = Add(2)(self.close_prices, open_prices)
        expected = self.close_prices + open_prices
        np.testing.assert_array_almost_equal(add_result, expected)
        
        # 测试Subtract
        sub_result = Subtract(2)(self.close_prices, open_prices)
        expected = self.close_prices - open_prices
        np.testing.assert_array_almost_equal(sub_result, expected)
        
        # 测试Multiply
        mul_result = Multiply(2)(self.close_prices, open_prices)
        expected = self.close_prices * open_prices
        np.testing.assert_array_almost_equal(mul_result, expected)
        
        # 测试Divide
        div_result = Divide(2)(self.close_prices, open_prices)
        expected = self.close_prices / (open_prices + 1e-9)
        np.testing.assert_array_almost_equal(div_result, expected)

    def test_ternary_functions(self):
        """测试三元函数"""
        # 使用收盘价、开盘价和最高价进行测试
        open_prices = self.data['open'].to_numpy().astype(np.float64)
        high_prices = self.data['high'].to_numpy().astype(np.float64)
        
        # 测试三元Add
        add_result = Add(3)(self.close_prices, open_prices, high_prices)
        expected = self.close_prices + open_prices + high_prices
        np.testing.assert_array_almost_equal(add_result, expected)
        
        # 测试三元Multiply
        mul_result = Multiply(3)(self.close_prices, open_prices, high_prices)
        expected = self.close_prices * open_prices * high_prices
        np.testing.assert_array_almost_equal(mul_result, expected)

if __name__ == '__main__':
    unittest.main()