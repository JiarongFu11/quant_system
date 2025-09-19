import unittest
import pandas as pd
from unittest.mock import patch, Mock, MagicMock, call
from datetime import datetime
from binance.exceptions import BinanceAPIException
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)
# 假设您的代码保存在名为 'data_retriever' 的文件中
# 如果文件名不同，请相应修改下面的导入语句
from data.data_fetcher import BinanceDataRetriever, DataRetrieverFacade

class TestBinanceDataRetriever(unittest.TestCase):
    """
    测试 BinanceDataRetriever 类。
    我们将模拟 (mock) `binance.Client`，以避免在测试中进行真实的网络 API 调用。
    """

    @patch('data.data_fetcher.Client')  # 模拟 Client 类
    def setUp(self, MockClient):
        """
        在每个测试方法运行前执行。
        """
        # 创建一个模拟的 client 实例
        self.mock_client_instance = Mock()
        # 让 Client() 的调用返回我们的模拟实例
        MockClient.return_value = self.mock_client_instance
        
        # 实例化我们要测试的类
        self.retriever = BinanceDataRetriever()

    def test_initialization(self):
        """
        测试初始化是否正确调用了 Binance Client。
        """
        # 验证 Client 是否用代码中硬编码的 key 和 secret 初始化
        from data.data_fetcher import Client # 从模块中获取 Client 以便 patch 生效
        Client.assert_called_once_with(
            'LGQjAko99Bp9RI1linlJAFH6zyTSQH6KZJGYAk26tNc299duQcvaDD3LqZXnZKyJ',
            'ErEkQuBFT6vSQsN4cP9S6wdaNvCLxuQGVFYAvRoTJZGEnyJuoA8g3A978oxe0lK3'
        )
        self.assertIsNotNone(self.retriever.client)

    def test_get_history_data_success(self):
        """
        测试 get_history_data 成功获取数据并正确处理的场景。
        """
        # 准备模拟的 API 返回数据
        mock_kline_data = [
            [1672531200000, '16500', '16501', '16499', '16500.5', '100', 1672531259999, '1650025', 50, '60', '990015', '0'],
            [1672531260000, '16500.5', '16502', '16500', '16501.5', '120', 1672531319999, '1980120', 60, '70', '1155070', '0']
        ]
        # 当 get_klines 被调用时，返回我们的模拟数据，并在下一次调用时返回空列表以终止循环
        self.mock_client_instance.get_klines.side_effect = [mock_kline_data, []]

        # 调用被测试的方法
        df = self.retriever.get_history_data(start_date='2023-01-01', end_date='2023-01-01')

        # 断言
        self.mock_client_instance.get_klines.assert_called_once()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertListEqual(list(df.columns), self.retriever.KLINE_COLUMNS)
        self.assertEqual(df.iloc[0]['datetime'], '2023-01-01 08:00:00') # 验证时区转换 (Asia/Shanghai)
        self.assertEqual(df.iloc[0]['open'], '16500')

    def test_get_history_data_raises_value_error(self):
        """
        测试在缺少日期参数时是否抛出 ValueError。
        """
        with self.assertRaises(ValueError):
            self.retriever.get_history_data(start_date='2023-01-01') # end_date is missing
        with self.assertRaises(ValueError):
            self.retriever.get_history_data(end_date='2023-01-01') # start_date is missing

    @patch('data.data_fetcher.time.sleep') # 模拟 time.sleep 防止测试等待
    def test_get_history_data_handles_rate_limit_exception(self, mock_sleep):
        """
        测试在遇到 API 速率限制时，代码是否会等待并重试。
        """
        rate_limit_exception = BinanceAPIException(request=None, response=None, message="Rate limit exceeded", code=-1003)
        mock_kline_data = [
            [1672531200000, '16500', '16501', '16499', '16500.5', '100', 1672531259999, '1650025', 50, '60', '990015', '0']
        ]
        # 设置 side_effect：第一次调用抛出异常，第二次返回数据，第三次返回空列表
        self.mock_client_instance.get_klines.side_effect = [rate_limit_exception, mock_kline_data, []]

        # 调用方法
        df = self.retriever.get_history_data(start_date='2023-01-01', end_date='2023-01-01')

        # 断言
        self.assertEqual(self.mock_client_instance.get_klines.call_count, 2)
        mock_sleep.assert_called_once_with(60) # 验证是否等待了60秒
        self.assertEqual(len(df), 1)

class TestDataRetrieverFacade(unittest.TestCase):
    """
    测试 DataRetrieverFacade 类。
    我们将模拟整个 BinanceDataRetriever 类和 DataStorageFacade。
    """

    # 模拟多个依赖项
    @patch('data.data_fetcher.DataStorageFacade')
    @patch('data.data_fetcher.BinanceDataRetriever')
    @patch('utils.date.generate_date_string_sequence')
    @patch('utils.date.datetime_to_str')
    def setUp(self, mock_dt_to_str, mock_date_seq, MockBinanceRetriever, MockStorageFacade):
        # 清理单例，确保每个测试都是独立的
        if hasattr(DataRetrieverFacade, '_instance'):
            DataRetrieverFacade._instance = None

        # --- 配置模拟对象 ---
        # 模拟日期生成函数
        self.mock_date_seq = mock_date_seq
        self.mock_dt_to_str = mock_dt_to_str
        self.mock_dt_to_str.return_value = ['2023-01-01']
        
        # 模拟数据获取器
        self.mock_retriever_instance = Mock()
        self.mock_retriever_instance.get_history_data.return_value = pd.DataFrame({'A': [1]})
        MockBinanceRetriever.return_value = self.mock_retriever_instance

        # 模拟存储
        self.mock_storage_instance = Mock()
        self.mock_storage_facade_instance = Mock()
        self.mock_storage_facade_instance.get_storage.return_value = self.mock_storage_instance
        # 将模拟实例赋值给类，以便在测试中断言
        self.MockStorageFacade = MockStorageFacade 
        self.MockStorageFacade.return_value = self.mock_storage_facade_instance

        # 实例化被测试的 Facade
        self.facade = DataRetrieverFacade(data_retriver='binance')
        

    def test_singleton_pattern(self):
        """
        测试 Facade 是否遵循单例模式。
        """
        facade1 = DataRetrieverFacade(data_retriver='binance')
        facade2 = DataRetrieverFacade(data_retriver='binance')
        self.assertIs(facade1, facade2)
        # 即使使用不同的参数，也应返回第一个实例
        facade3 = DataRetrieverFacade(data_retriver='other')
        self.assertIs(facade1, facade3)
        self.assertIsInstance(facade3.data_retriver, type(self.mock_retriever_instance))

    def test_unsupported_data_source_raises_key_error(self):
        """
        测试不支持的数据源是否会抛出 KeyError。
        """
        # 清理单例以重新初始化
        DataRetrieverFacade._instance = None
        with self.assertRaises(KeyError):
            DataRetrieverFacade(data_retriver='unsupported_source')
            
    def test_update_data_if_exists_cover(self):
        """
        测试 update_data 在 if_exists='cover' 模式下的行为。
        """
        # 调用方法
        self.facade.update_data(
            start_date='2023-01-01', end_date='2023-01-01',
            database_operator=self.mock_storage_facade_instance,
            alias='test_alias', if_exists='cover'
        )

        # 断言
        # 1. 数据获取器被调用
        self.mock_retriever_instance.get_history_data.assert_called_once()
        # 2. 数据被保存
        self.mock_storage_instance.save_data.assert_called_once_with(
            self.mock_retriever_instance.get_history_data.return_value,
            '2023-01-01' # table_name
        )
        # 3. 不会尝试加载数据
        self.mock_storage_instance.load_data.assert_not_called()

    def test_update_data_if_exists_keep_and_data_exists(self):
        """
        测试 if_exists='keep' 且数据已存在的场景。
        """
        # 模拟 load_data 成功（不抛出异常）
        self.mock_storage_instance.load_data.return_value = pd.DataFrame({'B': [2]})
        
        # 调用方法
        self.facade.update_data(
            start_date='2023-01-01', end_date='2023-01-01',
            database_operator=self.mock_storage_facade_instance,
            alias='test_alias', if_exists='keep'
        )
        
        # 断言
        # 1. 尝试加载数据
        self.mock_storage_instance.load_data.assert_called_once_with(table_name='2023-01-01')
        # 2. 因为加载成功，所以不获取新数据
        self.mock_retriever_instance.get_history_data.assert_not_called()
        # 3. 因为加载成功，所以不保存数据
        self.mock_storage_instance.save_data.assert_not_called()

    def test_update_data_if_exists_keep_and_data_not_exists(self):
        """
        测试 if_exists='keep' 且数据不存在的场景。
        """
        # 模拟 load_data 失败 (抛出 KeyError)
        self.mock_storage_instance.load_data.side_effect = KeyError("Table not found")
        
        # 调用方法
        self.facade.update_data(
            start_date='2023-01-01', end_date='2023-01-01',
            database_operator=self.mock_storage_facade_instance,
            alias='test_alias', if_exists='keep'
        )
        
        # 断言
        # 1. 尝试加载数据
        self.mock_storage_instance.load_data.assert_called_once_with(table_name='2023-01-01')
        # 2. 因为加载失败，所以获取新数据
        self.mock_retriever_instance.get_history_data.assert_called_once()
        # 3. 因为加载失败，所以保存新获取的数据
        self.mock_storage_instance.save_data.assert_called_once_with(
            self.mock_retriever_instance.get_history_data.return_value,
            '2023-01-01' # table_name
        )


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)