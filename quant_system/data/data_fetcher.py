import time
import pandas as pd
import os
import sys
import re
# 添加项目根目录到 PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from data.base import IDataRetriever
from datetime import datetime, timedelta
from binance.client import Client
from binance.exceptions import BinanceAPIException
from data.data_storage import DataStorageFacade
from utils.date import generate_date_string_sequence, datetime_to_str, strdatetime_to_timestamp

class BinanceDataRetriever(IDataRetriever):
    KLINE_COLUMNS = [
    'datetime',
    'open',
    'high',
    'low',
    'close',
    'volume',
    'close_time',
    'quote_volume',
    'trades',
    'taker_buy_base_volume',
    'taker_buy_quote_volume',
    'ignore'
    ] #如果binance的数据格式换了，此处需要更改
    def __init__(self):
        """
        初始化 Binance 客户端
        :param api_key: API 密钥（可选）
        :param api_secret: API 密钥（可选）
        """
        api_key = 'LGQjAko99Bp9RI1linlJAFH6zyTSQH6KZJGYAk26tNc299duQcvaDD3LqZXnZKyJ'
        api_secret = 'ErEkQuBFT6vSQsN4cP9S6wdaNvCLxuQGVFYAvRoTJZGEnyJuoA8g3A978oxe0lK3'
        self.client = Client(api_key, api_secret)

    def get_history_data(self, symbol='BTCUSDT', interval='1min', country='Asia/Shanghai',
                         start_date=None, end_date=None, delay=1):
        """
        根据日期范围获取历史 K 线数据，并转换为 DataFrame
        :param symbol: 交易对，例如 'BTCUSDT'
        :param interval: K 线间隔，例如 '1h'（1 小时）、'1d'（1 天）
        :param start_date: 开始日期，格式为 'YYYY-MM-DD'
        :param end_date: 结束日期，格式为 'YYYY-MM-DD'
        :param delay: 每次请求后的延迟时间（秒）
        :return: 包含 K 线数据的 DataFrame
        """
        if not start_date or not end_date:
            raise ValueError("start_date 和 end_date 不能为空")

        # 将日期字符串转换为时间戳（毫秒）
        start_timestamp = strdatetime_to_timestamp(start_date)
        if end_date == start_date:
            '此情况仅能用于单天数据获取'
            dt = datetime.strptime(end_date, '%Y-%m-%d')
            dt_end = dt.replace(hour=23, minute=59,microsecond=0)
            end_timestamp = int(dt_end.timestamp() * 1000)
        else:
            if re.match(r'^\d{4}-\d{2}-\d{2}$', end_date):
                '将时间调整为日期当天最后一时刻'
                dt = datetime.strptime(end_date, '%Y-%m-%d')
                dt_end = dt.replace(hour=23, minute=59,microsecond=0)
                end_timestamp = int(dt_end.timestamp() * 1000)
            else:
                end_timestamp = strdatetime_to_timestamp(end_date)

        # 获取历史 K 线数据
        klines = []
        while start_timestamp < end_timestamp:
            try:
                # 获取数据
                data = self.client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    startTime=start_timestamp,
                    endTime=end_timestamp,
                    limit=1440  # 每次最多获取 1000 条数据
                )
                if not data:
                    break

                # 添加到结果列表
                klines.extend(data)

                # 更新开始时间
                start_timestamp = data[-1][0] + 1  # 使用最后一条数据的时间戳 +1 毫秒

                # 控制请求频率
                time.sleep(delay)

            except BinanceAPIException as e:
                if e.code == -1003 or e.code == -1006:  # 速率限制错误
                    print(f"触发速率限制，等待 60 秒后重试... 错误信息: {e}")
                    time.sleep(60)  # 等待 60 秒后重试
                else:
                    raise e  # 其他错误直接抛出

        # 将 K 线数据转换为 DataFrame
        data = pd.DataFrame(klines, columns=self.KLINE_COLUMNS)
        data['datetime'] = pd.to_datetime(data['datetime'], unit='ms', utc=True)
        if country is not None:
            data['datetime'] = data['datetime'].dt.tz_convert(country)
        data['datetime'] = data['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        return data

    def get_realtime_data(self, symbol='BTCUSDT', interval='1min', start_date=None):
        utc_now = datetime.utcnow()
        end_date = utc_now.strftime('%Y-%m-%d %H:%M')
        return self.get_history_data(symbol, interval, start_date, end_date=end_date)

class DataRetrieverFacade():
    '一个工作流程只能用一个数据源'
    _instance = None
    def __new__(cls, data_retriver):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.data_retriver = None
        return cls._instance
    
    def __init__(self, data_retriver):
        if self.data_retriver is None:
            if data_retriver == 'binance':
                self.data_retriver = BinanceDataRetriever()
            else:
                raise KeyError(f"Unsupported data source: {data_retriver}")
        
    def create_data_retriver(self):
        '用于创建数据获取实例'
        return self.data_retriver
    
    def update_data(self, start_date, end_date, database_operator: DataStorageFacade, alias, 
                    symbol='BTCUSDT', interval='1m', if_exists='cover'):
        """
        :params start_date: 开始日期，格式为 'YYYY-MM-DD' 
        :params end_date: 结束日期，格式为 'YYYY-MM-DD' 
        :params database_operator: 已经添加storage或者database的实例
        :params alias: dbtype_filefactory
        :params symbol: 交易对
        :params interval: K线间隔，默认为 '1m'
        :params if_exists: 'cover' or 'keep'
        """
        format = '%Y-%m-%d'
        if alias is None:
            alias = f'hdf5_{symbol}'
        assert re.match(r'^\d{4}-\d{2}-\d{2}', start_date) and re.match(r'^\d{4}-\d{2}-\d{2}', end_date)
        dates = generate_date_string_sequence(start_date=start_date, end_date=end_date, freq='D', format=format)
        dates = datetime_to_str(dates)
        print(alias)
        storage = database_operator.get_storage(alias)

        for date in dates:
            table_name = f'{date}'
            if if_exists == 'cover':
                klines = self.data_retriver.get_history_data(symbol=symbol, interval=interval, 
                                                             start_date=start_date, end_date=end_date)
                print(klines)
                storage.save_data(klines, table_name)
                print(f'update data {table_name} successfully')
            elif if_exists == 'keep':
                try:
                    storage.load_data(table_name=table_name)
                    print(f'{table_name} has been existed')
                except KeyError:
                    klines = self.data_retriver.get_history_data(symbol=symbol, interval=interval, 
                                                             start_date=start_date, end_date=end_date)
                    storage.save_data(klines, table_name)
                    print(f'update data {table_name} successfully')

if __name__ == '__main__':
    data_retriver = DataRetrieverFacade('binance')
    datastorage = DataStorageFacade()
    datastorage.add_file_storage('hdf5', 'BTCUSDT')
    data_retriver.update_data(start_date='2025-01-01', end_date='2025-01-05', 
                              database_operator=datastorage, alias='hdf5_BTCUSDT')