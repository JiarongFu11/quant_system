import sys
import os
import pandas as pd
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from utils.date import generate_date_string_sequence, is_same_day, str_to_datetime, shift_start_date, shift_end_date, get_trade_dates

from data.data_storage import DataStorageFacade
from typing import List

class DataOperator():
    def __init__(self, data_storage: DataStorageFacade, db_type: str):
        """
        params: data_storage: 数据存储对象
        """
        self.data_storage = data_storage
        self.db_type = db_type

    def get_security_min_data(self, start_date: str, end_date: str, symbol:str) -> pd.DataFrame:
        """
        获取指定时间段的所有证券的分钟数据
        :param start_date: 开始日期, 格式为 'YYYY-MM-DD HH:MM'或者'YYYY-MM-DD'
        :param end_date: 结束日期, 格式为 'YYYY-MM-DD HH:MM'或者'YYYY-MM-DD'
        :return:
        """
        security_data_storage = self.data_storage.get_storage(f'{self.db_type}_{symbol}')
        if is_same_day(start_date, end_date):
            '获取一天内的数据'
            table_name = str_to_datetime(start_date).date
            return security_data_storage.load_data(table_name=table_name, start_date=start_date, end_date=end_date)
        else:
            date_range = generate_date_string_sequence(start_date=start_date, end_date=end_date, freq='D')
            for date_index, date in enumerate(date_range):
                table_name = str_to_datetime(date).date
                if date_index == 0:
                    all_security_data = security_data_storage.load_data(table_name=table_name, start_date=date)
                elif date_index == len(date_range) - 1:
                    security_data = security_data_storage.load_data(table_name=table_name, end_date=date)
                    all_security_data = pd.concat([all_security_data, security_data])
                else:
                    security_data = security_data_storage.load_data(table_name=table_name)
                    all_security_data = pd.concat([all_security_data, security_data])
        
        return all_security_data
    
    def get_security_data(self,
                          start_date: str, 
                          end_date: str,   
                          symbol: str,
                          freq: int = 1):
        
        minute_data = self.get_security_min_data(start_date, end_date, symbol)
        


        
        
    
    def get_security_data_with_shifted_start(self, 
                                             start_date: str,
                                             end_date: str, 
                                             symbol: str, 
                                             freq: int, 
                                             shift_period: int):
        start_date = shift_start_date(start_date, n_periods=shift_period, freq=freq)
        data = self.get_security_data(start_date=start_date, end_date=end_date, symbol=symbol, freq=freq)

        return data


    def _get_security_data_with_shifted_end(self, 
                                            start_date: str,
                                            end_date: str, 
                                            symbol: str, 
                                            freq: int, 
                                            shift_period: int):
        end_date = shift_end_date(end_date, n_periods=shift_period, freq=freq)
        data = self.get_security_min_data(start_date=start_date, end_date=end_date, symbol=symbol, freq=freq)
        
        return data

    def get_security_close_to_close_returns(self, 
                             start_date: str,
                             end_date: str, 
                             symbol: str, 
                             freq: int, 
                             shift_period: int,
                             trade_interval: int = 1):
        

        data = self._get_security_data_with_shifted_end(start_date=start_date,
                                                        end_date=end_date, 
                                                        symbol=symbol,
                                                        freq=freq,
                                                        shift_period=shift_period)
        data['buy_close'] = data['close'].shift(-trade_interval)

        if freq == 'T' or freq == 'min':
            min_shift_period = shift_period
        elif freq == 'H':
            min_shift_period = shift_period * 60
        elif freq == 'D':
            min_shift_period = shift_period * 60 * 24

        data['sell_close'] = data['close'].shift(-min_shift_period)
        data['return'] = (data['sell_close'] - data['buy_close']) / data['buy_close']

        dates = get_trade_dates(start_date, end_date, n_periods=shift_period, freq=freq)
        data.set_index('datetime', inplace=True)
        data = data.loc[dates, 'return'].reset_index()
        
        return data
        



