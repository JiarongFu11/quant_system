import pandas as pd
import re

from datetime import datetime, timedelta
from typing import List

def generate_date_sequence(start_date:str, end_date:str, freq='D'):
    """
    根据 start_date 和 end_date 生成日期序列
    :param start_date: 起始日期，格式为 'YYYY-MM-DD'
    :param end_date: 结束日期，格式为 'YYYY-MM-DD'
    :param freq: 日期间隔，默认为 'D'（天）
    :return: List[pd.DatetimeIndex] 日期序列
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq).to_list()
    return date_range

def generate_date_string_sequence(start_date, end_date, freq='D', format='%Y-%m-%d %H:%M:%S'):
    """
    根据 start_date 和 end_date 生成日期序列
    :param start_date: 起始日期，格式为 'YYYY-MM-DD'
    :param end_date: 结束日期，格式为 'YYYY-MM-DD'
    :param freq: 日期间隔，默认为 'D'（天）
    :return: List[str] 日期序列
    """
    date_range = generate_date_sequence(start_date, end_date, freq)
    new_date_range = [datetime_to_str(date, format=format) for date in date_range]
    return new_date_range

def datetime_to_str(dt, format='%Y-%m-%d %H:%M:%S'):
    if type(dt) == pd.DatetimeIndex:
        return dt.strftime(format)
    elif type(dt) == pd.Timestamp:
        return dt.strftime(format)
    elif type(dt) == str:
        return dt
    elif type(dt) == list:
        return [datetime_to_str(d, format) for d in dt]
    else:
        raise TypeError('Invalid type of datetime')

def str_to_datetime(dt, format='%Y-%m-%d %H:%M:%S'):
    if type(dt) == str:
        return datetime.strptime(dt, format)
    elif type(dt) == pd.Timestamp:
        return dt
    elif type(dt) == pd.DatetimeIndex:
        return dt
    elif type(dt) == list:
        return [str_to_datetime(d, format) for d in dt]
    else:
        raise TypeError('Invalid type of datetime')

def asian_time_to_utc(dt):
    if type(dt) == pd.Timestamp:
        return dt.tz_localize('Asia/Shanghai').tz_convert('UTC')
    elif type(dt) == str:
        return str(pd.to_datetime(dt).tz_localize('Asia/Shanghai').tz_convert('UTC'))

def utc_time_to_asian(dt) -> pd.Timestamp:
    if type(dt) == pd.Timestamp:
        return dt.tz_localize('UTC').tz_convert('Asia/Shanghai')
    elif type(dt) == str:
        return str(pd.to_datetime(dt).tz_localize('UTC').tz_convert('Asia/Shanghai'))

def strdatetime_to_timestamp(dt:str) -> int:
    if re.match(r'^\d{4}-\d{2}-\d{2}$', dt):
        dt = datetime.strptime(dt, '%Y-%m-%d')
        return int(dt.timestamp() * 1000)
    elif re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$', dt):
        dt = datetime.strptime(dt, '%Y-%m-%d %H:%M')
        return int(dt.timestamp() * 1000)
    else:
        raise ValueError('Invalid datetime format')


def is_same_day(date_str1: str, date_str2: str, fmt: str = "%Y-%m-%d %H:%M:%S") -> bool:
    """
    判断两个日期字符串是否是同一天。

    :param date_str1: 第一个日期字符串
    :param date_str2: 第二个日期字符串
    :param fmt: 日期格式，默认为 "%Y-%m-%d %H:%M:%S"
    :return: 如果是同一天返回 True，否则返回 False
    """
    dt1 = datetime.strptime(date_str1, fmt)
    dt2 = datetime.strptime(date_str2, fmt)
    return dt1.date() == dt2.date()

def shift_start_date(start_date: str, n_periods: int, freq: str = 'D') -> str:
    """
    将起始日期向前推 n 个指定频率的时间单位（如天、分钟、小时等）

    :param start_date: 原始起始日期字符串，格式为 "YYYY-MM-DD" 或含时间
    :param n_periods: 要往前推的周期数
    :param freq: 时间频率，支持 'S'(秒), 'T'/'min'(分钟), 'H'(小时), 'D'(天)
    :return: 新的起始日期字符串
    """
    dt = str_to_datetime(start_date)

    # 根据频率计算时间偏移
    if freq in ['D']:
        shifted_dt = dt - timedelta(days=n_periods)
    elif freq in ['H']:
        shifted_dt = dt - timedelta(hours=n_periods)
    elif freq in ['T', 'min']:
        shifted_dt = dt - timedelta(minutes=n_periods)
    elif freq in ['S']:
        shifted_dt = dt - timedelta(seconds=n_periods)
    else:
        raise ValueError(f"Unsupported frequency: {freq}")

    return shifted_dt.strftime("%Y-%m-%d %H:%M:%S")

def shift_end_date(start_date: str, n_periods: int, freq: str = 'D') -> str:
    """
    将起始日期向前推 n 个指定频率的时间单位（如天、分钟、小时等）

    :param start_date: 原始起始日期字符串，格式为 "YYYY-MM-DD" 或含时间
    :param n_periods: 要往前推的周期数
    :param freq: 时间频率，支持 'S'(秒), 'T'/'min'(分钟), 'H'(小时), 'D'(天)
    :return: 新的起始日期字符串
    """
    dt = str_to_datetime(start_date)

    # 根据频率计算时间偏移
    if freq in ['D']:
        shifted_dt = dt + timedelta(days=n_periods)
    elif freq in ['H']:
        shifted_dt = dt + timedelta(hours=n_periods)
    elif freq in ['T', 'min']:
        shifted_dt = dt + timedelta(minutes=n_periods)
    elif freq in ['S']:
        shifted_dt = dt + timedelta(seconds=n_periods)
    else:
        raise ValueError(f"Unsupported frequency: {freq}")

    return shifted_dt.strftime("%Y-%m-%d %H:%M:%S")

def get_trade_dates(start_date: str, end_date: str, n_periods:int, freq: str) -> List[str]:

    if freq == 'T' or freq == 'min':
        period_by_minute = n_periods
    elif freq == 'H':
        period_by_minute = n_periods * 60
    elif freq == 'D':
        period_by_minute = n_periods * 60 * 24

    dates = generate_date_string_sequence(start_date=start_date,
                                          end_date=end_date,
                                          freq='min')
    trade_dates = [dates[date_index] for date_index in range(len(dates) - 1, -1, -period_by_minute)]

    return trade_dates


if __name__ == '__main__':
    date = generate_date_sequence('2020-01-01', '2020-01-05')
    dates = generate_date_string_sequence('2020-01-01', '2020-01-05')
    print(dates)