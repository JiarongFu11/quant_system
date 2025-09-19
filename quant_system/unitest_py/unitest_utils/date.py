import unittest
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.date import is_same_day, shift_start_date, shift_end_date, get_trade_dates
from datetime import datetime

class TestDateUtils(unittest.TestCase):

    def test_is_same_day(self):
        self.assertTrue(is_same_day("2025-07-10 09:00:00", "2025-07-10 15:00:00"))
        self.assertFalse(is_same_day("2025-07-10 23:59:00", "2025-07-11 00:01:00"))

    def test_shift_start_date_daily(self):
        self.assertEqual(shift_start_date("2025-07-10 00:00:00", 2, 'D'), "2025-07-08 00:00:00")
        self.assertEqual(shift_start_date("2025-07-10 14:30:00", 1, 'D'), "2025-07-09 14:30:00")

    def test_shift_start_date_hourly(self):
        self.assertEqual(shift_start_date("2025-07-10 14:30:00", 3, 'H'), "2025-07-10 11:30:00")

    def test_shift_start_date_minutes(self):
        self.assertEqual(shift_start_date("2025-07-10 14:30:00", 15, 'T'), "2025-07-10 14:15:00")

    def test_shift_start_date_seconds(self):
        self.assertEqual(shift_start_date("2025-07-10 14:30:45", 30, 'S'), "2025-07-10 14:30:15")

    def test_shift_end_date_daily(self):
        self.assertEqual(shift_end_date("2025-07-10 00:00:00", 2, 'D'), "2025-07-12 00:00:00")
        self.assertEqual(shift_end_date("2025-07-10 14:30:00", 1, 'D'), "2025-07-11 14:30:00")

    def test_shift_end_date_hourly(self):
        self.assertEqual(shift_end_date("2025-07-10 14:30:00", 3, 'H'), "2025-07-10 17:30:00")

    def test_shift_end_date_minutes(self):
        self.assertEqual(shift_end_date("2025-07-10 14:30:00", 15, 'T'), "2025-07-10 14:45:00")

    def test_shift_end_date_seconds(self):
        self.assertEqual(shift_end_date("2025-07-10 14:30:45", 30, 'S'), "2025-07-10 14:31:15")

    def test_get_trade_dates(self):
        # 测试日频数据获取最近 3 天
        dates = get_trade_dates("2025-07-08 09:00:00", "2025-07-10 15:00:00", 1, 'D')
        expected = [
            '2025-07-10 15:00:00',
            '2025-07-09 15:00:00',
            '2025-07-08 15:00:00'
        ]
        self.assertListEqual(dates, expected)

        # 测试分钟级数据获取最近 30 分钟
        dates_min = get_trade_dates("2025-07-10 14:00:00", "2025-07-10 15:00:00", 30, 'T')
        expected_min = [
            '2025-07-10 15:00:00',
            '2025-07-10 14:30:00',
            '2025-07-10 14:00:00',
        ]
        self.assertListEqual(dates_min, expected_min)


if __name__ == '__main__':
    unittest.main()