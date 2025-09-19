import pandas as pd
from abc import ABC, abstractmethod

class IDataProvider():
    @abstractmethod
    def get_data(self):
        pass

class IDataStorage():
    @abstractmethod
    def save_data(self):
        pass

    @abstractmethod
    def load_data(self):
        pass

class IDataRetriever():
    @abstractmethod
    def get_history_data(self, symbol='BTCUSDT', interval='1min', start_date=None, end_date=None, delay=1) -> pd.DataFrame:
        pass

    def get_realtime_data(self, symbol='BTCUSDT', interval='1min', start_date=None) -> pd.DataFrame:
        pass