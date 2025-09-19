import pandas as pd
import numpy as np
import threading
import os
import h5py
import urllib.parse
import logging
import sys 

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from data.base import IDataStorage
from sqlalchemy import create_engine, text
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.pool import QueuePool
from typing import Dict
from h5py import special_dtype

class PostgresStorage(IDataStorage):
    def __init__(self, session_factory):
        self.session_factory = session_factory

    def load_data(self, table_name, start_date=None, end_date=None):
        """
        根据表名以及时间段查询数据
        :param table_name: 表名
        :param start_date: 开始时间
        :param end_date: 结束时间   
        """
        session = self.session_factory()
        try:
            # 统一表名大小写为小写
            table_name = table_name.lower()
            
            # 使用小写表名，不需要双引号包裹
            safe_table_name = table_name
            
            query = f"SELECT * FROM {safe_table_name}"
            conditions = []

            if start_date:
                # 使用小写表名
                conditions.append(f"{safe_table_name}.date >= '{start_date}'")
            if end_date:
                conditions.append(f"{safe_table_name}.date <= '{end_date}'")
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            return pd.read_sql(query, session.connection())
        finally:
            # 确保会话关闭
            session.close()
        
    def save_data(self, data, table_name, if_exists='fail', index=False):
        """
        存储数据到数据库
        :param data: 需要存储的数据
        :param table_name: 数据库表名
        :param if_exists: 存储方式，默认为fail，表示如果表已经存在，则不存储数据
        """
        session = self.session_factory()
        try:
            # 统一表名大小写为小写
            table_name = table_name.lower()
            
            # 检查表是否存在
            table_exists = session.connection().dialect.has_table(
                session.connection(), table_name
            )
            
            # 如果表不存在，强制使用'append'模式（会自动创建表）
            effective_mode = 'append' if not table_exists else if_exists
            
            # 保存数据
            data.to_sql(
                name=table_name,
                con=session.connection(),
                if_exists=effective_mode,
                index=index
            )
            
            # 关键：显式提交事务
            session.commit()
        except Exception as e:
            # 出错时回滚
            session.rollback()
            raise e
        finally:
            # 确保会话关闭
            session.close()


class HDF5Storage(IDataStorage):
    """HDF5数据库存储"""
    _instance = None

    def __init__(self, file_factory):
        self.original_file_path = '/Users/fu/Desktop/QuantSystem/h5_data'
        self.storage_path = os.path.join(self.original_file_path, file_factory)
          
    def _open_file(self):
        # 确保存储目录存在
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        return h5py.File(self.storage_path, 'a')
    
    def create_file_factory(self):
        self._open_file()

    def _convert_to_hdf5_compatible(self, data, index=True):
        """
        转换数据为HDF5兼容格式，保留原始数据类型。
        将所有非数值类型转换为字符串以确保兼容性。
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Unsupported data type: {type(data)}")

        # Work on a copy to avoid SettingWithCopyWarning
        df = data.copy()

        if index and df.index.name is not None:
            df = df.reset_index()

        dtypes = []
        for col_name in df.columns:
            col_dtype = df[col_name].dtype
            # If it's a numeric type, keep it.
            if np.issubdtype(col_dtype, np.number):
                dtypes.append((col_name, col_dtype))
                # Ensure NaNs are properly represented if any
                if df[col_name].isnull().any():
                    df[col_name] = df[col_name].fillna(np.nan)
            # Otherwise, convert it to a variable-length string for HDF5.
            else:
                dtypes.append((col_name, special_dtype(vlen=str)))
                # Explicitly convert the column in the DataFrame to string
                df[col_name] = df[col_name].astype(str).fillna('')

        # Create the structured array from the *modified* DataFrame
        rec_array = np.rec.fromarrays(df.values.T, dtype=dtypes)
        return rec_array, dtypes


    def load_data(self, table_name, start_date=None, end_date=None, 
                 columns=None, as_dataframe=True):
        """从HDF5加载数据"""
        with self._open_file() as f:
            if table_name not in f:
                raise KeyError(f"Dataset {table_name} not found")
                
            dataset = f[table_name]
            data = dataset[:]

            if as_dataframe:
                # 直接使用dtype中的列名
                data = pd.DataFrame(data, columns=dataset.dtype.names)
                
                if columns:
                    missing_cols = [col for col in columns if col not in data.columns]
                    if missing_cols:
                        raise ValueError(f"Columns not found: {missing_cols}")
                    data = data[columns]

                for col in data.columns:
                    # 检查列是否为字节字符串
                    if data[col].dtype == object and len(data[col]) > 0 and isinstance(data[col].iloc[0], bytes):
                        # 使用向量化方法转换所有值为字符串
                        data[col] = data[col].str.decode('utf-8')
                
                # 时间范围过滤
                if 'datetime' in data.columns:
                    if not pd.api.types.is_datetime64_any_dtype(data['datetime']):
                        data['datetime'] = pd.to_datetime(data['datetime'])
                if start_date:
                    data = data[data['datetime'] >= pd.to_datetime(start_date)]
                if end_date:
                    data = data[data['datetime'] <= pd.to_datetime(end_date)]
            return data
        
    def save_data(self, data: pd.DataFrame, table_name, if_exists='fail', 
                  format='table', index=True):
        """保存数据到HDF5"""
        with self._open_file() as f:
            exists = table_name in f
            data_rec, dtype = self._convert_to_hdf5_compatible(data, index=index)

            try:
                if exists:
                    if if_exists == 'fail':
                        raise ValueError(f"Table {table_name} already exists")
                    elif if_exists == 'replace':
                        del f[table_name]
                        f.create_dataset(table_name, data=data_rec, dtype=dtype)
                    elif if_exists == 'append':
                        # 获取现有数据集并追加数据
                        ds = f[table_name]
                        new_size = ds.shape[0] + len(data_rec)
                        ds.resize(new_size, axis=0)
                        ds[-len(data_rec):] = data_rec
                    else:
                        raise ValueError(f"Invalid if_exists: '{if_exists}'")
                else:
                    # 创建可扩展数据集
                    maxshape = (None,) if if_exists == 'append' else None
                    f.create_dataset(
                        table_name, 
                        data=data_rec, 
                        dtype=dtype,
                        maxshape=maxshape
                    )
            except Exception as e:
                raise RuntimeError(f"Failed to save data: {e}")
        
class StorageFactory():
    """数据库工厂模式"""
    @staticmethod
    def create_storage(db_type, session_factory=None, file_factory=None):
        if db_type == 'postgresql':
            if session_factory is None:
                raise ValueError('session_factory is None')
            return PostgresStorage(session_factory)
        elif db_type == 'hdf5':
            return HDF5Storage(file_factory)
        else:
            raise ValueError(f"Invalid database type: {db_type}")

class DataStorageFacade():
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._engines = {}
                cls._instance._session_factories = {}
                cls._instance._storages = {}
                cls._instance.config = set()
        return cls._instance
    
    def _read_db_config(self, db_type, config_file='db_config.ini'):
        """
        读取数据库配置文件
        :param db_type: 数据库类型
        :param config_file: 配置文件
        :return:
        """
        if not os.path.exists(config_file):
            raise FileNotFoundError(config_file)
        
        import configparser
        config = configparser.ConfigParser()
        config.read(config_file)

        if db_type not in config:
            raise ValueError(f"Not found db_type {db_type} in config file {config_file}")
        
        db_config = config[db_type]
        return {
            'host': db_config.get('host'),
            'user': db_config.get('user'),
            'password': db_config.get('password'),
            'port': db_config.get('port'),
        }
    
    def _add_engines(self, alias, db_type, database, 
                     pool_size, max_overflow, pool_recycle):
        # 读取配置文件
        config_dict = self._read_db_config(db_type)
        user = config_dict.get('user')
        password = config_dict.get('password')
        host = config_dict.get('host')
        port = config_dict.get('port')

        if db_type == 'postgresql':
            user = urllib.parse.quote_plus(user)
            password = urllib.parse.quote_plus(password)
            conn_str = f'postgresql://{user}:{password}@{host}:{port}/{database}'
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
        
        if conn_str in self.config: #避免重复添加
            print(f"Using existing connection string: {db_type}-{database}")
            #logger
            return None

        engine = create_engine(
            conn_str,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_recycle=pool_recycle,
        )
        self._engines[alias] = engine
        self.config.add(conn_str)

        return engine
    
    def add_database(self, db_type, database, 
                     pool_size=5, max_overflow=10, pool_recycle=3600):
        """
        添加数据库配置
        :param alias: 唯一标识符
        :param db_type: 数据库类型 (postgresql, mysql, sqlite等)
        :param kwargs: 数据库特定参数
        """
        alias = f'{db_type}_{database}'
        engine = self._add_engines(alias, db_type, database, pool_size, max_overflow, pool_recycle)
        
        if engine is not None:
            self._session_factories[alias] = scoped_session(sessionmaker(bind=engine, autocommit=False, autoflush=False))
            self._storages[alias] = StorageFactory.create_storage(db_type, session_factory=self._session_factories[alias])
    
    def add_file_storage(self, db_type, file_factory):
        alias = f'{db_type}_{file_factory}'
        self._storages[alias] = StorageFactory.create_storage(db_type, file_factory=file_factory)
        return self._storages[alias]
    
    def get_storage(self, alias):
        """获取数据库存储对象"""
        if alias not in self._storages:
            raise KeyError("Storage alias '%s' not found" % alias)
        return self._storages[alias]
    
    def get_session(self, alias):
        """直接获取数据库会话（高级使用）"""
        if alias not in self._session_factories:
            raise KeyError("Session alias '%s' not found" % alias)
        return self._session_factories[alias]()
    
    def close_all(self):
        """关闭所有连接池"""
        for alias, factory in self._session_factories.items():
            factory.remove()  # 移除所有会话
            self._engines[alias].dispose()  # 关闭连接池
        
        self._engines.clear()
        self._session_factories.clear()
        self._storages.clear()
        self.config.clear()


if __name__ == '__main__':
    data = pd.read_csv('/Users/fu/Desktop/量化/股票数据/688787.SH.csv')
    data['datetime'] = data['trade_date']
    data_operator = DataStorageFacade()
    data_operator.add_file_storage('hdf5', 'BTCUSDT').create_file_factory()
