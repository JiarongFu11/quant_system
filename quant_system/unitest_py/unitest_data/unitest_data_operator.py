import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from data.data_operator import DataOperator
from data.data_storage import DataStorageFacade

def create_data_engine():
    data_storage = DataStorageFacade()
    data_storage.add_file_storage(db_type='hdf5', file_factory='BTCUSDT')
    data_operator = DataOperator(data_storage=data_storage, db_type='hdf5')

    return data_operator

def test_get_security_min_data():
    data_operator = create_data_engine()
    data_operator.get_security_min_data(start_date='2025-06-21 00:01:00', end_date='2025-06-21 02:02:00', symbol='BTCUSDT')

if __name__ == '__main__':
    test_get_security_min_data()