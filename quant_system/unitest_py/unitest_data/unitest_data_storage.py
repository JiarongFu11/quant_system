import os
import sys

from datetime import datetime
from sqlalchemy.pool import QueuePool


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from data.data_storage import PostgresStorage, HDF5Storage, StorageFactory, DataStorageFacade

import unittest
from unittest.mock import patch, MagicMock, mock_open
import tempfile
import shutil
import pandas as pd

# --- Helper function to create virtual data ---
def create_virtual_data():
    """Creates a standard pandas DataFrame for testing."""
    return pd.DataFrame({
        'datetime': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04']),
        'value': [10.5, 11.0, 12.2, 13.0],
        'category': ['A', 'B', 'A', 'C'],
        'id': [1, 2, 3, 4]
    })

# --- Test Cases ---

class TestPostgresStorage(unittest.TestCase):
    
    def setUp(self):
        """Set up mock objects for each test."""
        self.mock_session_factory = MagicMock()
        self.mock_session = MagicMock()
        self.mock_connection = MagicMock()
        self.mock_dialect = MagicMock()
        
        self.mock_session_factory.return_value = self.mock_session
        self.mock_session.connection.return_value = self.mock_connection
        self.mock_connection.dialect = self.mock_dialect
        
        self.storage = PostgresStorage(self.mock_session_factory)
        self.data = create_virtual_data().rename(columns={'datetime': 'date'})

    @patch('pandas.read_sql')
    def test_load_data_full(self, mock_read_sql):
        """Test loading data without any filters."""
        mock_read_sql.return_value = self.data
        table_name = "MyTable"
        
        df = self.storage.load_data(table_name)
        
        self.mock_session.close.assert_called_once()
        mock_read_sql.assert_called_once()
        # Check if table name is lowercased and query is correct
        self.assertEqual(mock_read_sql.call_args[0][0], f"SELECT * FROM {table_name.lower()}")
        pd.testing.assert_frame_equal(df, self.data)

    @patch('pandas.read_sql')
    def test_load_data_with_date_range(self, mock_read_sql):
        """Test loading data with a start and end date."""
        table_name = "test_data"
        start_date = "2023-01-01"
        end_date = "2023-01-03"
        
        self.storage.load_data(table_name, start_date, end_date)
        
        expected_query = (
            f"SELECT * FROM {table_name} WHERE "
            f"{table_name}.date >= '{start_date}' AND {table_name}.date <= '{end_date}'"
        )
        self.assertEqual(mock_read_sql.call_args[0][0], expected_query)
        self.mock_session.close.assert_called_once()

    @patch('pandas.DataFrame.to_sql')
    def test_save_data_new_table(self, mock_to_sql):
        """Test saving data to a new table."""
        self.mock_dialect.has_table.return_value = False # Table does not exist
        
        self.storage.save_data(self.data, "new_table", if_exists='append')
        
        mock_to_sql.assert_called_once_with(
            name='new_table',
            con=self.mock_connection,
            if_exists='append', # Should be forced to append for new tables
            index=False
        )
        self.mock_session.commit.assert_called_once()
        self.mock_session.rollback.assert_not_called()
        self.mock_session.close.assert_called_once()

    @patch('pandas.DataFrame.to_sql')
    def test_save_data_replace_existing(self, mock_to_sql):
        """Test saving data by replacing an existing table."""
        self.mock_dialect.has_table.return_value = True # Table exists
        
        self.storage.save_data(self.data, "existing_table", if_exists='replace')
        
        mock_to_sql.assert_called_once_with(
            name='existing_table',
            con=self.mock_connection,
            if_exists='replace',
            index=False
        )
        self.mock_session.commit.assert_called_once()
        self.mock_session.close.assert_called_once()

    @patch('pandas.DataFrame.to_sql')
    def test_save_data_fail_on_error(self, mock_to_sql):
        """Test that rollback is called on an exception."""
        mock_to_sql.side_effect = ValueError("DB error")
        self.mock_dialect.has_table.return_value = False
        
        with self.assertRaises(ValueError):
            self.storage.save_data(self.data, "error_table")
            
        self.mock_session.commit.assert_not_called()
        self.mock_session.rollback.assert_called_once()
        self.mock_session.close.assert_called_once()


class TestHDF5Storage(unittest.TestCase):

    def setUp(self):
        """Create a temporary directory for HDF5 files."""
        self.temp_dir = tempfile.mkdtemp()
        self.file_factory_name = 'test_storage.h5'
        self.storage = HDF5Storage(self.file_factory_name)
        # Override storage path to use the temporary directory
        self.storage.storage_path = os.path.join(self.temp_dir, self.file_factory_name)
        self.data = create_virtual_data()

    def tearDown(self):
        """Remove the temporary directory and its contents."""
        shutil.rmtree(self.temp_dir)

    def test_save_and_load_data(self):
        """Test saving data and then loading it back."""
        table_name = 'stock_data'
        self.storage.save_data(self.data, table_name)
        
        loaded_data = self.storage.load_data(table_name)
        
        # Datetime is saved as string, so we convert it back for comparison
        loaded_data['datetime'] = pd.to_datetime(loaded_data['datetime'])
        
        pd.testing.assert_frame_equal(self.data, loaded_data)

    def test_save_if_exists_fail(self):
        """Test 'fail' mode for if_exists parameter."""
        table_name = 'data'
        self.storage.save_data(self.data, table_name)
        with self.assertRaises(ValueError):
            self.storage.save_data(self.data, table_name, if_exists='fail')

    def test_save_if_exists_replace(self):
        """Test 'replace' mode for if_exists parameter."""
        table_name = 'data'
        self.storage.save_data(self.data.head(2), table_name)
        self.storage.save_data(self.data, table_name, if_exists='replace')
        
        loaded_data = self.storage.load_data(table_name)
        self.assertEqual(len(loaded_data), len(self.data))

    def test_save_if_exists_append(self):
        """Test 'append' mode for if_exists parameter."""
        table_name = 'data'
        part1 = self.data.head(2)
        part2 = self.data.tail(2)
        
        self.storage.save_data(part1, table_name, if_exists='append')
        self.storage.save_data(part2, table_name, if_exists='append')
        
        loaded_data = self.storage.load_data(table_name)
        self.assertEqual(len(loaded_data), len(self.data))
        # Verify the content is correct after append
        pd.testing.assert_frame_equal(self.data.reset_index(drop=True), loaded_data.reset_index(drop=True))


    def test_load_data_with_date_filter(self):
        """Test loading data with start and end date filters."""
        table_name = 'timed_data'
        self.storage.save_data(self.data, table_name)
        
        start_date = '2023-01-02'
        end_date = '2023-01-03'
        
        loaded_data = self.storage.load_data(table_name, start_date=start_date, end_date=end_date)
        
        self.assertEqual(len(loaded_data), 2)
        self.assertEqual(loaded_data['datetime'].min(), pd.to_datetime(start_date))
        self.assertEqual(loaded_data['datetime'].max(), pd.to_datetime(end_date))

    def test_load_non_existent_table(self):
        """Test loading from a table that doesn't exist."""
        with self.assertRaises(KeyError):
            self.storage.load_data('non_existent_table')


class TestStorageFactory(unittest.TestCase):

    def test_create_postgres_storage(self):
        """Test creation of PostgresStorage."""
        mock_factory = MagicMock()
        storage = StorageFactory.create_storage('postgresql', session_factory=mock_factory)
        self.assertIsInstance(storage, PostgresStorage)

    def test_create_postgres_storage_no_factory(self):
        """Test that creating PostgresStorage without a factory raises an error."""
        with self.assertRaises(ValueError):
            StorageFactory.create_storage('postgresql')

    def test_create_hdf5_storage(self):
        """Test creation of HDF5Storage."""
        storage = StorageFactory.create_storage('hdf5', file_factory='test.h5')
        self.assertIsInstance(storage, HDF5Storage)

    def test_create_invalid_storage(self):
        """Test that an invalid type raises an error."""
        with self.assertRaises(ValueError):
            StorageFactory.create_storage('mongodb')


class TestDataStorageFacade(unittest.TestCase):

    def setUp(self):
        """Reset the singleton instance before each test."""
        # This is a common pattern for testing singletons
        DataStorageFacade._instance = None
        self.facade = DataStorageFacade()

    def tearDown(self):
        """Ensure the facade is cleaned up."""
        self.facade.close_all()
        DataStorageFacade._instance = None
    
    def test_singleton_instance(self):
        """Test that the facade is a singleton."""
        instance1 = DataStorageFacade()
        instance2 = DataStorageFacade()
        self.assertIs(instance1, instance2)

    @patch('data.data_storage.os.path.exists', return_value=True)
    @patch('data.data_storage.configparser.ConfigParser')
    def test_read_db_config_success(self, mock_config_parser, mock_exists):
        """Test successful reading of a config file."""
        mock_config = MagicMock()
        mock_config.__contains__.side_effect = lambda item: item == 'postgresql'
        mock_config['postgresql'].get.side_effect = lambda key: f'test_{key}'
        mock_config_parser.return_value.read.return_value = None
        mock_config_parser.return_value = mock_config

        config = self.facade._read_db_config('postgresql')
        self.assertEqual(config['host'], 'test_host')
        self.assertEqual(config['user'], 'test_user')

    @patch('data.data_storage.create_engine')
    @patch.object(DataStorageFacade, '_read_db_config')
    @patch('data.data_storage.urllib.parse.quote_plus', side_effect=lambda x: x)
    def test_add_database(self, mock_quote, mock_read_config, mock_create_engine):
        """Test adding a new database configuration."""
        mock_read_config.return_value = {
            'user': 'testuser', 'password': 'pw', 'host': 'localhost', 'port': '5432'
        }
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        
        self.facade.add_database('postgresql', 'test_db')
        
        alias = 'postgresql_test_db'
        self.assertIn(alias, self.facade._engines)
        self.assertIn(alias, self.facade._session_factories)
        self.assertIn(alias, self.facade._storages)
        self.assertIsInstance(self.facade.get_storage(alias), PostgresStorage)
        
        # Verify engine was created with correct params
        expected_conn_str = 'postgresql://testuser:pw@localhost:5432/test_db'
        mock_create_engine.assert_called_once_with(
            expected_conn_str,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_recycle=3600
        )

    def test_add_file_storage(self):
        """Test adding a new file storage."""
        storage = self.facade.add_file_storage('hdf5', 'my_data.h5')
        alias = 'hdf5_my_data.h5'
        
        self.assertIn(alias, self.facade._storages)
        self.assertIsInstance(self.facade.get_storage(alias), HDF5Storage)
        self.assertIs(storage, self.facade.get_storage(alias))

    def test_get_storage_not_found(self):
        """Test that getting a non-existent storage raises a KeyError."""
        with self.assertRaises(KeyError):
            self.facade.get_storage('non_existent_alias')
            
    @patch('data.data_storage.create_engine')
    @patch.object(DataStorageFacade, '_read_db_config')
    def test_close_all(self, mock_read_config, mock_create_engine):
        """Test that close_all disposes engines and clears dictionaries."""
        mock_read_config.return_value = {
            'user': 'user', 'password': 'pw', 'host': 'host', 'port': '5432'
        }
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        
        self.facade.add_database('postgresql', 'db1')
        self.facade.add_file_storage('hdf5', 'file1.h5')
        
        self.facade.close_all()
        
        mock_engine.dispose.assert_called_once()
        self.assertEqual(len(self.facade._engines), 0)
        self.assertEqual(len(self.facade._session_factories), 0)
        self.assertEqual(len(self.facade._storages), 0)
        self.assertEqual(len(self.facade.config), 0)


if __name__ == '__main__':
    # The original file's main block is replaced by the test runner
    # We need to add the current directory to sys.path for the imports inside the classes to work
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
         sys.path.insert(0, current_dir)
         
    # To run the tests, we can simply use unittest.main()
    unittest.main(argv=['first-arg-is-ignored'], exit=False)