import unittest
from unittest.mock import AsyncMock, patch
from spade_llm.platform.api import KeyValueStorage
from spade_llm.platform.prefixkeyvaluestorage import PrefixKeyValueStorage

class TestPrefixKeyValueStorage(unittest.IsolatedAsyncioTestCase):
    async def test_get_item(self):
        mock_storage = AsyncMock(spec=KeyValueStorage)
        mock_storage.get_item.return_value = 'mock_value'
        prefix_storage = PrefixKeyValueStorage(mock_storage, 'test')
        
        result = await prefix_storage.get_item('key')
        self.assertEqual(result, 'mock_value')
        mock_storage.get_item.assert_called_once_with('test:key')

    async def test_put_item(self):
        mock_storage = AsyncMock(spec=KeyValueStorage)
        prefix_storage = PrefixKeyValueStorage(mock_storage, 'test')
        
        await prefix_storage.put_item('key', 'value')
        mock_storage.put_item.assert_called_once_with('test:key', 'value')

    async def test_close(self):
        mock_storage = AsyncMock(spec=KeyValueStorage)
        prefix_storage = PrefixKeyValueStorage(mock_storage, 'test')
        
        await prefix_storage.put_item('key', 'value')
        await prefix_storage.close()
        mock_storage.put_item.assert_any_call('test:_tracked_keys', '{"tracked_keys":[]}')
        mock_storage.put_item.assert_any_call('test:key', None)

    async def test_add_and_remove_tracked_key(self):
        mock_storage = AsyncMock(spec=KeyValueStorage)
        prefix_storage = PrefixKeyValueStorage(mock_storage, 'test')
        
        await prefix_storage.put_item('key1', 'value1')  # Adds key1 to tracked keys
        await prefix_storage.put_item('key2', 'value2')  # Adds key2 to tracked keys
        await prefix_storage.close()                     # Removes all tracked keys
        
        mock_storage.put_item.assert_any_call('test:_tracked_keys', '{"tracked_keys":[]}')
        mock_storage.put_item.assert_any_call('test:key1', None)
        mock_storage.put_item.assert_any_call('test:key2', None)

if __name__ == "__main__":
    unittest.main()
