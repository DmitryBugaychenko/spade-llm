import unittest
from unittest.mock import AsyncMock
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
        mock_storage.put_item.assert_any_call('test:key', None)

if __name__ == "__main__":
    unittest.main()
