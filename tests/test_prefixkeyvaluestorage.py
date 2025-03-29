import unittest
from spade_llm.platform.inmemory import InMemoryKeyValueStorage
from spade_llm.platform.prefixkeyvaluestorage import PrefixKeyValueStorage

class TestPrefixKeyValueStorage(unittest.IsolatedAsyncioTestCase):
    async def test_get_item(self):
        memory_storage = InMemoryKeyValueStorage()
        prefix_storage = PrefixKeyValueStorage(memory_storage, 'test')
        
        await memory_storage.put_item('test:key', 'mock_value')
        
        result = await prefix_storage.get_item('key')
        self.assertEqual(result, 'mock_value')

    async def test_put_item(self):
        memory_storage = InMemoryKeyValueStorage()
        prefix_storage = PrefixKeyValueStorage(memory_storage, 'test')
        
        await prefix_storage.put_item('key', 'value')
        
        result = await memory_storage.get_item('test:key')
        self.assertEqual(result, 'value')

    async def test_close(self):
        memory_storage = InMemoryKeyValueStorage()
        prefix_storage = PrefixKeyValueStorage(memory_storage, 'test')
        
        await prefix_storage.put_item('key', 'value')
        await prefix_storage.close()
        
        result = await memory_storage.get_item('test:_tracked_keys')
        self.assertIsNone(result)
        result = await memory_storage.get_item('test:key')
        self.assertIsNone(result)

    async def test_add_and_remove_tracked_key(self):
        memory_storage = InMemoryKeyValueStorage()
        prefix_storage = PrefixKeyValueStorage(memory_storage, 'test')
        
        await prefix_storage.put_item('key1', 'value1')  
        await prefix_storage.put_item('key2', 'value2')  
        await prefix_storage.close()                     
        
        result = await memory_storage.get_item('test:_tracked_keys')
        self.assertIsNone(result)
        result = await memory_storage.get_item('test:key1')
        self.assertIsNone(result)
        result = await memory_storage.get_item('test:key2')
        self.assertIsNone(result)

if __name__ == "__main__":
    unittest.main()
