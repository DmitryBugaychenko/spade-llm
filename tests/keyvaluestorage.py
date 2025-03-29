import unittest
from spade_llm.platform.storage import InMemoryKeyValueStorage, PrefixKeyValueStorage, TransientKeyValueStorage

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
        
        await prefix_storage.close()

class TestTransientKeyValueStorage(unittest.IsolatedAsyncioTestCase):
    async def test_transient_behavior(self):
        memory_storage = InMemoryKeyValueStorage()
        transient_storage = TransientKeyValueStorage(memory_storage)
        
        await transient_storage.put_item('key1', 'value1')
        await transient_storage.put_item('key2', 'value2')
        
        result1 = await transient_storage.get_item('key1')
        result2 = await transient_storage.get_item('key2')
        
        self.assertEqual(result1, 'value1')
        self.assertEqual(result2, 'value2')
        
        await transient_storage.close()
        
        result_after_closing_1 = await memory_storage.get_item('key1')
        result_after_closing_2 = await memory_storage.get_item('key2')
        
        self.assertIsNone(result_after_closing_1)
        self.assertIsNone(result_after_closing_2)

if __name__ == "__main__":
    unittest.main()
