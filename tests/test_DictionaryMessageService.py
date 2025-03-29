import unittest
import asyncio
from spade_llm.platform.core import DictionaryMessageService, Message

class TestDictionaryMessageService(unittest.IsolatedAsyncioTestCase):
    async def test_get_or_create_source(self):
        dms = DictionaryMessageService()
        source = await dms.get_or_create_source('test-agent-type')
        self.assertIsNotNone(source)
        self.assertEqual(source.agent_type, 'test-agent-type')

    async def test_post_and_retrieve_message(self):
        dms = DictionaryMessageService()
        source = await dms.get_or_create_source('test-agent-type')
        msg = Message(sender='sender', receiver='receiver', content='Test message')
        
        # Posting a message
        await dms.post_message(msg)
        
        # Retrieving the message through the source
        retrieved_msg = await source.fetch_message()
        
        self.assertEqual(retrieved_msg.content, 'Test message')

if __name__ == "__main__":
    unittest.main()
