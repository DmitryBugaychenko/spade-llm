import unittest
import asyncio
from spade_llm.platform.core import MessageSourceImpl, Message

class TestMessageSourceImpl(unittest.IsolatedAsyncioTestCase):
    async def test_fetch_and_post_message(self):
        ms = MessageSourceImpl('test-agent')
        msg = Message(sender='sender', receiver='receiver', content='Test message')
        
        # Posting a message
        await ms.post_message(msg)
        
        # Fetching the posted message
        retrieved_msg = await ms.fetch_message()
        
        self.assertEqual(retrieved_msg.content, 'Test message')

    async def test_shutdown_and_join(self):
        ms = MessageSourceImpl('test-agent')
        await ms.shutdown()
        await ms.join()

if __name__ == "__main__":
    unittest.main()
