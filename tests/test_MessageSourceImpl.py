import unittest

from spade_llm.platform.api import AgentId
from spade_llm.platform.api import Message
from spade_llm.platform.messaging import MessageSourceImpl


class TestMessageSourceImpl(unittest.IsolatedAsyncioTestCase):
    async def test_fetch_and_post_message(self):
        ms = MessageSourceImpl('test-agent')
        
        sender = AgentId(agent_type="sender-type", agent_id="sender-id")
        receiver = AgentId(agent_type="receiver-type", agent_id="receiver-id")
        
        msg = Message(sender=sender, receiver=receiver, content='Test message', performative="inform")
        
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
