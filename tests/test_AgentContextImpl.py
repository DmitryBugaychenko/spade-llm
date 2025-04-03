import unittest
from unittest.mock import AsyncMock
from uuid import UUID

from spade_llm.core.api import KeyValueStorage, MessageService, Message, AgentId
from spade_llm.core.context import AgentContextImpl


class TestAgentContextImpl(unittest.IsolatedAsyncioTestCase):
    async def test_fork_thread_creates_new_context_with_new_thread_id(self):
        kv_store_mock = AsyncMock(spec=KeyValueStorage)
        message_service_mock = AsyncMock(spec=MessageService)
        original_context = AgentContextImpl(kv_store_mock, "original_agent_id", UUID("123e4567-e89b-12d3-a456-426655440000"), message_service_mock)
        
        new_context = await original_context.fork_thread()
        
        self.assertNotEqual(new_context.thread_id, original_context.thread_id)
        self.assertEqual(new_context.agent_id, original_context.agent_id)
        self.assertIsInstance(new_context.thread_id, UUID)

    async def test_close_thread_clears_thread_id_and_closes_prefix_storage(self):
        kv_store_mock = AsyncMock(spec=KeyValueStorage)
        message_service_mock = AsyncMock(spec=MessageService)
        original_context = AgentContextImpl(kv_store_mock, "original_agent_id", UUID("123e4567-e89b-12d3-a456-426655440000"), message_service_mock)
        thread_context_mock = AsyncMock()
        original_context._thread_kv_store = thread_context_mock
        
        await original_context.close_thread()
        
        self.assertIsNone(original_context.thread_id)
        self.assertTrue(thread_context_mock.close.called)

    async def test_send_posts_message_via_message_service(self):
        kv_store_mock = AsyncMock(spec=KeyValueStorage)
        message_service_mock = AsyncMock(spec=MessageService)

        context = AgentContextImpl(kv_store_mock, "test_agent_id", None, message_service_mock)

        sender = AgentId(agent_type="sender-type", agent_id="sender-id")
        receiver = AgentId(agent_type="receiver-type", agent_id="receiver-id")

        message = Message(sender=sender, receiver=receiver, content="Test message", performative="inform")
        
        await context.send(message)
        
        message_service_mock.post_message.assert_awaited_once_with(message)

    async def test_get_item_delegates_to_kv_store(self):
        kv_store_mock = AsyncMock(spec=KeyValueStorage)
        message_service_mock = AsyncMock(spec=MessageService)
        context = AgentContextImpl(kv_store_mock, "test_agent_id", None, message_service_mock)
        
        result = await context.get_item("some_key")
        
        kv_store_mock.get_item.assert_awaited_once_with("some_key")
        self.assertEqual(result, kv_store_mock.get_item.return_value)

    async def test_put_item_delegates_to_kv_store(self):
        kv_store_mock = AsyncMock(spec=KeyValueStorage)
        message_service_mock = AsyncMock(spec=MessageService)
        context = AgentContextImpl(kv_store_mock, "test_agent_id", None, message_service_mock)
        
        await context.put_item("some_key", "some_value")
        
        kv_store_mock.put_item.assert_awaited_once_with("some_key", "some_value")

if __name__ == "__main__":
    unittest.main()
