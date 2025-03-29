import unittest
from unittest.mock import MagicMock
from spade_llm.platform.api import AgentId, Message
from spade_llm.platform.agent import ThreadDispatcher, MessageTemplate, Behaviour

class TestThreadDispatcher(unittest.TestCase):
    def setUp(self):
        self.dispatcher = ThreadDispatcher()
        self.thread_id_1 = 'thread-id-1'
        self.thread_id_2 = 'thread-id-2'
        self.msg_template_1 = MessageTemplate(thread_id=self.thread_id_1)
        self.msg_template_2 = MessageTemplate(thread_id=self.thread_id_2)
        self.behavior_mock_1 = MagicMock(spec=Behaviour)
        self.behavior_mock_1.template = self.msg_template_1
        self.behavior_mock_2 = MagicMock(spec=Behaviour)
        self.behavior_mock_2.template = self.msg_template_2

    def test_add_and_remove_behaviour(self):
        self.assertEqual(len(self.dispatcher._dispatchers_by_thread), 0)
        self.dispatcher.add_behaviour(self.behavior_mock_1)
        self.assertIn(self.thread_id_1, self.dispatcher._dispatchers_by_thread)
        self.assertEqual(len(self.dispatcher._dispatchers_by_thread[self.thread_id_1]._behaviors_by_performative.get(None)), 1)
        self.dispatcher.remove_behaviour(self.behavior_mock_1)
        self.assertNotIn(self.thread_id_1, self.dispatcher._dispatchers_by_thread)

    def test_find_matching_behaviour(self):
        self.dispatcher.add_behaviour(self.behavior_mock_1)
        sender_id = AgentId(agent_type='test-sender', agent_id='sender-agent')
        receiver_id = AgentId(agent_type='test-receiver', agent_id='receiver-agent')
        msg = Message(
            sender=sender_id,
            receiver=receiver_id,
            body="Test message",
            thread_id=self.thread_id_1,
            performative="inform"
        )
        result = self.dispatcher.find_matching_behaviour(msg)
        self.assertIs(result, self.behavior_mock_1)

    def test_is_empty(self):
        self.assertTrue(self.dispatcher.is_empty)
        self.dispatcher.add_behaviour(self.behavior_mock_1)
        self.assertFalse(self.dispatcher.is_empty)
        self.dispatcher.remove_behaviour(self.behavior_mock_1)
        self.assertTrue(self.dispatcher.is_empty)

if __name__ == "__main__":
    unittest.main()
