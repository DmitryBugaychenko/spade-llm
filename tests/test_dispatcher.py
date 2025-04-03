import logging
import unittest
import uuid
from unittest.mock import MagicMock
from spade_llm.core.agent import PerformativeDispatcher, Message, \
    MessageHandlingBehavior, ThreadDispatcher
from spade_llm.core.api import AgentId
from spade_llm.core.behaviors import MessageTemplate


class MockBehavior(MessageHandlingBehavior):
    async def step(self):
        pass

    def __init__(self, template: MessageTemplate):
        super().__init__(template)
        self._template = template
        self._done = False

    @property
    def template(self) -> MessageTemplate:
        return self._template

    def is_done(self) -> bool:
        return self._done

    async def handle_message(self, context, message):
        pass

class TestPerformativeDispatcher(unittest.TestCase):
    def setUp(self):
        self.dispatcher = PerformativeDispatcher(logging.getLogger("test"))

    def test_add_and_remove_behaviour(self):
        template = MessageTemplate(performative='test')
        mock_behaviour = MockBehavior(template)
        self.dispatcher.add_behaviour(mock_behaviour)
        self.assertTrue('test' in self.dispatcher._behaviors_by_performative)
        self.assertEqual(len(self.dispatcher._behaviors_by_performative['test']), 1)
        self.dispatcher.remove_behaviour(mock_behaviour)
        self.assertFalse('test' in self.dispatcher._behaviors_by_performative)

    def test_is_empty(self):
        # Initially dispatcher should be empty
        self.assertTrue(self.dispatcher.is_empty)

        # Adding a behavior makes it non-empty
        template = MessageTemplate(performative='test')
        mock_behaviour = MockBehavior(template)
        self.dispatcher.add_behaviour(mock_behaviour)
        self.assertFalse(self.dispatcher.is_empty)

        # Removing the behavior makes it empty again
        self.dispatcher.remove_behaviour(mock_behaviour)
        self.assertTrue(self.dispatcher.is_empty)

    def test_find_matching_behaviour_single_match(self):
        template = MessageTemplate(performative='test')
        mock_behaviour = MockBehavior(template)
        self.dispatcher.add_behaviour(mock_behaviour)
        
        # Use AgentId for sender and receiver
        sender = AgentId(agent_type="TestSender", agent_id="1")
        receiver = AgentId(agent_type="TestReceiver", agent_id="2")
        
        message = Message(
            sender=sender,
            receiver=receiver,
            performative='test',
            content=''
        )
        
        # Synchronous comprehension to gather results
        found_behaviours = [item for item in self.dispatcher.find_matching_behaviour(message)]
        self.assertEqual(len(found_behaviours), 1)
        self.assertIn(mock_behaviour, found_behaviours)

    def test_multiple_matching_behaviors(self):
        # Create two behaviors with different templates but both matching 'test' performative
        template_1 = MessageTemplate(performative='test')
        template_2 = MessageTemplate(performative='test')
        mock_behaviour_1 = MockBehavior(template_1)
        mock_behaviour_2 = MockBehavior(template_2)
        self.dispatcher.add_behaviour(mock_behaviour_1)
        self.dispatcher.add_behaviour(mock_behaviour_2)

        # Prepare a message matching both behaviors
        sender = AgentId(agent_type="TestSender", agent_id="1")
        receiver = AgentId(agent_type="TestReceiver", agent_id="2")
        message = Message(sender=sender, receiver=receiver, performative='test', content='')

        # Synchronous comprehension to gather results
        found_behaviours = [item for item in self.dispatcher.find_matching_behaviour(message)]
        self.assertEqual(len(found_behaviours), 2)
        self.assertIn(mock_behaviour_1, found_behaviours)
        self.assertIn(mock_behaviour_2, found_behaviours)


class TestThreadDispatcher(unittest.TestCase):
    def setUp(self):
        self.dispatcher = ThreadDispatcher(logging.getLogger("test"))
        self.thread_id_1 = uuid.uuid4()  # Use UUID instead of string
        self.thread_id_2 = uuid.uuid4()
        self.msg_template_1 = MessageTemplate(thread_id=self.thread_id_1)
        self.msg_template_2 = MessageTemplate(thread_id=self.thread_id_2)
        self.behavior_mock_1 = MagicMock(spec=MessageHandlingBehavior)
        self.behavior_mock_1.template = self.msg_template_1
        self.behavior_mock_2 = MagicMock(spec=MessageHandlingBehavior)
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
            content="Test message",
            thread_id=self.thread_id_1,
            performative="inform"
        )
        result = list(self.dispatcher.find_matching_behaviour(msg))
        self.assertListEqual(result, [self.behavior_mock_1])

    def test_is_empty(self):
        self.assertTrue(self.dispatcher.is_empty)
        self.dispatcher.add_behaviour(self.behavior_mock_1)
        self.assertFalse(self.dispatcher.is_empty)
        self.dispatcher.remove_behaviour(self.behavior_mock_1)
        self.assertTrue(self.dispatcher.is_empty)

if __name__ == "__main__":
    unittest.main()
