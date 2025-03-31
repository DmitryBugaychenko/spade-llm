import unittest
from unittest.mock import AsyncMock, MagicMock
from spade_llm.platform.agent import PerformativeDispatcher, Behaviour, MessageTemplate, Message, \
    MessageHandlingBehavior
from spade_llm.platform.api import AgentId

class MockBehavior(MessageHandlingBehavior):
    def __init__(self, template: MessageTemplate):
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
        self.dispatcher = PerformativeDispatcher()

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

    def test_find_matching_behaviour(self):
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
        
        found_behaviour = self.dispatcher.find_matching_behaviour(message)
        self.assertIsNotNone(found_behaviour)
        self.assertEqual(found_behaviour, mock_behaviour)

    async def test_handle_message(self):
        template = MessageTemplate(performative='test')
        mock_behaviour = MockBehavior(template)
        mock_behaviour.handle_message = AsyncMock()
        self.dispatcher.add_behaviour(mock_behaviour)
        context_mock = MagicMock()
        message = Message(performative='test', content='')
        await self.dispatcher.handle_message(context_mock, message)
        mock_behaviour.handle_message.assert_called_once_with(context_mock, message)

if __name__ == "__main__":
    unittest.main()
