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

    def test_handle_message(self):
        template = MessageTemplate(performative='test')
        mock_behaviour = MockBehavior(template)
        mock_behaviour.handle_message = AsyncMock()
        self.dispatcher.add_behaviour(mock_behaviour)
        context_mock = MagicMock()
        message = Message(performative='test', content='')
        self.loop = asyncio.get_event_loop()
        self.loop.run_until_complete(self.dispatcher.handle_message(context_mock, message))
        mock_behaviour.handle_message.assert_called_once_with(context_mock, message)

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

if __name__ == "__main__":
    unittest.main()
