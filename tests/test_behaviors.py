import asyncio
import unittest
from unittest.mock import MagicMock, patch
from spade_llm.platform.behaviors import (
    Behaviour,
    BehaviorsOwner,
    MessageTemplate,
    Message,
    ReceiverBehavior, MessageHandlingBehavior,
)
from spade_llm.platform.api import AgentId

class MockAgent(BehaviorsOwner):
    received_behaviour: MessageHandlingBehavior
    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        return asyncio.get_event_loop()
    
    def remove_behaviour(self, beh: Behaviour):
        pass

    def add_behaviour(self, beh: Behaviour):
        self.received_behaviour = beh

class CounterBehavior(Behaviour):
    def __init__(self):
        super().__init__()
        self.counter = 0

    async def step(self):
        print("Making step")
        self.counter += 1

    def is_done(self) -> bool:
        return self.counter >= 3

class TestBehaviours(unittest.TestCase):
    async def execute_behavior(self, behavior: Behaviour):
        # Create instance of MockAgent
        agent = MockAgent()

        # Set up behavior with agent
        await behavior.setup(agent)

        # Schedule execution manually
        await behavior.start()

        # Wait for completion
        await behavior.join()

    def test_counter_behavior(self):
        # Create instance of CounterBehavior
        behavior = CounterBehavior()

        asyncio.run(self.execute_behavior(behavior))
        
        # Check final counter value
        self.assertEqual(behavior.counter, 3)

    async def mock_post_message(self, message: Message):
        """Simulate posting a message"""
        agent = MockAgent()
        receiver = ReceiverBehavior(message.template)
        agent.add_behaviour(receiver)
        await receiver.handle_message(None, message)
        await receiver.join()

    def test_receive_method(self):
        # Prepare mocks and variables
        template = MessageTemplate(performative="inform")
        sender_id = AgentId("sender_1")
        receiver_id = AgentId("receiver_1")
        message = Message(sender=sender_id, receiver=receiver_id, performative="inform", content="Test message")

        # Run the test
        result: Message = asyncio.run(self.receive_and_wait(template, message))
        self.assertIsNotNone(result)
        self.assertEqual(result, message)

    async def receive_and_wait(self, template: MessageTemplate, expected_message: Message):
        agent = MockAgent()
        behavior = Behaviour()
        await behavior.setup(agent)

        task = asyncio.create_task(behavior.receive(template, timeout=5))  # Start receiving task

        # Simulate sending the message after a delay
        await asyncio.sleep(0.1)
        await agent.received_behaviour.handle_message(None, expected_message)

        # Await the result of the receive call
        return await task

if __name__ == "__main__":
    unittest.main()
