import asyncio
import unittest
from spade_llm.platform.behaviors import (
    Behaviour,
    BehaviorsOwner,
    MessageTemplate,
    Message,
    MessageHandlingBehavior,
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
        beh.setup(self)

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
        behavior.setup(agent)

        # Schedule execution manually
        behavior.start()

        # Wait for completion
        await behavior.join()

    def test_counter_behavior(self):
        # Create instance of CounterBehavior
        behavior = CounterBehavior()

        asyncio.run(self.execute_behavior(behavior))
        
        # Check final counter value
        self.assertEqual(behavior.counter, 3)

    def test_receive_method(self):
        # Prepare mocks and variables
        template = MessageTemplate(performative="inform")
        sender_id = AgentId(agent_type='test-sender', agent_id='sender-agent')
        receiver_id = AgentId(agent_type='test-receiver', agent_id='receiver-agent')
        message = Message(sender=sender_id, receiver=receiver_id, performative="inform", content="Test message")

        # Run the test
        result: Message = asyncio.run(self.receive_and_wait(template, message))
        self.assertIsNotNone(result)
        self.assertEqual(result, message)

    async def receive_and_wait(self, template: MessageTemplate, expected_message: Message):
        agent = MockAgent()
        behavior = CounterBehavior()
        behavior.setup(agent)

        task = asyncio.create_task(behavior.receive(template, timeout=5000))  # Start receiving task

        # Simulate sending the message after a delay
        await asyncio.sleep(0.1)
        await agent.received_behaviour.handle_message(None, expected_message)

        # Await the result of the receive call
        return await task

if __name__ == "__main__":
    unittest.main()
