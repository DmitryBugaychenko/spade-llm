import asyncio
import unittest
from unittest.mock import MagicMock
from spade_llm.platform.behaviors import Behaviour, BehaviorsOwner

class MockAgent(BehaviorsOwner):
    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        return asyncio.get_event_loop()
    
    def remove_behaviour(self, beh: Behaviour):
        pass

class CounterBehavior(Behaviour):
    def __init__(self):
        super().__init__()
        self.counter = 0

    async def step(self):
        print("Making step")
        self.counter += 1

    def is_done(self):
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

        asyncio.new_event_loop().run_until_complete(self.execute_behavior(behavior))
        
        # Check final counter value
        self.assertEqual(behavior.counter, 3)

if __name__ == "__main__":
    unittest.main()
