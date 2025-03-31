import asyncio
import unittest
from spade_llm.platform.behaviors import Behaviour

class CounterBehavior(Behaviour):
    def __init__(self):
        super().__init__()
        self.counter = 0

    async def step(self):
        self.counter += 1

    def is_done(self):
        return self.counter >= 3

class TestBehaviours(unittest.TestCase):
    def test_counter_behavior(self):
        # Create instance of CounterBehavior
        behavior = CounterBehavior()
        
        # Schedule execution manually
        behavior.start()
        
        # Wait for completion
        asyncio.run(behavior.join())
        
        # Check final counter value
        self.assertEqual(behavior.counter, 3)

if __name__ == "__main__":
    unittest.main()
