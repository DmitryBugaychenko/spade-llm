import asyncio
import logging
from spade_llm.core.agent import Agent
from spade_llm.core.behaviors import Behaviour

# Configure logging
logging.basicConfig(level=logging.INFO)


class HelloWorldBehavior(Behaviour):
    """
    A simple one-shot behavior that prints 'Hello World' when executed.
    """
    
    async def run(self):
        """
        Run the behavior - prints Hello World once.
        """
        print("Hello World!")
        self._is_done = True  # Mark as done for one-shot execution


class HelloWorldAgent(Agent):
    """
    A simple Hello World agent.
    """
    
    def setup(self):
        """
        Setup method to initialize the agent with behaviors.
        """
        # Add the Hello World behavior
        hello_behavior = HelloWorldBehavior()
        self.add_behaviour(hello_behavior)


async def main():
    """
    Main function to run the Hello World agent.
    """
    # Create the agent
    agent = HelloWorldAgent(agent_type="hello_world_agent")
    
    # Note: You'll need to provide a proper AgentContext to start the agent
    # For now, this is a template showing how to structure the agent
    
    print("Hello World Agent created successfully!")
    print("To run this agent, you need to provide a proper AgentContext.")


if __name__ == "__main__":
    asyncio.run(main())
