import asyncio
import logging
from spade_llm.core.agent import Agent
from spade_llm.core.behaviors import MessageHandlingBehavior
from spade_llm.core.api import Message, AgentContext

# Configure logging
logging.basicConfig(level=logging.INFO)

class HelloWorldBehavior(MessageHandlingBehavior):
    """
    A simple behavior that prints 'Hello World' when executed.
    """
    
    async def handle_message(self, context: AgentContext, message: Message):
        print("Hello World!")
        # You can also reply to the sender
        # await context.send_message(to=message.sender, body="Hello World!")


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
    
    # Create a simple context (this would need a real implementation in production)
    # For demonstration purposes, we'll just start the agent
    # In a real scenario, you would provide a proper AgentContext
    
    # Note: You'll need to implement or provide a real AgentContext
    # For now, this is a template showing how to structure the agent
    
    print("Hello World Agent created successfully!")
    print("To run this agent, you need to provide a proper AgentContext.")


if __name__ == "__main__":
    asyncio.run(main())
