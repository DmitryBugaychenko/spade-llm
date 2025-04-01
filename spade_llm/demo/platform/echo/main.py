import asyncio
from aioconsole import ainput
import logging

from spade_llm.platform.agent import Agent
from spade_llm.platform.api import Message, AgentId
from spade_llm.platform.behaviors import MessageHandlingBehavior, MessageTemplate, ContextBehaviour
from spade_llm.platform.platform import AgentPlatformImpl
from spade_llm.platform.storage import InMemoryStorageFactory
from spade_llm.platform.messaging import DictionaryMessageService

# Enable logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EchoAgentHandler(Agent):
    def __init__(self):
        super().__init__("echo")

    class EchoBehaviour(MessageHandlingBehavior):
        async def step(self):
            print(f"\nEchoAgent received message: {self.message.content}")

    class InputBehavior(ContextBehaviour):
        async def step(self):
            user_input = await ainput("Enter a message to send to the agent (type 'exit' to quit): ")
            if user_input.lower() == 'exit':
                self.agent.stop()
            else:
                # Construct a message and send it to the agent
                message = Message(
                    sender=AgentId(agent_type="console", agent_id="user"),
                    receiver=AgentId(agent_type="echo", agent_id="echo_agent"),
                    performative="inform",
                    content=user_input)

                await self.context.send(message)


    def setup(self):
        self.add_behaviour(self.EchoBehaviour(MessageTemplate()))
        self.add_behaviour(self.InputBehavior(self.default_context))



async def main():
    # Initialize the platform components
    storage_factory = InMemoryStorageFactory()
    message_service = DictionaryMessageService()
    platform = AgentPlatformImpl(storage_factory, message_service)

    # Register the echo agent
    echo_handler = EchoAgentHandler()
    await platform.register_agent(echo_handler, [])

    # # Main loop to read user input and send messages
    # while True:
    #     user_input = await ainput("Enter a message to send to the agent (type 'exit' to quit): ")
    #     if user_input.lower() == 'exit':
    #         break
    #
    #     # Construct a message and send it to the agent
    #     message = Message(
    #         sender=AgentId(agent_type="console", agent_id="user"),
    #         receiver=AgentId(agent_type="echo", agent_id="echo_agent"),
    #         performative="inform",
    #         content=user_input
    #     )
    #     await message_service.post_message(message)

    await echo_handler.join()
    # Shut down the platform gracefully
    await platform.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
