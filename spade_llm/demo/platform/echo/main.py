import asyncio
from aioconsole import ainput
from spade_llm.platform.api import AgentHandler, Message, AgentId
from spade_llm.platform.platform import AgentPlatformImpl
from spade_llm.platform.storage import InMemoryStorageFactory
from spade_llm.platform.messaging import DictionaryMessageService

class EchoAgentHandler(AgentHandler):
    @property
    def agent_type(self) -> str:
        return "echo"

    async def handle_message(self, context, message: Message):
        print(f"EchoAgent received message: {message.content}")

async def main():
    # Initialize the platform components
    storage_factory = InMemoryStorageFactory()
    message_service = DictionaryMessageService()
    platform = AgentPlatformImpl(storage_factory, message_service)

    # Register the echo agent
    echo_handler = EchoAgentHandler()
    await platform.register_agent(echo_handler, [])

    # Main loop to read user input and send messages
    while True:
        user_input = await ainput("Enter a message to send to the agent (type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break

        # Construct a message and send it to the agent
        message = Message(
            sender=AgentId(agent_type="console", agent_id="user"),
            receiver=AgentId(agent_type="echo", agent_id="echo_agent"),
            performative="inform",
            content=user_input
        )
        await message_service.post_message(message)

    # Shut down the platform gracefully
    await platform.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
