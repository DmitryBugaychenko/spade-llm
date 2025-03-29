import asyncio
from typing import Dict, List

from langchain_core.tools import BaseTool

from spade_llm.platform.api import (
    AgentHandler,
    AgentPlatform,
    KeyValueStorage,
    MessageService,
    MessageSource,
    StorageFactory,
)
from spade_llm.platform.context import AgentContextImpl


class AgentPlatformImpl(AgentPlatform):
    def __init__(self, storage_factory: StorageFactory, message_service: MessageService):
        self.storage_factory = storage_factory
        self.message_service = message_service
        self.agents: Dict[str, AgentHandler] = {}
        self.storages: Dict[str, KeyValueStorage] = {}
        self.run_loop_tasks: Dict[str, asyncio.Task] = {}  # Store tasks per agent type

    async def register_agent(self, handler: AgentHandler, tools: List[BaseTool]):
        agent_type = handler.agent_type
        if agent_type in self.agents:
            raise ValueError(f"An agent with type '{agent_type}' is already registered.")

        self.agents[agent_type] = handler
        self.storages[agent_type] = await self.storage_factory.create_storage(agent_type)

        # Start listening for incoming messages for this agent type
        message_source = await self.message_service.get_or_create_source(agent_type)
        task = asyncio.create_task(self.consume_messages(handler, message_source))
        self.run_loop_tasks[agent_type] = task  # Store task for this agent type

    async def consume_messages(self, handler: AgentHandler, message_source: MessageSource):
        while True:
            message = await message_source.fetch_message()
            if message is None:
                break  # No more messages to process

            # Prepare the context for handling the message
            context = AgentContextImpl(
                kv_store=self.storages[handler.agent_type],
                agent_id=message.receiver.agent_id,
                thread_id=message.thread_id,
                message_service=self.message_service,
            )

            # Pass the message to the agent handler
            await handler.handle_message(context, message)

            # Notify the message source that we have finished processing the message
            await message_source.message_handled()

    async def shutdown(self):
        # Cancel all running tasks
        for task in self.run_loop_tasks.values():
            task.cancel()
        
        # Await cancellation of tasks
        await asyncio.gather(*self.run_loop_tasks.values(), return_exceptions=True)

        # Close storages
        for storage in self.storages.values():
            await storage.close()
