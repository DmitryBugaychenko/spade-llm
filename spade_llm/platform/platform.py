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
from spade_llm.platform.storage import PrefixKeyValueStorage


class AgentPlatformImpl(AgentPlatform):
    def __init__(self, storage_factory: StorageFactory, message_service: MessageService):
        self.storage_factory = storage_factory
        self.message_service = message_service
        self.agents: Dict[str, AgentHandler] = {}
        self.storages: Dict[str, KeyValueStorage] = {}
        self.tools_by_agent: Dict[str, List[BaseTool]] = {}
        self.run_loop_tasks: Dict[str, asyncio.Task] = {}
        self.message_sources: Dict[str, MessageSource] = {}  # New dictionary to hold message sources

    async def register_agent(self, handler: AgentHandler, tools: List[BaseTool]):
        agent_type = handler.agent_type
        if agent_type in self.agents:
            raise ValueError(f"An agent with type '{agent_type}' is already registered.")

        self.agents[agent_type] = handler
        self.storages[agent_type] = await self.storage_factory.create_storage(agent_type)
        self.tools_by_agent[agent_type] = tools

        # Create and store the message source
        message_source = await self.message_service.get_or_create_source(agent_type)
        self.message_sources[agent_type] = message_source

        # Start consuming messages
        task = asyncio.create_task(self.consume_messages(handler, message_source))
        self.run_loop_tasks[agent_type] = task

    async def consume_messages(self, handler: AgentHandler, message_source: MessageSource):
        while True:
            message = await message_source.fetch_message()
            if message is None:
                break

            agent_id = message.receiver.agent_id
            kv_store = PrefixKeyValueStorage(self.storages[handler.agent_type], agent_id)
            tools = self.tools_by_agent.get(handler.agent_type, [])

            context = AgentContextImpl(kv_store, agent_id, message.thread_id, self.message_service, tools=tools)
            await handler.handle_message(context, message)
            await message_source.message_handled()

    async def shutdown(self):
        # Shutdown all message sources
        for message_source in self.message_sources.values():
            await message_source.shutdown()

        # Join all message sources
        for message_source in self.message_sources.values():
            await message_source.join()

        # Cancel all run loop tasks
        for task in self.run_loop_tasks.values():
            task.cancel()

        # Await cancellation of tasks
        await asyncio.gather(*self.run_loop_tasks.values(), return_exceptions=True)

        # Close storages
        for storage in self.storages.values():
            await storage.close()
