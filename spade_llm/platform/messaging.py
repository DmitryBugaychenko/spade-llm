import asyncio
from typing import Dict, Optional

from spade_llm.platform.api import MessageService, MessageSource, Message


class DictionaryMessageService(MessageService):
    def __init__(self):
        self.sources: Dict[str, MessageSourceImpl] = {}

    async def get_or_create_source(self, agent_type: str) -> MessageSource:
        if agent_type not in self.sources:
            self.sources[agent_type] = MessageSourceImpl(agent_type)
        return self.sources[agent_type]

    async def post_message(self, msg: Message):
        agent_type = msg.receiver.agent_type
        if agent_type not in self.sources:
            raise Exception(f"No message source found for agent type '{agent_type}'")  # Threw exception when source not found
        source = self.sources[agent_type]
        await source.post_message(msg)


class MessageSourceImpl(MessageSource):
    """Concrete implementation of MessageSource using asyncio.Queue."""

    def __init__(self, agent_type: str, queue_size: int = 50):  # Added optional constructor param
        self._agent_type = agent_type
        self.queue = asyncio.Queue(maxsize=queue_size)
        self.shutdown_event = asyncio.Event()

    @property
    def agent_type(self) -> str:
        return self._agent_type

    async def fetch_message(self) -> Optional[Message]:
        try:
            return await self.queue.get()
        except asyncio.CancelledError:
            return None

    async def message_handled(self):
        self.queue.task_done()

    async def shutdown(self):
        self.shutdown_event.set()
        await self.join()

    async def join(self):
        await self.queue.join()

    async def post_message(self, message: Message):
        if not self.shutdown_event.is_set():
            await self.queue.put(message)

    @agent_type.setter
    def agent_type(self, value):
        self._agent_type = value
