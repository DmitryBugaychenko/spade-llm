import asyncio
from typing import Dict, Optional, cast

from spade_llm.core.api import MessageService, MessageSource, Message
from spade_llm.core.conf import configuration, Configurable, ConfigurableRecord, EmptyConfig


class MessageServiceConfig(ConfigurableRecord):
    def create_messaging_service(self) -> MessageService:
        return cast(MessageService, self.create_configurable_instance())

@configuration(EmptyConfig)
class DictionaryMessageService(MessageService, Configurable[EmptyConfig]):
    _sources: Dict[str, "MessageSourceImpl"]

    def __init__(self):
        super().__init__()
        self._sources: Dict[str, MessageSourceImpl] = {}

    async def get_or_create_source(self, agent_type: str) -> MessageSource:
        if agent_type not in self._sources:
            self._sources[agent_type] = MessageSourceImpl(agent_type)
        return self._sources[agent_type]

    async def post_message(self, msg: Message):
        agent_type = msg.receiver.agent_type
        if agent_type not in self._sources:
            raise Exception(f"No message source found for agent type '{agent_type}'")
        source = self._sources[agent_type]
        await source.post_message(msg)

    def post_message_sync(self, msg: Message):
        agent_type = msg.receiver.agent_type
        if agent_type not in self._sources:
            raise Exception(f"No message source found for agent type '{agent_type}'")
        source = self._sources[agent_type]
        source.post_message_sync(msg)


class MessageSourceImpl(MessageSource):
    """Concrete implementation of MessageSource using asyncio.Queue."""

    def __init__(self, agent_type: str, queue_size: int = 50):
        self._agent_type = agent_type
        self.queue = asyncio.Queue(maxsize=queue_size)
        self.shutdown_event = asyncio.Event()
        self._loop = asyncio.get_event_loop()  # Memorize current event loop

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
        self.post_message_sync(message)

    def post_message_sync(self, message: Message):
        if not self.shutdown_event.is_set():
            # Use call_soon_threadsafe to safely enqueue the message
            callback = lambda: asyncio.ensure_future(self.queue.put(message))
            self._loop.call_soon_threadsafe(callback)

    @agent_type.setter
    def agent_type(self, value):
        self._agent_type = value
