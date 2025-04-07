import asyncio
from typing import Dict, Optional, cast

from pydantic import BaseModel, Field

from spade_llm.core.api import MessageService, MessageSource, Message, MessageBridge, MessageSink
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

class ExposedMessageServiceConfig(BaseModel):
    system_name: str = Field(description="The name of the system used for external messaging.")
    internal: MessageServiceConfig = (
        Field(description="The message source to use internally."))
    bridge: ConfigurableRecord = Field(description="The message bridge to use to fetch messages and pass them to the internal source.")
    exposed_agents: set[str] = Field(
        description="List of agents to listen for messages from external systems."
    )
    external_systems: dict[str, ConfigurableRecord] = Field(
        description="External systems agents can send messages to.")

@configuration(ExposedMessageServiceConfig)
class ExposedMessageSource(MessageService, Configurable[ExposedMessageServiceConfig]):
    internal: MessageService
    bridge: MessageBridge
    external_systems: dict[str, MessageSink]

    def configure(self):
        self.internal = self.config.internal.create_messaging_service()
        self.bridge = self.config.bridge.create_configurable_instance()
        self.external_systems = {name: sink.create_configurable_instance()
                                 for name, sink in self.config.external_systems.items()}

    async def start_bridges(self):
        for sink in self.external_systems.values():
            sink.start()
        await self.bridge.start(self.internal, self.config.exposed_agents)

    async def shutdown(self):
        self.bridge.stop()
        await self.bridge.join()
        await self.internal.shutdown()
        for sink in self.external_systems.values():
            sink.close()

    async def get_or_create_source(self, agent_type: str) -> MessageSource:
        return await self.internal.get_or_create_source(agent_type)

    def get_sink_for_message(self, msg: Message) -> (Message,MessageSink):
        parts = msg.receiver.agent_type.split('@')
        if len(parts) == 1:
            return msg, self.internal

        if len(parts) != 2:
            raise Exception(f"Invalid agent type '{msg.receiver.agent_type}'")

        if parts[-1] in self.external_systems:
            receiver = msg.receiver.model_copy(update={"agent_type": parts[0]})
            sender = msg.sender.model_copy(update={"agent_type": f"{msg.sender.agent_type}@{parts[-1]}"})
            new_msg = msg.model_copy(update={"receiver": receiver, "sender": sender})
            return new_msg, self.external_systems[parts[-1]]

        raise Exception(f"No sink found for agent type '{msg.receiver.agent_type}'")



    async def post_message(self, msg: Message):
        new_msg, sink = self.get_sink_for_message(msg)
        await sink.post_message(new_msg)

    def post_message_sync(self, msg: Message):
        new_msg, sink = self.get_sink_for_message(msg)
        sink.post_message_sync(new_msg)