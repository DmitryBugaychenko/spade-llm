import uuid
from uuid import UUID
from typing import List, Optional

from langchain_core.tools import BaseTool

from spade_llm.platform.api import KeyValueStorage, MessageService, Message, AgentContext
from spade_llm.platform.storage import PrefixKeyValueStorage


class AgentContextImpl(AgentContext):

    def __init__(self, kv_store: KeyValueStorage, agent_id: str, thread_id: Optional[UUID], message_service: MessageService):
        self.kv_store = kv_store
        self.agent_id = agent_id
        self.thread_id = thread_id
        self.message_service = message_service
        self.tools: List[BaseTool] = []
        self._thread_kv_store: Optional[PrefixKeyValueStorage] = None  # New attribute to hold the thread-specific KV store

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @agent_id.setter
    def agent_id(self, value: str):
        self._agent_id = value

    @property
    def thread_id(self) -> Optional[UUID]:
        return self._thread_id

    @thread_id.setter
    def thread_id(self, value: Optional[UUID]):
        self._thread_id = value

    @property
    def has_thread(self) -> bool:
        return self.thread_id is not None

    @property
    def thread_context(self) -> KeyValueStorage:
        if self.has_thread:
            if not self._thread_kv_store:
                self._thread_kv_store = PrefixKeyValueStorage(wrapped_storage=self.kv_store, prefix=str(self.thread_id))
            return self._thread_kv_store
        raise RuntimeError("Thread context unavailable because there's no active thread.")

    async def fork_thread(self) -> "AgentContextImpl":
        new_thread_id = uuid.uuid4()
        new_context = AgentContextImpl(self.kv_store, self.agent_id, new_thread_id, self.message_service)
        return new_context

    async def close_thread(self) -> "AgentContextImpl":
        if self._thread_kv_store:
            await self._thread_kv_store.close()
            self._thread_kv_store = None
        self.thread_id = None
        return self

    async def send(self, message: Message):
        await self.message_service.post_message(message)

    async def get_item(self, key: str) -> str:
        return await self.kv_store.get_item(key)

    async def put_item(self, key: str, value: Optional[str]) -> None:
        await self.kv_store.put_item(key, value)

    async def close(self):
        await self.kv_store.close()

    @property
    def tools(self) -> List[BaseTool]:
        return self.tools

    @tools.setter
    def tools(self, value):
        self._tools = value
