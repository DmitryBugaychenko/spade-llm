import uuid
from typing import List, Optional
from uuid import UUID

from langchain_core.tools import BaseTool

from spade_llm.core.api import KeyValueStorage, MessageService, Message, AgentContext
from spade_llm.core.models import ModelsProvider
from spade_llm.core.storage import PrefixKeyValueStorage, TransientKeyValueStorage


class AgentContextImpl(AgentContext):

    def __init__(self,
                 kv_store: KeyValueStorage,
                 agent_type: str,
                 agent_id: str,
                 thread_id: Optional[UUID],
                 message_service: MessageService,
                 tools: List[BaseTool] = None,
                 model_provider: ModelsProvider=None):

        if tools is None:
            tools = []
        self.kv_store = kv_store
        self._agent_id = agent_id
        self._agent_type = agent_type
        self._thread_id = thread_id
        self._message_service = message_service
        self._tools = tools
        self._model_provider = model_provider  # Store the model provider
        self._thread_kv_store: Optional[PrefixKeyValueStorage] = None

    @property
    def agent_type(self) -> str:
        return self._agent_type

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def thread_id(self) -> Optional[UUID]:
        return self._thread_id

    @property
    def has_thread(self) -> bool:
        return self.thread_id is not None

    @property
    def thread_context(self) -> KeyValueStorage:
        if self.has_thread:
            if not self._thread_kv_store:
                self._thread_kv_store = (
                    TransientKeyValueStorage(
                        PrefixKeyValueStorage(wrapped_storage=self.kv_store, prefix=str(self.thread_id))))
            return self._thread_kv_store
        raise RuntimeError("Thread context unavailable because there's no active thread.")

    async def fork_thread(self) -> "AgentContextImpl":
        new_thread_id = uuid.uuid4()
        new_context = AgentContextImpl(
            self.kv_store, self.agent_type, self.agent_id, new_thread_id,
            self._message_service, self._tools, self._model_provider)
        return new_context

    async def close_thread(self) -> "AgentContextImpl":
        if self._thread_kv_store:
            await self._thread_kv_store.close()
            self._thread_kv_store = None
        self._thread_id = None
        return self

    async def send(self, message: Message):
        await self._message_service.post_message(message)

    async def get_item(self, key: str) -> str:
        return await self.kv_store.get_item(key)

    async def put_item(self, key: str, value: Optional[str]) -> None:
        await self.kv_store.put_item(key, value)

    async def close(self):
        await self.kv_store.close()

    @property
    def tools(self) -> List[BaseTool]:
        return self._tools

    def create_chat_model(self, name):
        return self._model_provider.create_chat_model(name)

    def create_embeddings_model(self, name):
        return self._model_provider.create_embeddings_model(name)
