from uuid import UUID
from typing import List, Optional
from .api import KeyValueStorage, MessageService, Message, AgentContext, BaseTool

class ConcreteAgentContext(AgentContext):
    def __init__(self, kv_store: KeyValueStorage, agent_id: str, thread_id: Optional[UUID], message_service: MessageService):
        self.kv_store = kv_store
        self.agent_id = agent_id
        self.thread_id = thread_id
        self.message_service = message_service
        self.tools: List[BaseTool] = []

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
            return self.kv_store
        raise RuntimeError("Thread context unavailable because there's no active thread.")

    async def fork_thread(self) -> "ConcreteAgentContext":
        new_thread_id = UUID(int=random.getrandbits(128))  # Generate random UUID
        new_context = ConcreteAgentContext(self.kv_store, self.agent_id, new_thread_id, self.message_service)
        return new_context

    async def close_thread(self) -> "ConcreteAgentContext":
        self.thread_id = None
        return self

    async def send(self, message: Message):
        await self.message_service.send_message(message)

    @property
    def tools(self) -> List[BaseTool]:
        return self.tools

    def add_tool(self, tool: BaseTool):
        self.tools.append(tool)
