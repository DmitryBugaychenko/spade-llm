import uuid
from uuid import UUID
from typing import List, Optional

from langchain_core.tools import BaseTool

from spade_llm.platform.api import KeyValueStorage, MessageService, Message, AgentContext
from spade_llm.platform.storage import PrefixKeyValueStorage, TransientKeyValueStorage


class AgentContextImpl(AgentContext):

    def __init__(self, kv_store: KeyValueStorage, agent_id: str, thread_id: Optional[UUID], message_service: MessageService, tools: List[BaseTool] = []):
        self.kv_store = kv_store
        self.agent_id = agent_id
        self.thread_id = thread_id
        self.message_service = message_service
        self.tools = tools
        self._thread_kv_store: Optional[PrefixKeyValueStorage] = None  

    # Remaining methods stay unchanged...
