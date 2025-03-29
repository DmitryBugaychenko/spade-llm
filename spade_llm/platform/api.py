import uuid
from abc import ABCMeta, abstractmethod
from typing import Optional, Any

from langchain_core.messages import ToolCall
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class AgentId(BaseModel):
    agent_type: str = Field(description="Name of the agent type used to route message to a proper system."
                                        "Agent of the same type has the same code base and instruction, but might"
                                        "be provided with different message and context.")
    #TODO: Support other type of IDs, including UUID and int
    agent_id: str = Field(description="ID of the agent used to provide context and partition work."
                                      "Usually agent ID correspond to a natural key, for example ID of user.")

class Message(BaseModel):
    sender: AgentId = Field(description="Who sent this message.")
    receiver: AgentId = Field(description="To whom this message id sent.")
    thread_id: Optional[uuid.uuid4] = Field(description="Id of the conversation message belongs to.", default=None)
    performative: str = Field(description="What does the sender mean by this message. Is it a request or just an information.")
    metadata: dict[str, str] = Field(description="Arbitrary extra information about message.", default=dict())
    #TODO: Allow for typed content with raw nested JSON
    content: str = Field(description="Content of the message")


class KeyValueStorage:
    @abstractmethod
    async def get_item(self, key: str) -> str:
        """
        Retrieves an item from agent personal key/value store
        :param key: Key for the item to retrieve
        """
        pass

    @abstractmethod
    async def put_item(self, key: str, value: Optional[str]):
        """
        Sets a new value for an item in agent key/value store
        :param key: Key of the item to set or update
        :param value: Value to put or none to remove the item
        """
        pass


class AgentContext(ABCMeta, KeyValueStorage):
    """
    Provides information for an agent about current context, including access to the agent's key/value
    storage, thread (conversation) context, adds ability to send messages, start and stop threads and
    execute tools filling context-related parameters automatically.
    """

    @property
    @abstractmethod
    def agent_id(self) -> str:
        """
        Identifier of an agent this context belongs to. Used to retrieve proper items from
        key/value store and automatically attached to each message sent via context
        """
        pass

    @property
    @abstractmethod
    def thread_id(self) -> Optional[uuid.uuid4]:
        """
        Thread this context belongs to. Agent can handle multiple threads simultaneously, but
        each message belong to at most one thread. Thread also has an associated key/value cache
        which is automatically cleaned after context is closed.
        """
        pass

    @property
    def has_thread(self) -> bool:
        """
        :return: Whenever this context has a thread associated
        """
        return self.thread_id is not None

    @property
    @abstractmethod
    def thread_context(self) -> KeyValueStorage:
        pass

    @abstractmethod
    async def fork_thread(self) -> "AgentContext":
        """
        Starts a new thread and returns its context.
        """
        pass

    @abstractmethod
    async def close_thread(self) -> "AgentContext":
        """
        Closes thread, cleans up all the thread cache and returns un-threaded context.
        """
        pass

    @abstractmethod
    async def send(self, message: Message):
        """
        Sends a message on behalf of the current agent in current thread (if context belongs to a thread)
        :param message:
        """
        pass

    @property
    @abstractmethod
    def tools(self) -> list[BaseTool]:
        """
        Returns a list of tools available for the agent in current context. These tools are
        already bound to the context, meaning that invoke is traced and context related parameters
        are automatically provided.
        """
        pass


class AgentHandler(ABCMeta):
    """
    Interface provided by the agent to integrate with platform.
    """

    @property
    @abstractmethod
    def agent_type(self) -> str:
        """
        Type of the agent unique in the system. Used as a part of agent address and has
        its own namespace in agent context store.
        """
        pass

    @abstractmethod
    async def handle_message(self, context: AgentContext, message: Message):
        """
        Handles a single message addressed to particular agent in particular thread (optional)
        :param context: Context with access to key/value storage and tool calling
        :param message: Message to handle
        """
        pass

class AgentPlatform(ABCMeta):
    """
    Abstraction of an agent platform. Allows to add agents with their handlers and accessible tools
    """

    @abstractmethod
    async def register_agent(self, handler: AgentHandler, tools: list[BaseTool]):
        """
        Add a new agent to the platform
        :param handler: Message handler for the agent.
        :param tools: Tools available for the agent.
        """
        pass