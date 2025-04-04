import uuid
from abc import ABCMeta, abstractmethod
from typing import Optional, Self, Any

from langchain_core.tools import BaseTool
from multipledispatch import dispatch
from pydantic import BaseModel, Field

from spade_llm import consts


class AgentId(BaseModel):
    agent_type: str = Field(description="Name of the agent type used to route message to a proper system."
                                         "Agent of the same type has the same code base and instruction, but might"
                                         "be provided with different message and context.")
    # TODO: Support other types of IDs, including UUID and int
    agent_id: str = Field(description="ID of the agent used to provide context and partition work."
                                       "Usually agent ID corresponds to a natural key, for example ID of user.")

class Message(BaseModel):
    sender: AgentId = Field(description="Who sent this message.")
    receiver: AgentId = Field(description="To whom this message is sent.")
    thread_id: Optional[uuid.UUID] = Field(description="ID of the conversation message belongs to.", default=None)
    performative: str = Field(description="What does the sender mean by this message. Is it a request or just an information?")
    metadata: dict[str, str] = Field(description="Arbitrary extra information about message.", default=dict())
    # TODO: Allow for typed content with raw nested JSON
    content: str = Field(description="Content of the message")

class MessageBuilder:
    _message: dict[str,Any]

    def __init__(self, performative: str):
        self._message = {consts.PERFORMATIVE : performative}

    @staticmethod
    def request() -> "MessageBuilder":
        return MessageBuilder(consts.REQUEST)

    @staticmethod
    def request_proposal() -> "MessageBuilder":
        return MessageBuilder(consts.REQUEST_PROPOSAL)

    @staticmethod
    def inform() -> "MessageBuilder":
        return MessageBuilder(consts.INFORM)

    @staticmethod
    def acknowledge() -> "MessageBuilder":
        return MessageBuilder(consts.ACKNOWLEDGE)

    @staticmethod
    def failure() -> "MessageBuilder":
        return MessageBuilder(consts.FAILURE)

    @staticmethod
    def propose() -> "MessageBuilder":
        return MessageBuilder(consts.PROPOSE)

    @staticmethod
    def accept() -> "MessageBuilder":
        return MessageBuilder(consts.ACCEPT)

    @staticmethod
    def refuse() -> "MessageBuilder":
        return MessageBuilder(consts.REFUSE)

    @staticmethod
    def reply_with_inform(msg: Message) -> "MessageBuilder":
        return (MessageBuilder
                .inform()
                .to_agent(msg.sender)
                .from_agent(msg.to)
                .in_thread(msg.thread))

    @staticmethod
    def reply_with_ack(msg: Message) -> "MessageBuilder":
        return (MessageBuilder
                .acknowledge()
                .to_agent(msg.sender)
                .from_agent(msg.to)
                .in_thread(msg.thread))

    @staticmethod
    def reply_with_failure(msg: Message) -> "MessageBuilder":
        return (MessageBuilder
                .failure()
                .to_agent(msg.sender)
                .from_agent(msg.to)
                .in_thread(msg.thread))

    @staticmethod
    def reply_with_propose(msg: Message) -> "MessageBuilder":
        return (MessageBuilder
                .propose()
                .to_agent(msg.sender)
                .from_agent(msg.to)
                .in_thread(msg.thread))

    @staticmethod
    def reply_with_accept(msg: Message) -> "MessageBuilder":
        return (MessageBuilder
                .accept()
                .to_agent(msg.sender)
                .from_agent(msg.to)
                .in_thread(msg.thread))

    @staticmethod
    def reply_with_refuse(msg: Message) -> "MessageBuilder":
        return (MessageBuilder
                .refuse()
                .to_agent(msg.sender)
                .from_agent(msg.to)
                .in_thread(msg.thread))

    def in_thread(self, thread: uuid.UUID) -> Self:
        self._message["thread_id"] = thread
        return self

    def to_agent(self, to_agent: AgentId) -> Self:
        self._message["receiver"] = to_agent
        return self

    def from_agent(self, from_agent: AgentId) -> Self:
        self._message["sender"] = from_agent
        return self


    @dispatch(str)
    def with_content(self, body: str) -> Message:
        self._message["content"] = body
        return Message(**self._message)

    @dispatch(BaseModel)
    def with_content(self, body: BaseModel) -> Message:
        self._message["content"] = body.model_dump_json()
        return Message(**self._message)

class KeyValueStorage(metaclass=ABCMeta):
    @abstractmethod
    async def get_item(self, key: str) -> str:
        """
        Retrieves an item from agent personal key/value store
        :param key: Key for the item to retrieve
        """
        pass

    @abstractmethod
    async def put_item(self, key: str, value: Optional[str]) -> None:
        """
        Sets a new value for an item in agent key/value store
        :param key: Key of the item to set or update
        :param value: Value to put or None to remove the item
        """
        pass

    @abstractmethod
    async def close(self):
        """
        Closes connection to the storage. For transient storage clears all the data
        """
        pass

class StorageFactory(metaclass=ABCMeta):
    """
    Used to created storage connections for the agents
    """
    @abstractmethod
    async def create_storage(self, agent_type: str):
        """
        Create a new storage connection
        :param agent_type: Agent type to use connection for
        :return: Storage connection to use for the agent type
        """
        pass

class MessageSource(metaclass=ABCMeta):
    """
    Allows agents to asynchronously fetch messages for certain agent type.
    """
    @property
    @abstractmethod
    def agent_type(self) -> str:
        """
        Name of the agent type this queue belongs to. Only messages for agents of this type
        are fetched from this queue.
        """
        pass

    @abstractmethod
    async def fetch_message(self) -> Optional[Message]:
        """
        Fetches the next message from the queue. Returns a message or None if the queue is drained.
        """
        pass

    @abstractmethod
    async def message_handled(self):
        """
        Notifies the queue that the message has been handled.
        """
        pass

    @abstractmethod
    async def shutdown(self):
        """
        Notifies the queue that it is being shut down. Messages currently in the queue should be processed,
        but no more new messages should arrive.
        """
        pass

    @abstractmethod
    async def join(self):
        """
        Wait for the queue to shut down.
        :return:
        """
        pass

class MessageService(metaclass=ABCMeta):
    """
    Allows agents to connect to the message sources and to send messages.
    """
    @abstractmethod
    async def get_or_create_source(self, agent_type: str) -> MessageSource:
        """
        Creates or returns previously created message source for agent type.
        :param agent_type: Agent type to get messages for.
        :return: Message source to consume messages from.
        """
        pass

    @abstractmethod
    async def post_message(self, msg: Message):
        """
        Posts a new message into one of the registered message sources.
        :param msg: Message to put.
        """
        pass

class AgentContext(KeyValueStorage, metaclass=ABCMeta):
    """
    Provides information for an agent about the current context, including access to the agent's key/value
    storage, thread (conversation) context, adds ability to send messages, start and stop threads, and
    execute tools while filling context-related parameters automatically.
    """

    @property
    @abstractmethod
    def agent_type(self) -> str:
        """
        Type of the agent context is created for. Used to address messages
        """
        pass

    @property
    @abstractmethod
    def agent_id(self) -> str:
        """
        Identifier of an agent this context belongs to. Used to retrieve proper items from
        key/value store and automatically attached to each message sent via context.
        """
        pass

    @property
    @abstractmethod
    def thread_id(self) -> Optional[uuid.UUID]:
        """
        Thread this context belongs to. An agent can handle multiple threads simultaneously, but
        each message belongs to at most one thread. A thread also has an associated key/value cache
        which is automatically cleaned after the context is closed.
        """
        pass

    @property
    def has_thread(self) -> bool:
        """
        :return: Whether this context has a thread associated.
        """
        return self.thread_id is not None

    @property
    @abstractmethod
    def thread_context(self) -> KeyValueStorage:
        """
        For context with thread set returns key value storage associated with the thread
        :return: Key value storage for the thread.
        """
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
        Closes the thread, cleans up all the thread cache, and returns an unthreaded context.
        """
        pass

    @abstractmethod
    async def send(self, message: Message):
        """
        Sends a message on behalf of the current agent in the current thread (if context belongs to a thread).
        :param message:
        """
        pass

    @property
    @abstractmethod
    def tools(self) -> list[BaseTool]:
        """
        Returns a list of tools available for the agent in the current context. These tools are
        already bound to the context, meaning that invoke is traced and context-related parameters
        are automatically provided.
        """
        pass

    def create_message_builder(self, performative: str) -> MessageBuilder:
        """
        Creates a new builder for a message, initializes sender and thread
        """
        bld = MessageBuilder(performative).from_agent(
            AgentId(agent_type = self.agent_type, agent_id = self.agent_id))
        if self.has_thread:
            return bld.in_thread(self.thread_id)
        else:
            return bld

    def inform(self, receiver: AgentId) -> MessageBuilder:
        return self.create_message_builder(consts.INFORM).to_agent(receiver)

    def request(self, receiver: AgentId) -> MessageBuilder:
        return self.create_message_builder(consts.REQUEST).to_agent(receiver)

    def request_proposal(self, receiver: AgentId) -> MessageBuilder:
        return self.create_message_builder(consts.REQUEST_PROPOSAL).to_agent(receiver)

    def propose(self, receiver: AgentId) -> MessageBuilder:
        return self.create_message_builder(consts.PROPOSE).to_agent(receiver)

    def reply_with_inform(self, message: Message) -> MessageBuilder:
        return self.create_message_builder(consts.INFORM).to_agent(message.sender)

    def reply_with_accept(self, message: Message) -> MessageBuilder:
        return self.create_message_builder(consts.ACCEPT).to_agent(message.sender)

    def reply_with_refuse(self, message: Message) -> MessageBuilder:
        return self.create_message_builder(consts.REFUSE).to_agent(message.sender)

    def reply_with_acknowledge(self, message: Message) -> MessageBuilder:
        return self.create_message_builder(consts.ACKNOWLEDGE).to_agent(message.sender)

    def reply_with_failure(self, message: Message) -> MessageBuilder:
        return self.create_message_builder(consts.FAILURE).to_agent(message.sender)





class MessageHandler:
    """
    Abstraction for the entity capable of handling messages
    """
    @abstractmethod
    async def handle_message(self, context: AgentContext, message: Message):
        """
        Handles a single message addressed to a particular agent in a particular thread (optional).
        :param context: Context with access to key/value storage and tool calling.
        :param message: Message to handle.
        """
        pass


class AgentHandler(MessageHandler, metaclass=ABCMeta):
    """
    Interface provided by the agent to integrate with the platform.
    """
    @property
    @abstractmethod
    def agent_type(self) -> str:
        """
        Type of the agent unique in the system. Used as part of the agent address and has
        its own namespace in the agent context store.
        """
        pass

    @abstractmethod
    def start(self, default_context: AgentContext) -> None:
        """
        Starts agent thread and event loop
        :param default_context: Default context for agent.
        """

    @abstractmethod
    def stop(self):
        """
        Stops the agent by stopping the event loop and signaling completion.
        """

    @abstractmethod
    async def join(self):
        """
        Waits for the agent to complete its operations.
        """

    async def shutdown(self):
        """
        Stop agent and wait for its completion
        """
        self.stop()
        await self.join()


class AgentPlatform(metaclass=ABCMeta):
    """
    Abstraction of an agent platform. Allows adding agents with their handlers and accessible tools.
    """
    @abstractmethod
    async def register_agent(self, handler: AgentHandler, tools: list[BaseTool]):
        """
        Adds a new agent to the platform.
        :param handler: Message handler for the agent.
        :param tools: Tools available for the agent.
        """
        pass
