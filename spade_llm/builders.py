import uuid
from typing import Self

from multipledispatch import dispatch
from pydantic import BaseModel

from spade.agent import Agent
from spade.message import Message

from slixmpp.jid import JID

from spade_llm import consts


class MessageBuilder:
    _message: Message

    def __init__(self, performative: str):
        self._message = Message(metadata={consts.PERFORMATIVE : performative})
        # By default start a new stream with message, can be overridden using in_thread
        self._message.thread = str(uuid.uuid4())

    @staticmethod
    def request() -> "MessageBuilder":
        return MessageBuilder(consts.REQUEST)

    @staticmethod
    def request_porposal() -> "MessageBuilder":
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

    @dispatch(str)
    def in_thread(self, thread: str) -> Self:
        self._message.thread = thread
        return self

    @dispatch(uuid.UUID)
    def in_thread(self, thread: uuid.UUID) -> Self:
        self._message.thread = str(thread)
        return self

    @dispatch(str)
    def to_agent(self, to_agent: str) -> Self:
        self._message.to = to_agent
        return self

    @dispatch(JID)
    def to_agent(self, to_agent: JID) -> Self:
        self._message.to = str(to_agent)
        return self

    @dispatch(Agent)
    def to_agent(self, to_agent: Agent) -> Self:
        self._message.to = str(to_agent.jid)
        return self

    @dispatch(str)
    def from_agent(self, from_agent: str) -> Self:
        self._message.sender = from_agent
        return self

    @dispatch(Agent)
    def from_agent(self, from_agent: Agent) -> Self:
        self._message.sender = str(from_agent.jid)
        return self

    @dispatch(JID)
    def from_agent(self, from_agent: JID) -> Self:
        self._message.sender = str(from_agent)
        return self

    @dispatch(str)
    def with_content(self, body: str) -> Message:
        self._message.body = body
        return self._message

    @dispatch(BaseModel)
    def with_content(self, body: BaseModel) -> Message:
        self._message.body = body.model_dump_json()
        return self._message

    def follow_or_create_thread(self, msg) -> Self:
        if msg.thread:
            return self.in_thread(msg.thread)
        else:
            return self.in_thread(str(uuid.uuid4()))