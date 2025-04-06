import logging
import uuid
from abc import ABCMeta, abstractmethod
import asyncio
from typing import Optional, Callable

from spade_llm import consts
from spade_llm.core.api import MessageHandler, AgentContext, Message


class BehaviorsOwner(metaclass=ABCMeta):
    """
    Abstraction for an entity holding behaviours. Used to avoid circular references.
    """
    @property
    @abstractmethod
    def loop(self) -> asyncio.AbstractEventLoop:
        """
        Event loop used for steps execution.
        """
        pass

    @abstractmethod
    def remove_behaviour(self, beh: "Behaviour"):
        """
        Removes behavior from the list
        :param beh: Behaviour to remove
        """
        pass

    @abstractmethod
    def add_behaviour(self, beh: "Behaviour"):
        """
        Add behavior to the list and configures to use this container
        :param beh: Behavior to add
        """
        pass

class MessageTemplate:
    """
    Templates are used to filter messages and dispatch them to proper behavior
    """
    def __init__(self,
                 thread_id: Optional[uuid.UUID] = None,
                 performative: Optional[str] = None,
                 validator: Optional[Callable[[Message], bool]] = None):
        """
        Initializes the MessageTemplate with optional thread_id, performative, and validator.

        Args:
            thread_id (Optional[uuid.UUID], optional): The thread identifier. Defaults to None.
            performative (Optional[str], optional): The performative string. Defaults to None.
            validator (Optional[Callable[[Message], bool]], optional): Lambda function validating the message. Defaults to None.
        """
        self._thread_id = thread_id
        self._performative = performative
        self._validator = validator

    @property
    def thread_id(self) -> Optional[uuid.UUID]:
        """
        Gets the thread id if provided.
        """
        return self._thread_id

    @property
    def performative(self) -> Optional[str]:
        """
        Gets the performative if provided.
        """
        return self._performative

    def match(self, msg: Message) -> bool:
        """
        Checks whether the given message matches this template.

        Args:
            msg (Message): The message to check.

        Returns:
            bool: True if the message matches the template, False otherwise.
        """
        if self._thread_id is not None and msg.thread_id != self._thread_id:
            return False
        if self._performative is not None and msg.performative != self._performative:
            return False
        if self._validator is not None and not self._validator(msg):
            return False
        return True

    @classmethod
    def inform(cls) -> "MessageTemplate":
        """
        Returns a MessageTemplate for the INFORM message type.
        """
        return cls(performative=consts.INFORM)

    @classmethod
    def request(cls) -> "MessageTemplate":
        """
        Returns a MessageTemplate for the REQUEST message type.
        """
        return cls(performative=consts.REQUEST)

    @classmethod
    def request_proposal(cls) -> "MessageTemplate":
        """
        Returns a MessageTemplate for the REQUEST_PROPOSAL message type.
        """
        return cls(performative=consts.REQUEST_PROPOSAL)

    @classmethod
    def request_approval(cls) -> "MessageTemplate":
        """
        Returns a MessageTemplate for the REQUEST_APPROVAL message type.
        """
        return cls(performative=consts.REQUEST_APPROVAL)

    @classmethod
    def propose(cls) -> "MessageTemplate":
        """
        Returns a MessageTemplate for the PROPOSE message type.
        """
        return cls(performative=consts.PROPOSE)

    @classmethod
    def accept(cls) -> "MessageTemplate":
        """
        Returns a MessageTemplate for the ACCEPT message type.
        """
        return cls(performative=consts.ACCEPT)

    @classmethod
    def refuse(cls) -> "MessageTemplate":
        """
        Returns a MessageTemplate for the REFUSE message type.
        """
        return cls(performative=consts.REFUSE)

    @classmethod
    def acknowledge(cls) -> "MessageTemplate":
        """
        Returns a MessageTemplate for the ACKNOWLEDGE message type.
        """
        return cls(performative=consts.ACKNOWLEDGE)

    @classmethod
    def failure(cls) -> "MessageTemplate":
        """
        Returns a MessageTemplate for the FAILURE message type.
        """
        return cls(performative=consts.FAILURE)

    @staticmethod
    def from_agent(agent_type: str) -> Callable[[Message], bool]:
        def validator(msg: Message) -> bool:
            return msg.sender.agent_type == agent_type
        return validator


class Behaviour(metaclass=ABCMeta):
    """
    Reusable code block for the agent, consumes a message matching certain template and
    handles it
    """
    _agent: BehaviorsOwner
    _logger: logging.Logger
    _is_done: bool = False

    def __init__(self):
        self._completion_event = asyncio.Event()

    @property
    def agent(self) -> BehaviorsOwner:
        """
        Readonly property returning the associated agent.
        """
        return self._agent

    def setup(self, agent: BehaviorsOwner):
        """
        Setup method to initialize the behavior with its associated agent.
        """
        self._agent = agent
        if hasattr(self.agent, "agent_type"):
            self._logger = logging.getLogger(agent.agent_type + "." + self.__class__.__name__)
        else:
            self._logger = logging.getLogger(self.__class__.__name__)

    def start(self):
        """
        Start behavior by scheduling its execution
        """
        self._is_done = False
        self.schedule_execution()

    def schedule_execution(self):
        """
        Schedules execution using agents event loop
        """
        callback = lambda: asyncio.ensure_future(self._run())
        self.agent.loop.call_soon(callback)

    @abstractmethod
    async def step(self):
        """
        Executes a single step of this behaviour. Steps should not block for IO (use asyncio instead),
        nor perform long computations (offload them using an executor).
        """

    async def _run(self):
        """
        Internal method, executes single step and then either removes behavior or schedules next
        execution depending on whether behavior is done or not.
        """
        await self.step()
        if self.is_done():
            self._completion_event.set()
            self.agent.remove_behaviour(self)
        else:
            self.schedule_execution()

    async def join(self):
        """
        Waits for the behavior to complete before continuing.
        """
        await self._completion_event.wait()

    def is_done(self) -> bool:
        """
        Returns True if behavior is completed and should not accept messages anymore,
        False otherwise.
        """
        return self._is_done

    def set_is_done(self):
        """
        Used by inheritors to signal that this behavior is done.
        """
        self._is_done = True
    
    async def receive(self, template: MessageTemplate, timeout: float) -> Optional[Message]:
        """
        Waits for the message matching template for given time.
        :param template: Template for messages to wait for
        :param timeout: Maximum waiting time
        :return: Message if it was received and false otherwise
        """

        receiver = ReceiverBehavior(template)
        self.agent.add_behaviour(receiver)
        
        try:
            await asyncio.wait_for(receiver.join(), timeout)
            return receiver.message
        except asyncio.TimeoutError:
            self.agent.remove_behaviour(receiver)
            return None

    @property
    def logger(self) -> logging.Logger:
        return self._logger


class ContextBehaviour(Behaviour, metaclass=ABCMeta):
    """
    This implementation provides access to agent context for the behavior.
    """
    _context: AgentContext

    def __init__(self, context: Optional[AgentContext] = None):
        """
        Initialize ContextBehaviour optionally passing an initial context.
        :param context: Initial context object (optional).
        """
        super().__init__()
        if context is not None:
            self.context = context

    @property
    def context(self) -> AgentContext:
        """
        Property providing read/write access to the agent context.
        """
        return self._context

    @context.setter
    def context(self, value: AgentContext):
        self._context = value


class MessageHandlingBehavior(ContextBehaviour, MessageHandler, metaclass=ABCMeta):
    """
    Behavior used to wait for messages. It does not schedule execution but waits until
    a suitable message is dispatched.
    """
    _template: MessageTemplate
    _message: Optional[Message]

    def __init__(self, template: MessageTemplate):
        """
        :param template: Template for the message to wait for
        """
        super().__init__()
        self._template = template

    @property
    def template(self) -> MessageTemplate:
        """Template used to receive messages for this behavior."""
        return self._template

    @property
    def message(self) -> Optional[Message]:
        """
        Received message to handle or None if nothing received
        """
        return self._message

    def schedule_execution(self):
        """
        No-op since this behavior awaits incoming messages rather than being scheduled.
        """

    async def handle_message(self, context: AgentContext, message: Message):
        """
        Handles received messages by storing them into internal state and calling `_run`.
        """
        self._context = context
        self._message = message
        await self._run()


class ReceiverBehavior(MessageHandlingBehavior):
    """
    Inherits from MessageHandlingBehavior and returns 'True' for is_done() once a message is received.
    """

    async def step(self):
        """
        Do nothing, we just expect the message
        """

    def is_done(self) -> bool:
        return self.message is not None
