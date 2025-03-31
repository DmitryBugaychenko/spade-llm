import uuid
from abc import ABCMeta, abstractmethod
import asyncio
from typing import Optional, Callable

from spade_llm.platform.api import MessageHandler, AgentContext, Message

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
        pass

class Behaviour(metaclass=ABCMeta):
    """
    Reusable code block for the agent, consumes a message matching certain template and
    handles it
    """
    _agent: BehaviorsOwner
    _completion_event = asyncio.Event()

    @property
    def agent(self) -> BehaviorsOwner:
        """
        Readonly property returning the associated agent.
        """
        return self._agent

    async def setup(self, agent: BehaviorsOwner):
        """
        Setup method to initialize the behavior with its associated agent.
        """
        self._agent = agent

    async def start(self):
        """
        Start behavior by scheduling its execution
        """
        self.schedule_execution()

    def schedule_execution(self):
        """
        Schedules execution using agents event loop
        """
        callback = lambda: asyncio.ensure_future(self._run())
        self.agent.loop.call_soon(callback)

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

    @abstractmethod
    def is_done(self) -> bool:
        """
        Returns True if behavior is completed and should not accept messages anymore,
        False otherwise.
        """


class ContextBehaviour(Behaviour, metaclass=ABCMeta):
    """
    This implementation provides access to agent context for the behavior.
    """
    _context: AgentContext

    @property
    def context(self) -> AgentContext:
        """
        Property providing read/write access to the agent context.
        """
        return self._context

    @context.setter
    def context(self, value: AgentContext):
        self._context = value

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

class MessageHandlingBehavior(ContextBehaviour, MessageHandler, metaclass=ABCMeta):
    """
    Behavior used to wait for messages. It does not schedule execution but waits until
    a suitable message is dispatched.
    """
    _message: Optional[Message]

    @property
    def message(self) -> Optional[Message]:
        """
        Received message to handle or None if nothing received
        """
        return self._message

    @property
    @abstractmethod
    def template(self) -> MessageTemplate:
        """Template used to receive messages for this behavior."""

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
