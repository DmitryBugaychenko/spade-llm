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

class Behaviour(metaclass=ABCMeta):
    """
    Reusable code block for the agent, consumes a message matching certain template and
    handles it
    """
    _agent: BehaviorsOwner

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

    def start(self):
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
        return False
    
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
