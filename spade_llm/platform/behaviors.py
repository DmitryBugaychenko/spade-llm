from abc import ABCMeta, abstractmethod
import asyncio

from spade_llm.platform.agent import Agent, MessageTemplate
from spade_llm.platform.api import MessageHandler, AgentContext


class Behaviour(metaclass=ABCMeta):
    """
    Reusable code block for the agent, consumes a message matching certain template and
    handles it
    """
    _agent: Agent
    @property
    def agent(self) -> Agent:
        """
        Readonly property returning the associated agent.
        """
        return self._agent

    def setup(self, agent: Agent):
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
        self.agent.loop.call_soon(self._run)

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
            self.agent.remove_behaviour(self)
        else:
            self.schedule_execution()

    async def join(self):
        """
        Waits for the behavior to complete before continuing.
        """
        await self._completion_event.wait()

    def set_completion_event(self):
        """
        Sets up the event signaling when the behavior completes.
        """
        self._completion_event = asyncio.Event()

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


class MessageHandlingBehavior(ContextBehaviour, MessageHandler, metaclass=ABCMeta):
    """
    Behavior used to wait for messages. It does not schedule execution but waits until
    a suitable message is dispatched.
    """
    @property
    @abstractmethod
    def template(self) -> MessageTemplate:
        """Template used to receive messages for this behavior."""

    def schedule_execution(self):
        """
        No-op since this behavior awaits incoming messages rather than being scheduled.
        """

    async def handle_message(self, context: AgentContext, message):
        """
        Handles received messages by storing them into internal state and calling `_run`.
        """
        self.context = context
        self.message = message
        await self._run()
