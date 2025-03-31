from abc import ABCMeta, abstractmethod

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
        Executes a single step of this behaviour. Steps should not block for io (use asyncio instead),
        neither perform long computation (unload them using Executor).
        """

    async def _run(self):
        """
        Internal method, executes single step and then either remove behaviour or schedule next
        execution depending if behavior is done or not
        """
        await self.step()
        if self.is_done():
            self.agent.remove_behaviour(self)
        else:
            self.schedule_execution()


    #TODO: Add async method 'join' to wait for behaviour completion, use event to signal about completion from '_run'

    @abstractmethod
    def is_done(self) -> bool:
        """
        Returns true if behavior is completed and should not accept messages anymore and false otherwise
        """

class ContextBehaviour(Behaviour, metaclass=ABCMeta):
    """
    This implementation provides access to agent context for the behaviour
    """
    _context: AgentContext

    #TODO: Add read/write property for context

class MessageHandlingBehavior(ContextBehaviour, MessageHandler, metaclass=ABCMeta):
    """
    Behaviour used to wait for messages. Does not schedule execution, waits until
    proper message is dispatched
    """
    @property
    @abstractmethod
    def template(self) -> MessageTemplate:
        """Template used to get messages for this behavior"""

    def schedule_execution(self):
        """
        Do nothing, this behavior waits for messages, not for schedule
        """

    #TODO: Override 'handle_message', store context and message to properties, then call _run