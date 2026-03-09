import queue
from collections import deque
from typing import List, Optional

from spade_llm.core.agent import Agent
from spade_llm.core.api import Message, AgentContext
from spade_llm.core.behaviors import MessageHandlingBehavior, MessageTemplate, ContextBehaviour


class AccumulateMessagesBehavior(MessageHandlingBehavior):
    """Behavior for accumulating all incoming messages"""

    def __init__(self, message_queue: queue.Queue):
        super().__init__(MessageTemplate())
        self.message_queue = message_queue

    async def step(self):
        if self.message:
            self.message_queue.put(self.message)

class ExecuteInContext:
    async def execute(self, context: ContextBehaviour):
        pass


class FunctionExecutorBehavior(ContextBehaviour):
    """Behavior for executing functions from the FIFO queue one by one"""

    def __init__(self, function_queue: deque, context: AgentContext):
        super().__init__(context)
        self.function_queue = function_queue

    async def step(self):
        if self.function_queue:
            func = self.function_queue.popleft()
            try:
                await func.execute(self)
            except Exception as e:
                self.logger.error(f"Error executing function from queue: {e}", exc_info=True)
        else:
            # No more functions to execute, shut down the agent
            self.set_is_done()
            self.agent.stop()


class DummyAgent(Agent):
    """This is a dummy agent for testing purposes. It allows to send messages to over agents
    and collect all incoming messages for assertions"""

    def __init__(self, agent_type: str = "dummy"):
        super().__init__(agent_type)
        self.message_queue = queue.Queue()
        self.function_queue = deque()  # FIFO queue for functions
        self.accumulate_behavior = AccumulateMessagesBehavior(self.message_queue)

    def setup(self):
        self.add_behaviour(self.accumulate_behavior)
        # Add the function executor behavior
        function_executor = FunctionExecutorBehavior(self.function_queue, self.default_context)
        self.add_behaviour(function_executor)

    def get_received_messages(self) -> List[Message]:
        """Get all accumulated messages"""
        messages = []
        while not self.message_queue.empty():
            try:
                messages.append(self.message_queue.get_nowait())
            except queue.Empty:
                break
        return messages

    def get_message(self, timeout: float = 5.0) -> Optional[Message]:
        """Fetch a single message from the queue with a timeout.

        Args:
            timeout: Maximum time to wait for a message in seconds. Defaults to 5.0.

        Returns:
            Message if available within timeout, None otherwise.
        """
        try:
            return self.message_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear_messages(self):
        """Clear accumulated messages"""
        while not self.message_queue.empty():
            try:
                self.message_queue.get_nowait()
            except queue.Empty:
                break

    def as_agent(self, func: ExecuteInContext):
        """Add a function to the FIFO queue for execution.

        Args:
            func: An async callable that takes the agent's default_context as parameter
        """
        self.function_queue.append(func)