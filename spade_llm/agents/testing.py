import queue
from typing import List, Callable, Any
from spade_llm.core.agent import Agent
from spade_llm.core.behaviors import MessageHandlingBehavior, MessageTemplate
from spade_llm.builders import MessageBuilder
from spade_llm.core.api import Message, AgentContext
import logging


class AccumulateMessagesBehavior(MessageHandlingBehavior):
    """Behavior for accumulating all incoming messages"""
    
    def __init__(self, message_queue: queue.Queue):
        super().__init__(MessageTemplate())
        self.message_queue = message_queue
    
    async def step(self):
        if self.message:
            self.message_queue.put(self.message)
            # Send acknowledgment
            await self.context.reply_with_acknowledge(self.message).with_content("")


class ExecuteContextLambdaBehavior(MessageHandlingBehavior):
    """Behavior for executing a lambda expression over the agent's default context"""
    
    def __init__(self, func: Callable[[AgentContext], Any], future: Any):
        super().__init__(None)  # No template, one-time use
        self.func = func
        self.future = future
        self.logger = logging.getLogger(__name__)
    
    async def step(self):
        try:
            result = self.func(self.context)
            if hasattr(result, '__await__'):
                result = await result
            self.future.set_result(result)
        except Exception as e:
            self.logger.error(f"Error executing context lambda: {e}", exc_info=True)
            self.future.set_exception(e)
        finally:
            self.set_is_done()


class DummyAgent(Agent):
    """This is a dummy agent for testing purposes. It allows to send messages to over agents
    and collect all incoming messages for assertions"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.message_queue = queue.Queue()
        self.accumulate_behavior = AccumulateMessagesBehavior(self.message_queue)
    
    def setup(self):
        self.add_behaviour(self.accumulate_behavior)
    
    def get_received_messages(self) -> List[Message]:
        """Get all accumulated messages"""
        messages = []
        while not self.message_queue.empty():
            try:
                messages.append(self.message_queue.get_nowait())
            except queue.Empty:
                break
        return messages
    
    def clear_messages(self):
        """Clear accumulated messages"""
        while not self.message_queue.empty():
            try:
                self.message_queue.get_nowait()
            except queue.Empty:
                break
    
    def as_agent(self, func: Callable[[AgentContext], Any]):
        """Execute a function over the agent's default context, marshaling execution to the event loop.
        
        Args:
            func: A callable that takes the agent's default_context as parameter
            
        Returns:
            A concurrent.futures.Future that will contain the result of the execution
        """
        import concurrent.futures
        future = concurrent.futures.Future()
        execute_behavior = ExecuteContextLambdaBehavior(func, future)
        self.add_behaviour(execute_behavior)
        return future
