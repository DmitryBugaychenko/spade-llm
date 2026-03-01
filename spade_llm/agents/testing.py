from typing import List, Callable, Any
from spade_llm.core.agent import Agent
from spade_llm.core.behaviors import MessageHandlingBehavior, MessageTemplate
from spade_llm.builders import MessageBuilder
from spade_llm.core.api import Message


class AccumulateMessagesBehavior(MessageHandlingBehavior):
    """Behavior for accumulating all incoming messages"""
    
    def __init__(self):
        super().__init__(MessageTemplate())
        self.messages: List[Message] = []
    
    async def step(self):
        if self.message:
            self.messages.append(self.message)
            # Send acknowledgment
            await self.context.reply_with_acknowledge(self.message).with_content("")


class SendMessageBehavior(MessageHandlingBehavior):
    """Behavior for sending a message and then removing itself"""
    
    def __init__(self, message_builder: MessageBuilder):
        super().__init__(None)  # No template, one-time use
        self.message_builder = message_builder
        self.sent = False
    
    async def step(self):
        if not self.sent:
            # Send the message
            await self.context.send_message(self.message_builder)
            self.sent = True
            # Remove this behavior after sending
            self.kill()


class ExecuteContextLambdaBehavior(MessageHandlingBehavior):
    """Behavior for executing a lambda expression over the agent's default context"""
    
    def __init__(self, func: Callable, future: Any):
        super().__init__(None)  # No template, one-time use
        self.func = func
        self.future = future
        self.executed = False
    
    async def step(self):
        if not self.executed:
            try:
                result = self.func(self.context)
                if hasattr(result, '__await__'):
                    result = await result
                self.future.set_result(result)
            except Exception as e:
                self.future.set_exception(e)
            finally:
                self.executed = True
                self.kill()


class DummyAgent(Agent):
    """This is a dummy agent for testing purposes. It allows to send messages to over agents
    and collect all incoming messages for assertions"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accumulate_behavior = AccumulateMessagesBehavior()
    
    def setup(self):
        self.add_behaviour(self.accumulate_behavior)
    
    def get_received_messages(self) -> List[Message]:
        """Get all accumulated messages"""
        return self.accumulate_behavior.messages.copy()
    
    def clear_messages(self):
        """Clear accumulated messages"""
        self.accumulate_behavior.messages.clear()
    
    def send_message_from_outside(self, message_builder: MessageBuilder):
        """Send a message to another agent from outside the agent event loop.
        This method adds a behavior to handle the message sending within the agent's event loop."""
        send_behavior = SendMessageBehavior(message_builder)
        self.add_behaviour(send_behavior)
    
    def execute_context_lambda(self, func: Callable):
        """Execute a lambda expression over the agent's default context, marshaling execution to the event loop.
        
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
