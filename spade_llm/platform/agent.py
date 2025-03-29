import uuid
from abc import ABCMeta, abstractmethod
from typing import Callable, Optional

from spade_llm.platform.api import MessageHandler, Message, AgentContext


class MessageTemplate:
    """
    Templates are used to filter messages and dispatch them to proper behaviour
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


class Behaviour(MessageHandler, metaclass=ABCMeta):
    """
    Reusable code block for the agent, consumes a message matching certain template and
    handles it
    """
    @property
    def template(self) -> MessageTemplate:
        pass

    @abstractmethod
    def is_done(self) -> bool:
        """
        Returns true if behaviour is completed and should not accept messages any more and false otherwise
        """
        pass

class PerformativeDispatcher(MessageHandler):
    """
    Stores behaviours grouped by performative configured in their template. Uses dictionary with performative
    as a key and list of behaviours as values plus extra list for behaviours without performative specified
    """

    def add_behaviour(self, beh: Behaviour):
        """
        Add behaviour to dispatch list
        :param beh: Behaviour to add.
        """
        pass

    def remove_behaviour(self, beh: Behaviour):
        """
        Removes behaviour from dispatch
        :param beh: Behaviour to remove.
        """
        pass

    @property
    def is_empty(self) -> bool:
        """Returns true if there are no behaviours"""
        pass

    def find_matching_behaviour(self, msg: Message) -> Optional[Behaviour]:
        """Lookups for behaviour matching given message, first select proper list based on performative,
        then find the first one with matching template"""
        pass

    async def handle_message(self, context: AgentContext, message: Message):
        """
        Tries to find behaviour matching message and pass message to it. After that check is behaviour is
        done and if so removes it.
        """
        pass
