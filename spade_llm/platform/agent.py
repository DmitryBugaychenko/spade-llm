import uuid
from abc import ABCMeta, abstractmethod
from typing import Callable, Dict, List, Optional

from spade_llm.platform.api import MessageHandler, Message, AgentContext


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
        Returns true if behavior is completed and should not accept messages anymore and false otherwise
        """
        pass


class PerformativeDispatcher(MessageHandler):
    """
    Stores behaviors grouped by performative configured in their templates. Uses dictionary with performative
    as a key and list of behaviors as values plus extra list for behaviors without performative specified.
    """

    def __init__(self):
        self.behaviors_by_performative: Dict[Optional[str], List[Behaviour]] = {}
    
    def add_behaviour(self, beh: Behaviour):
        """
        Add behavior to dispatch list.
        :param beh: Behavior to add.
        """
        performative = beh.template.performative
        if performative not in self.behaviors_by_performative:
            self.behaviors_by_performative[performative] = []
        self.behaviors_by_performative[performative].append(beh)

    def remove_behaviour(self, beh: Behaviour):
        """
        Removes behavior from dispatch.
        :param beh: Behavior to remove.
        """
        performative = beh.template.performative
        if performative in self.behaviors_by_performative:
            self.behaviors_by_performative[performative].remove(beh)
            if not self.behaviors_by_performative[performative]:  
                del self.behaviors_by_performative[performative]

    @property
    def is_empty(self) -> bool:
        """Returns true if there are no behaviors."""
        return len(self.behaviors_by_performative) == 0

    def find_matching_behaviour(self, msg: Message) -> Optional[Behaviour]:
        """Lookups for behavior matching given message, first selects proper list based on performative,
        then finds the first one with matching template."""
        performative = msg.performative
        if performative in self.behaviors_by_performative:
            for beh in self.behaviors_by_performative[performative]:
                if beh.template.match(msg):
                    return beh
        return None

    async def handle_message(self, context: AgentContext, message: Message):
        """
        Tries to find behavior matching message and passes message to it. Afterward checks if behavior is
        done and if so removes it.
        """
        matched_behavior = self.find_matching_behaviour(message)
        if matched_behavior:
            await matched_behavior.handle_message(context, message)
            if matched_behavior.is_done():
                self.remove_behaviour(matched_behavior)
