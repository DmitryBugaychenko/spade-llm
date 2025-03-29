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
    @abstractmethod
    def template(self) -> MessageTemplate:
        """Template used to get messages for this behavior"""

    @abstractmethod
    def is_done(self) -> bool:
        """
        Returns true if behavior is completed and should not accept messages anymore and false otherwise
        """

class CyclicBehaviour(Behaviour, metaclass=ABCMeta):
    """Cyclic behaviour which is never done."""
    def is_done(self) -> bool:
        return False

class OneShotBehaviour(Behaviour, metaclass=ABCMeta):
    """One shot behaviour is done after first execution."""
    def is_done(self) -> bool:
        return True


class MessageDispatcher(MessageHandler, metaclass=ABCMeta):
    @abstractmethod
    def add_behaviour(self, beh):
        """
        Add behavior to dispatch list.
        :param beh: Behavior to add.
        """

    @abstractmethod
    def remove_behaviour(self, beh):
        """
        Removes behavior from dispatch.
        :param beh: Behavior to remove.
        """

    @abstractmethod
    def find_matching_behaviour(self, msg: Message) -> Optional[Behaviour]:
        """Lookups for behavior matching given message, first selects proper list based on performative,
        then finds the first one with matching template."""

    async def handle_message(self, context, message):
        """
        Tries to find behavior matching message and passes message to it. Afterward checks if behavior is
        done and if so removes it.
        """
        matched_behavior = self.find_matching_behaviour(message)
        if matched_behavior:
            await matched_behavior.handle_message(context, message)
            if matched_behavior.is_done():
                self.remove_behaviour(matched_behavior)

    @property
    @abstractmethod
    def is_empty(self) -> bool:
        """Returns true if there are no behaviors."""


class PerformativeDispatcher(MessageDispatcher):
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

    def find_matching_behaviour(self, msg: Message) -> Optional[Behaviour]:
        """Lookups for behavior matching given message, first selects proper list based on performative,
        then finds the first one with matching template."""
        performative = msg.performative
        if performative in self.behaviors_by_performative:
            for beh in self.behaviors_by_performative[performative]:
                if beh.template.match(msg):
                    return beh
        return None


    @property
    def is_empty(self) -> bool:
        """Returns true if there are no behaviors."""
        return len(self.behaviors_by_performative) == 0

class ThreadDispatcher(MessageDispatcher):
    """
    Stores PerformativeDispatcher grouped by thread id with a separate list for behaviors without thread specified.
    When behaviour is removed checks if nested dispatcher is empty and removes it if so.
    """
    def __init__(self):
        self.dispatchers_by_thread: Dict[Optional[uuid.UUID], PerformativeDispatcher] = {}

    def add_behaviour(self, beh: Behaviour):
        """
        Add behavior to dispatch list.
        :param beh: Behavior to add.
        """
        thread_id = beh.template.thread_id
        if thread_id not in self.dispatchers_by_thread:
            self.dispatchers_by_thread[thread_id] = PerformativeDispatcher()
        self.dispatchers_by_thread[thread_id].add_behaviour(beh)

    def remove_behaviour(self, beh: Behaviour):
        """
        Removes behavior from dispatch.
        :param beh: Behavior to remove.
        """
        thread_id = beh.template.thread_id
        if thread_id in self.dispatchers_by_thread:
            self.dispatchers_by_thread[thread_id].remove_behaviour(beh)
            if self.dispatchers_by_thread[thread_id].is_empty:
                del self.dispatchers_by_thread[thread_id]

    def find_matching_behaviour(self, msg: Message) -> Optional[Behaviour]:
        """
        Find matching behavior for the given message.
        """
        thread_id = msg.thread_id
        if thread_id in self.dispatchers_by_thread:
            return self.dispatchers_by_thread[thread_id].find_matching_behaviour(msg)
        return None

    @property
    def is_empty(self) -> bool:
        """
        Check if all dispatchers are empty.
        """
        return len(self.dispatchers_by_thread) == 0
