import asyncio
import uuid
from abc import ABCMeta, abstractmethod
from asyncio import AbstractEventLoop
from typing import Dict, List, Optional

from spade_llm.platform.api import MessageHandler, Message
from spade_llm.platform.behaviors import Behaviour, MessageHandlingBehavior, BehaviorsOwner


class Agent(MessageHandler, BehaviorsOwner, metaclass=ABCMeta):
    """
    Base class for all agents. Provide asyncio event loop for execution, adds and removes
    behaviors, handle messages by dispatching them to interested behaviours
    """
    _loop: AbstractEventLoop
    _behaviors: List[Behaviour]  # New field to hold behaviors

    def __init__(self):
        self._loop = None
        self._behaviors = []  # Initialize the list here

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """
        Event loop used to execute agent logic and handle incoming messages.
        """
        return self._loop

    def setup(self):
        """
        Setup method to initialize agent and construct all default behaviors
        """

    def start(self) -> None:
        """
        Starts an agent and all its behaviors
        """
        for behavior in self._behaviors:
            behavior.start()

    def add_behaviour(self, beh: Behaviour):
        """
        Adds a behavior to the agent's list of behaviors.
        :param beh: The behavior to add.
        """
        self._behaviors.append(beh)

    def remove_behaviour(self, beh: Behaviour):
        """
        Removes a behavior from the agent's list of behaviors.
        :param beh: The behavior to remove.
        """
        if beh in self._behaviors:
            self._behaviors.remove(beh)


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
    def find_matching_behaviour(self, msg: Message):
        """Lookups for behavior(s) matching given message, yields all matching ones."""

    async def handle_message(self, context, message):
        """
        Tries to find behavior(s) matching message and passes message to each of them. Afterward checks if behavior(s) are
        done and if so removes them.
        """
        done_behaviors = set()  # Collect behaviors marked as done
        for matched_behavior in self.find_matching_behaviour(message):
            await matched_behavior.handle_message(context, message)
            if matched_behavior.is_done():
                done_behaviors.add(matched_behavior)
        for behavior in done_behaviors:
            self.remove_behaviour(behavior)

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
        self._behaviors_by_performative: Dict[Optional[str], List[MessageHandlingBehavior]] = {}
    
    def add_behaviour(self, beh: MessageHandlingBehavior):
        """
        Add behavior to dispatch list.
        :param beh: Behavior to add.
        """
        performative = beh.template.performative
        if performative not in self._behaviors_by_performative:
            self._behaviors_by_performative[performative] = []
        self._behaviors_by_performative[performative].append(beh)

    def remove_behaviour(self, beh: MessageHandlingBehavior):
        """
        Removes behavior from dispatch.
        :param beh: Behavior to remove.
        """
        performative = beh.template.performative
        if performative in self._behaviors_by_performative:
            self._behaviors_by_performative[performative].remove(beh)
            if not self._behaviors_by_performative[performative]:
                del self._behaviors_by_performative[performative]

    def find_matching_behaviour(self, msg: Message):
        """Yields all behaviors matching the given message."""
        performative = msg.performative
        if performative in self._behaviors_by_performative:
            for beh in self._behaviors_by_performative[performative]:
                if beh.template.match(msg):
                    yield beh
        # Check behaviours without performative specified if not found
        if None in self._behaviors_by_performative:
            for beh in self._behaviors_by_performative[None]:
                if beh.template.match(msg):
                    yield beh

    @property
    def is_empty(self) -> bool:
        """Returns true if there are no behaviors."""
        return len(self._behaviors_by_performative) == 0

class ThreadDispatcher(MessageDispatcher):
    """
    Stores PerformativeDispatcher grouped by thread id with a separate list for behaviors without thread specified.
    When behaviour is removed checks if nested dispatcher is empty and removes it if so.
    """
    def __init__(self):
        self._dispatchers_by_thread: Dict[Optional[uuid.UUID], PerformativeDispatcher] = {}

    def add_behaviour(self, beh: MessageHandlingBehavior):
        """
        Add behavior to dispatch list.
        :param beh: Behavior to add.
        """
        thread_id = beh.template.thread_id
        if thread_id not in self._dispatchers_by_thread:
            self._dispatchers_by_thread[thread_id] = PerformativeDispatcher()
        self._dispatchers_by_thread[thread_id].add_behaviour(beh)

    def remove_behaviour(self, beh: MessageHandlingBehavior):
        """
        Removes behavior from dispatch.
        :param beh: Behavior to remove.
        """
        thread_id = beh.template.thread_id
        if thread_id in self._dispatchers_by_thread:
            self._dispatchers_by_thread[thread_id].remove_behaviour(beh)
            if self._dispatchers_by_thread[thread_id].is_empty:
                del self._dispatchers_by_thread[thread_id]

    def find_matching_behaviour(self, msg: Message):
        """
        Yields all matching behaviors for the given message.
        """
        thread_id = msg.thread_id
        if thread_id in self._dispatchers_by_thread:
            for beh in self._dispatchers_by_thread[thread_id].find_matching_behaviour(msg):
                yield beh
        if None in self._dispatchers_by_thread:
            for beh in self._dispatchers_by_thread[None].find_matching_behaviour(msg):
                yield beh

    @property
    def is_empty(self) -> bool:
        """
        Check if all dispatchers are empty.
        """
        return len(self._dispatchers_by_thread) == 0
