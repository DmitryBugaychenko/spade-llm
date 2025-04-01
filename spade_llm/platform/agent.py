import asyncio
import uuid
from abc import ABCMeta, abstractmethod
from asyncio import AbstractEventLoop
from typing import Dict, List, Optional

from spade_llm.platform.api import MessageHandler, Message
from spade_llm.platform.behaviors import Behaviour, MessageHandlingBehavior, BehaviorsOwner

class MessageDispatcher(MessageHandler, metaclass=ABCMeta):
    # ... (unchanged)

class PerformativeDispatcher(MessageDispatcher):
    # ... (unchanged)

class ThreadDispatcher(MessageDispatcher):
    # ... (unchanged)

class Agent(MessageHandler, BehaviorsOwner, metaclass=ABCMeta):
    """
    Base class for all agents. Provide asyncio event loop for execution, adds and removes
    behaviors, handle messages by dispatching them to interested behaviours
    """
    _loop: AbstractEventLoop
    _dispatcher: ThreadDispatcher
    _local_behaviors: List[Behaviour]  # Internal list for storing non-MHB Behaviors

    def __init__(self):
        self._dispatcher = ThreadDispatcher()
        self._local_behaviors = []

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
        # Start behaviors managed by the dispatcher
        for behavior in self._dispatcher._dispatchers_by_thread.values():
            for b in behavior._behaviors_by_performative.values():
                for bhv in b:
                    bhv.start()
        
        # Also start local behaviors
        for bhv in self._local_behaviors:
            bhv.start()

    def add_behaviour(self, beh: Behaviour):
        """
        Adds a behavior to either the dispatcher or the internal list based on its type.
        :param beh: The behavior to add.
        """
        if isinstance(beh, MessageHandlingBehavior):
            self._dispatcher.add_behaviour(beh)
        else:
            self._local_behaviors.append(beh)

    def remove_behaviour(self, beh: Behaviour):
        """
        Removes a behavior from either the dispatcher or the internal list based on its type.
        :param beh: The behavior to remove.
        """
        if isinstance(beh, MessageHandlingBehavior):
            self._dispatcher.remove_behaviour(beh)
        elif beh in self._local_behaviors:
            self._local_behaviors.remove(beh)

    async def handle_message(self, context, message):
        """
        Tries to find behavior(s) matching message and passes message to each of them. Afterward checks if behavior(s) are
        done and if so removes them.
        """
        done_behaviors = set()  # Collect behaviors marked as done
        for matched_behavior in self._dispatcher.find_matching_behaviour(message):
            await matched_behavior.handle_message(context, message)
            if matched_behavior.is_done():
                done_behaviors.add(matched_behavior)
        for behavior in done_behaviors:
            self.remove_behaviour(behavior)
