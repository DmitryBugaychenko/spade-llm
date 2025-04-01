import asyncio
import uuid
from abc import ABCMeta, abstractmethod
from asyncio import AbstractEventLoop
from typing import Dict, List, Optional

from spade_llm.platform.api import MessageHandler, Message
from spade_llm.platform.behaviors import Behaviour, MessageHandlingBehavior, BehaviorsOwner
from spade_llm.platform.agent import ThreadDispatcher


class Agent(MessageHandler, BehaviorsOwner, metaclass=ABCMeta):
    """
    Base class for all agents. Provide asyncio event loop for execution, adds and removes
    behaviors, handle messages by dispatching them to interested behaviours
    """
    _loop: AbstractEventLoop
    _dispatcher: ThreadDispatcher  # Dispatcher to manage behaviors

    def __init__(self):
        self._loop = None
        self._dispatcher = ThreadDispatcher()  # Instantiate ThreadDispatcher

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

    def add_behaviour(self, beh: Behaviour):
        """
        Adds a behavior to the agent's dispatcher.
        Only MessageHandlingBehavior types are supported.
        :param beh: The behavior to add.
        """
        if isinstance(beh, MessageHandlingBehavior):  # Check if it's a valid behavior type
            self._dispatcher.add_behaviour(beh)
        else:
            raise ValueError("Only MessageHandlingBehavior can be added.")

    def remove_behaviour(self, beh: Behaviour):
        """
        Removes a behavior from the agent's dispatcher.
        :param beh: The behavior to remove.
        """
        if isinstance(beh, MessageHandlingBehavior):  # Check if it's a valid behavior type
            self._dispatcher.remove_behaviour(beh)
        else:
            raise ValueError("Only MessageHandlingBehavior can be removed.")


# Remaining classes remain unchanged...
class MessageDispatcher(MessageHandler, metaclass=ABCMeta):
    # ... (unchanged)

class PerformativeDispatcher(MessageDispatcher):
    # ... (unchanged)

class ThreadDispatcher(MessageDispatcher):
    # ... (unchanged)
