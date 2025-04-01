import asyncio
import threading
import uuid
from abc import ABCMeta, abstractmethod
from asyncio import AbstractEventLoop
from typing import Dict, List, Optional

import logging

from spade_llm.platform.api import Message, AgentHandler
from spade_llm.platform.behaviors import Behaviour, MessageHandlingBehavior, BehaviorsOwner

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class MessageDispatcher(metaclass=ABCMeta):
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
        logger.info("Added behavior %s with performative %s", beh, performative)

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
            logger.info("Removed behavior %s with performative %s", beh, performative)

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
        logger.info("Added behavior %s with thread ID %s", beh, thread_id)

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
            logger.info("Removed behavior %s with thread ID %s", beh, thread_id)

    def find_matching_behaviour(self, msg: Message):
        """
        Yields all matching behaviors for the given message.
        """
        thread_id = msg.thread_id
        if thread_id and thread_id in self._dispatchers_by_thread:
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

class Agent(AgentHandler, BehaviorsOwner, metaclass=ABCMeta):
    """
    Base class for all agents. Provides asyncio event loop for execution, adds and removes
    behaviors, handles messages by dispatching them to interested behaviors.
    """
    _loop: AbstractEventLoop
    _dispatcher: ThreadDispatcher
    _behaviors: list[Behaviour]  # Internal list for storing non-MHB Behaviors
    _is_stopped: asyncio.Event  # Event flag indicating completion
    _is_completed: asyncio.Event  # Event flag indicating completion
    _thread: Optional[threading.Thread] = None  # Store reference to the thread
    _agent_type: str  # New private variable for agent_type

    def __init__(self, agent_type: str):  # Updated constructor signature
        self._agent_type = agent_type  # Assign agent_type during initialization
        self._dispatcher = ThreadDispatcher()
        self._behaviors = []
        self._is_stopped = asyncio.Event()
        self._is_completed = asyncio.Event()

    @property
    def agent_type(self) -> str:
        return self._agent_type

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """
        Event loop used to execute agent logic and handle incoming messages.
        """
        return self._loop

    def setup(self):
        """
        Setup method to initialize agent and construct all default behaviors.
        """

    def start(self) -> None:
        """
        Creates a new event loop, starts a new thread, runs the event loop in the thread,
        and calls run_until_complete for the loop.
        """
        self.setup()

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_agent_in_thread,
            name=f"{self.agent_type}-Thread",  # Set thread name based on agent_type
        )
        self._thread.start()

        self.loop.call_soon_threadsafe(self._start_behaviors)

    def _start_behaviors(self):
        """
        Starts all behaviors.
        """
        for beh in self._behaviors:
            beh.start()

    def _run_agent_in_thread(self):
        """
        Runs the event loop in a separate thread.
        """
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._is_stopped.wait())
        self._is_completed.set()

    def add_behaviour(self, beh: Behaviour):
        """
        Adds a behavior to either the dispatcher or the internal list based on its type.
        :param beh: The behavior to add.
        """
        self._behaviors.append(beh)
        if isinstance(beh, MessageHandlingBehavior):
            self._dispatcher.add_behaviour(beh)
        beh.setup(self)
        if self.is_running():
            beh.start()
        logger.info("Added behavior %s to agent %s", beh, self)

    def remove_behaviour(self, beh: Behaviour):
        """
        Removes a behavior from either the dispatcher or the internal list based on its type.
        :param beh: The behavior to remove.
        """
        if beh in self._behaviors:
            self._behaviors.remove(beh)
            if isinstance(beh, MessageHandlingBehavior):
                self._dispatcher.remove_behaviour(beh)
            logger.info("Removed behavior %s from agent %s", beh, self)

    async def handle_message(self, context, message):
        logger.debug("Handling message: %s", message)
        callback = lambda: asyncio.ensure_future(self._handle_message_in_loop(context, message))
        self.loop.call_soon_threadsafe(callback)

    async def _handle_message_in_loop(self, context, message):
        """
        Tries to find behavior(s) matching message and passes message to each of them. Afterward checks if behavior(s) are
        done and if so removes them.
        """
        done_behaviors = []  # Collect behaviors marked as done
        for matched_behavior in self._dispatcher.find_matching_behaviour(message):
            await matched_behavior.handle_message(context, message)
            if matched_behavior.is_done():
                done_behaviors.append(matched_behavior)
        for behavior in done_behaviors:
            self.remove_behaviour(behavior)
        logger.debug("Handled message: %s", message)

    def stop(self):
        """
        Stops the agent by stopping the event loop and signaling completion.
        """
        self._is_stopped.set()
        logger.info("%s stopped.", self.__class__.__name__)

    async def join(self):
        """
        Waits for the agent to complete its operations.
        """
        if self._thread:
            await self._is_completed.wait()

    def is_running(self) -> bool:
        """
        Checks if the agent's thread is alive and running.
        """
        return self._thread is not None and self._thread.is_alive()
