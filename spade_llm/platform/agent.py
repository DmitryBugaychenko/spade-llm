import asyncio
import threading
import uuid
from abc import ABCMeta, abstractmethod
from asyncio import AbstractEventLoop
from typing import Dict, List, Optional

from spade_llm.platform.api import MessageHandler, Message
from spade_llm.platform.behaviors import Behaviour, MessageHandlingBehavior, BehaviorsOwner


# Rest of the imports remain unchanged...

class MessageDispatcher(metaclass=ABCMeta):
    # Existing implementation remains unchanged...
    
class PerformativeDispatcher(MessageDispatcher):
    # Existing implementation remains unchanged...

class ThreadDispatcher(MessageDispatcher):
    # Existing implementation remains unchanged...

class Agent(MessageHandler, BehaviorsOwner, metaclass=ABCMeta):
    """
    Base class for all agents. Provide asyncio event loop for execution, adds and removes
    behaviors, handle messages by dispatching them to interested behaviours
    """
    _loop: AbstractEventLoop
    _dispatcher: ThreadDispatcher
    _behaviors: list[Behaviour]  # Internal list for storing non-MHB Behaviors
    _is_done: asyncio.Event  # Event flag indicating completion
    _thread: Optional[threading.Thread] = None  # Store reference to the thread
    _agent_type: str  # New private variable for agent_type

    def __init__(self, agent_type: str):  # Updated constructor signature
        self._agent_type = agent_type  # Assign agent_type during initialization
        self._dispatcher = ThreadDispatcher()
        self._behaviors = []
        self._is_done = asyncio.Event()  # Initialize the event flag

    @property
    def agent_type(self) -> str:  # Read-only getter for agent_type
        return self._agent_type

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
        Creates a new event loop, starts a new thread, runs the event loop in the thread,
        and calls run_until_complete for the loop.
        """
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_agent_in_thread,
            name=f"{self.agent_type}-Thread",  # Set thread name based on agent_type
        )
        self._thread.start()

        self.loop.call_soon_threadsafe(self._start_behaviors)

    def _start_behaviors(self):
        """
        Starts all behaviors
        """
        for beh in self._behaviors:
            beh.start()

    def _run_agent_in_thread(self):
        """
        Runs the event loop in a separate thread.
        """
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._is_done.wait())

    def add_behaviour(self, beh: Behaviour):
        """
        Adds a behavior to either the dispatcher or the internal list based on its type.
        :param beh: The behavior to add.
        """
        self._behaviors.append(beh)
        if isinstance(beh, MessageHandlingBehavior):
            self._dispatcher.add_behaviour(beh)
        beh.setup(self)

    def remove_behaviour(self, beh: Behaviour):
        """
        Removes a behavior from either the dispatcher or the internal list based on its type.
        :param beh: The behavior to remove.
        """
        if beh in self._behaviors:
            self._behaviors.remove(beh)
            if isinstance(beh, MessageHandlingBehavior):
                self._dispatcher.remove_behaviour(beh)

    async def handle_message(self, context, message):
        """
        Tries to find behavior(s) matching message and passes message to each of them. Afterward checks if behavior(s) are
        done and if so removes them.
        """
        done_behaviors = list()  # Collect behaviors marked as done
        for matched_behavior in self._dispatcher.find_matching_behaviour(message):
            await matched_behavior.handle_message(context, message)
            if matched_behavior.is_done():
                done_behaviors.append(matched_behavior)
        for behavior in done_behaviors:
            self.remove_behaviour(behavior)

    def stop(self):
        """
        Stops the agent by stopping the event loop and signaling completion.
        """
        self._loop.stop()
        self._is_done.set()

    async def join(self):
        """
        Waits for the agent to complete its operations.
        """
        if self._thread:
            self._thread.join()

    def is_running(self) -> bool:
        """
        Checks if the agent's thread is alive and running.
        """
        return self._thread is not None and self._thread.is_alive()
