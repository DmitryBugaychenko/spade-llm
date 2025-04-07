import asyncio
import logging
import threading
from asyncio import AbstractEventLoop
from typing import Optional

import aiologic


class EventLoopThread:
    """
    Class to manage an asyncio event loop in a separate thread.
    """

    _loop: AbstractEventLoop
    _is_stopped: asyncio.Event
    _thread: Optional[threading.Thread] = None
    _logger: logging.Logger
    _is_completed: aiologic.Event  # This is a thread-safe event


    def __init__(self):  # Constructor signature remains unchanged
        self._logger = logging.getLogger(self.__class__.__name__)
        self._is_stopped = asyncio.Event()
        self._is_completed = aiologic.Event() # This is a thread-safe event

    @property
    def loop(self):
        return self._loop

    @property
    def logger(self):
        return self._logger

    def start(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self.run_asyncio_loop,
            args=(self.loop,),
            name=f"{self.__class__.__name__}-Thread")
        self._thread.start()

    def run_asyncio_loop(self, loop):
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._is_stopped.wait())
        finally:
            # Cleanly shutdown the event loop
            self.logger.info("Shutting down thread...")
            all_tasks = asyncio.all_tasks(loop=self.loop)
            for task in all_tasks:
                task.cancel()
            loop.run_until_complete(asyncio.gather(*all_tasks, return_exceptions=True))
            loop.close()
            self._is_completed.set()
            self.logger.info("Exited thread.")

    def stop(self):
        self.loop.call_soon_threadsafe(self._is_stopped.set)

    async def join(self):
        await self._is_completed