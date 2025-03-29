import uuid
from abc import ABCMeta
from typing import Optional

from spade_llm.platform.api import MessageHandler, Message


class MessageTemplate:
    def __init__(self, thread_id: Optional[uuid.UUID] = None, performative: Optional[str] = None):
        """
        Initializes the MessageTemplate with optional thread_id and performative.
        
        Args:
            thread_id (Optional[uuid.UUID], optional): The thread identifier. Defaults to None.
            performative (Optional[str], optional): The performative string. Defaults to None.
        """
        self._thread_id = thread_id
        self._performative = performative

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
        return True


class Behaviour(MessageHandler, metaclass=ABCMeta):
    pass
