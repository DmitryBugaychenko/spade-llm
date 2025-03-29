import uuid
from abc import ABCMeta
from typing import Optional

from spade_llm.platform.api import MessageHandler, Message


class MessageTemplate:
    @property
    def thread_id(self) -> Optional[uuid.UUID]:
        """
        If provided, message must be a part of this thread
        """
        pass

    @property
    def performative(self) -> Optional[str]:
        """
        If provided, message must be of this performative
        :return:
        """
        pass

    def match(self, msg: Message) -> bool:
        """
        Check is current message match this template
        :param msg: Message to check
        :return: True if message matched and false otherwise
        """
        return True




class Behaviour(MessageHandler, metaclass=ABCMeta):
    pass