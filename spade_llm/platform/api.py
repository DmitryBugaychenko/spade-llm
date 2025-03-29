import uuid
from abc import ABCMeta, abstractmethod
from typing import Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


# Other imports remain unchanged...

class AgentHandler(metaclass=ABCMeta):
    """
    Interface provided by the agent to integrate with the platform.
    """
    @property
    @abstractmethod
    def agent_type(self) -> str:
        """
        Type of the agent unique in the system. Used as part of the agent address and has
        its own namespace in the agent context store.
        """
        pass

    @property
    @abstractmethod
    def tools(self) -> list[BaseTool]:
        """
        Tools available for the agent.
        """
        pass

    @abstractmethod
    async def handle_message(self, context: AgentContext, message: Message):
        """
        Handles a single message addressed to a particular agent in a particular thread (optional).
        :param context: Context with access to key/value storage and tool calling.
        :param message: Message to handle.
        """
        pass


class AgentPlatform(metaclass=ABCMeta):
    """
    Abstraction of an agent platform. Allows adding agents with their handlers and accessible tools.
    """
    @abstractmethod
    async def register_agent(self, handler: AgentHandler):
        """
        Adds a new agent to the platform.
        :param handler: Message handler for the agent.
        """
        pass
