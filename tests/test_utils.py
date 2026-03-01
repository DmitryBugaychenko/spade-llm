import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional, Dict

from pydantic import BaseModel

from spade_llm.core.agent import Agent
from spade_llm.core.messaging import DictionaryMessageService
from spade_llm.core.models import ModelsProvider
from spade_llm.core.platform import AgentPlatformImpl
from spade_llm.core.storage import InMemoryStorageFactory
from spade_llm.core.tools import DelegateToolConfig


@dataclass
class AgentEntry:
    """Describes an agent to be registered on the test platform."""
    agent: Agent
    tools: list[Any] = field(default_factory=list)
    contacts: list[DelegateToolConfig] = field(default_factory=list)
    configuration: Optional[BaseModel] = None


class SimpleModelsProvider(ModelsProvider):
    """Simple implementation of ModelsProvider using dictionaries"""
    
    def __init__(self, chat_models: Dict[str, Any], embedding_models: Dict[str, Any]):
        self.chat_models = chat_models or {}
        self.embedding_models = embedding_models or {}
    
    def get_chat_model(self, model_name: str):
        return self.chat_models.get(model_name)
    
    def get_embeddings_model(self, model_name: str):
        return self.embedding_models.get(model_name)


class TestPlatform:
    """
    A test utility that starts pre-configured agents using in-memory
    implementations of core platform services (storage, messaging).

    Usage:
        agent_a = MyAgentA(agent_type="agent_a")
        agent_b = MyAgentB(agent_type="agent_b")

        platform = TestPlatform(
            agents=[
                AgentEntry(agent=agent_a),
                AgentEntry(agent=agent_b, tools=[some_tool]),
            ],
            wait_for={"agent_a"},
        )
        await platform.run()
    """

    def __init__(
        self,
        agents: list[AgentEntry],
        wait_for: set[str] | None = None,
        chat_models: Dict[str, Any] | None = None,
        embedding_models: Dict[str, Any] | None = None,
    ):
        """
        :param agents: List of AgentEntry instances describing agents to register.
        :param wait_for: Set of agent_type names to wait for before shutting down.
                         If None or empty, all agents are awaited.
        :param chat_models: Dictionary mapping model names to chat model instances.
        :param embedding_models: Dictionary mapping model names to embedding model instances.
        """
        self.agents = None
        self.platform = None
        self.agent_entries = agents
        self.wait_for = wait_for or set()
        self.chat_models = chat_models or {}
        self.embedding_models = embedding_models or {}

    async def run(self):
        await self.start()

        await self.stop()

    async def stop(self):
        if len(self.wait_for) > 0:
            await self._wait_for_agents([self.agents[a] for a in self.wait_for])
            for agent_type, agent in self.agents.items():
                if agent_type not in self.wait_for:
                    agent.stop()
        await self._wait_for_agents(list(self.agents.values()))
        await self.platform.shutdown()

    async def start(self):
        storage_factory = InMemoryStorageFactory()
        message_service = DictionaryMessageService()
        
        # Create models provider from dictionaries
        models_provider = SimpleModelsProvider(self.chat_models, self.embedding_models)
        
        self.platform = AgentPlatformImpl(storage_factory, message_service, models_provider)
        self.agents: dict[str, Agent] = {}
        for entry in self.agent_entries:
            agent = entry.agent
            agent_type = agent.agent_type
            
            # If the agent is configurable and has a configuration, apply it
            if entry.configuration is not None and hasattr(agent, '_configure'):
                agent._configure(entry.configuration)
                
            await self.platform.register_agent(agent, entry.tools, entry.contacts)
            self.agents[agent_type] = agent
        await message_service.start_bridges()


    @staticmethod
    async def _wait_for_agents(agents: list[Agent]):
        tasks = [a.join() for a in agents]
        await asyncio.gather(*tasks, return_exceptions=True)
