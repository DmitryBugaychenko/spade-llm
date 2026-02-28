import asyncio
from dataclasses import dataclass, field
from typing import Any

from spade_llm.core.agent import Agent
from spade_llm.core.messaging import InMemoryMessageService
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
        models_provider: ModelsProvider | None = None,
    ):
        """
        :param agents: List of AgentEntry instances describing agents to register.
        :param wait_for: Set of agent_type names to wait for before shutting down.
                         If None or empty, all agents are awaited.
        :param models_provider: Optional ModelsProvider; if not provided, a no-op default is used.
        """
        self.agent_entries = agents
        self.wait_for = wait_for or set()
        self.models_provider = models_provider

    async def run(self):
        storage_factory = InMemoryStorageFactory()
        message_service = InMemoryMessageService()
        platform = AgentPlatformImpl(storage_factory, message_service, self.models_provider)

        agents: dict[str, Agent] = {}

        for entry in self.agent_entries:
            agent = entry.agent
            agent_type = agent.agent_type
            await platform.register_agent(agent, entry.tools, entry.contacts)
            agents[agent_type] = agent

        await message_service.start_bridges()

        if len(self.wait_for) > 0:
            await self._wait_for_agents([agents[a] for a in self.wait_for])
            for agent_type, agent in agents.items():
                if agent_type not in self.wait_for:
                    agent.stop()

        await self._wait_for_agents(list(agents.values()))

        await platform.shutdown()

    @staticmethod
    async def _wait_for_agents(agents: list[Agent]):
        tasks = [a.join() for a in agents]
        await asyncio.gather(*tasks, return_exceptions=True)
