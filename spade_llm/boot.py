import argparse
import asyncio
from typing import cast

import yaml
from pydantic import Field

from spade_llm.core.agent import Agent
from spade_llm.core.conf import ConfigurableRecord
from spade_llm.core.messaging import MessageServiceConfig
from spade_llm.core.models import ModelsProviderConfig
from spade_llm.core.platform import AgentPlatformImpl
from spade_llm.core.storage import StorageFactoryConfig
from spade_llm.core.tools import ToolProviderConfig


class AgentConfig(ConfigurableRecord):
    agent_type: str = Field(
        default="",
        description="Type of the agent. If not explicitly set name of the agent in configuration is used.")

    def _create_instance(self, cls):
        return cls(agent_type=self.agent_type)

    def create_agent(self) -> Agent:
        return cast(Agent,self.create_configurable_instance())

class PlatformConfiguration(ToolProviderConfig, ModelsProviderConfig):
    messaging: MessageServiceConfig = Field(description="Configuration for agents messaging service")
    storage: StorageFactoryConfig = Field(description="Configuration for agents state storage")
    agents: dict[str,AgentConfig] = Field(description="Agents for the system")
    wait_for_agents: set[str] = Field(
        default=set(),
        description="Name of agents to wait for before shutting down platform. If empty - all agents are awaited.")

class Boot:
    def __init__(self, config: PlatformConfiguration):
        self.config = config

    async def run(self):
        storage_factory = self.config.storage.create_factory()
        message_service = self.config.messaging.create_messaging_service()
        platform = AgentPlatformImpl(storage_factory, message_service)
        agents: dict[str,Agent] = dict()

        for agent_type, agent_conf in self.config.agents.items():
            agent_conf.agent_type = agent_type
            agent = agent_conf.create_agent()
            await platform.register_agent(agent, [])
            agents[agent_type] = agent

        if len(self.config.wait_for_agents) > 0:
            await self.wait_for_agents([agents[a] for a in self.config.wait_for_agents])
            for agent_type, agent in agents.items():
                if agent_type not in self.config.wait_for_agents:
                    agent.stop()

        await self.wait_for_agents(list(agents.values()))

        # Shut down the platform gracefully
        await platform.shutdown()

    async def wait_for_agents(self, agents: list[Agent]):
        tasks = [a.join() for a in agents]
        await asyncio.gather(*tasks, return_exceptions=True)



async def main():
    parser = argparse.ArgumentParser(description='Run Spade LLM Platform')
    parser.add_argument('config_file', help='Path to the YAML configuration file')
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        raw_config = yaml.safe_load(f)

    config = PlatformConfiguration(**raw_config)
    boot = Boot(config)
    await boot.run()

if __name__ == "__main__":
    asyncio.run(main())
