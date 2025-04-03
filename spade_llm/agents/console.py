from aioconsole import ainput
from pydantic import BaseModel, Field

from spade_llm.core.agent import Agent
from spade_llm.core.api import Message, AgentId, AgentContext
from spade_llm.core.behaviors import ContextBehaviour
from spade_llm.core.conf import configuration, Configurable


class ConsoleAgentConf(BaseModel):
    prompt: str = Field(default="User input:", description="Prompt to show in console when asking for input.")
    delegate_type: str = Field(description="Type of the delegate agent to send messages to.")
    delegate_id: str = Field(default="default", description="Type of the delegate agent to send messages to.")
    stop_words: set[str] = Field(default={"exit"}, description="Stop words used to shut down the agent")

@configuration(ConsoleAgentConf)
class ConsoleAgent(Agent, Configurable[ConsoleAgentConf]):

    class InputBehavior(ContextBehaviour):
        def __init__(self, context: AgentContext, config: ConsoleAgentConf):
            super().__init__(context)
            self.config = config

        async def step(self):
            user_input = await ainput(self.config.prompt)
            if user_input.lower() in self.config.stop_words:
                self.agent.stop()
                self.set_is_done()
            else:
                # Construct a message and send it to the agent
                message = Message(
                    sender=AgentId(agent_type=self.agent.agent_type, agent_id=self.context.agent_id),
                    receiver=AgentId(agent_type=self.config.delegate_type, agent_id=self.config.delegate_id),
                    performative="inform",
                    content=user_input)

                await self.context.send(message)


    def setup(self):
        self.add_behaviour(self.InputBehavior(self.default_context, self.config))