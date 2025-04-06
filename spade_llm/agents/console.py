from aioconsole import ainput
from pydantic import BaseModel, Field

from spade_llm.core.agent import Agent
from spade_llm.core.api import AgentId, AgentContext
from spade_llm.core.behaviors import ContextBehaviour, MessageTemplate, MessageHandlingBehavior
from spade_llm.core.conf import configuration, Configurable


class ConsoleAgentConf(BaseModel):
    prompt: str = Field(default="User input: ", description="Prompt to show in console when asking for input.")
    delegate_type: str = Field(description="Type of the delegate agent to send messages to.")
    delegate_id: str = Field(default="default", description="Type of the delegate agent to send messages to.")
    stop_words: set[str] = Field(default={"exit"}, description="Stop words used to shut down the agent")
    timeout: float = Field(default=30.0, description="Timeout for receiving messages.")

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
                thread = await self.context.fork_thread()
                await (thread
                        .request(AgentId(agent_type=self.config.delegate_type, agent_id=self.config.delegate_id))
                        .with_content(user_input))

                reply = await self.receive(MessageTemplate(thread_id=thread.thread_id), self.config.timeout)
                if reply:
                    print(reply.content)
                else:
                    print(f"No response received in {self.config.timeout} seconds")

                await thread.close()

    class ApproveAction(MessageHandlingBehavior):

        async def step(self):
            action = self.message.content
            response = await ainput(f"Do you want to approve the action '{action}'? [y/n] ")
            if response.lower() == "y":
                await (self.context.reply_with_acknowledge(self.message).with_content(""))
            else:
                await (self.context.reply_with_refuse(self.message).with_content(""))


    def setup(self):
        self.add_behaviour(self.InputBehavior(self.default_context, self.config))
        self.add_behaviour(self.ApproveAction(MessageTemplate.request_approval()))