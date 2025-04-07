from asyncio import sleep as asleep

from aioconsole import ainput
from pydantic import BaseModel, Field

from spade_llm import consts
from spade_llm.core.agent import Agent
from spade_llm.core.api import AgentId, AgentContext
from spade_llm.core.behaviors import ContextBehaviour, MessageTemplate
from spade_llm.core.conf import configuration, Configurable
from spade_llm.core.tools import SingleMessage


class ConsoleAgentConf(BaseModel):
    prompt: str = Field(default="User input: ", description="Prompt to show in console when asking for input.")
    delegate_type: str = Field(description="Type of the delegate agent to send messages to.")
    delegate_id: str = Field(default="default", description="Type of the delegate agent to send messages to.")
    stop_words: set[str] = Field(default={"exit"}, description="Stop words used to shut down the agent")
    timeout: float = Field(default=60.0, description="Timeout for receiving messages.")

class HandleRequestBehavior(ContextBehaviour):

    def __init__(self, context: AgentContext, config: ConsoleAgentConf, request: str):
        super().__init__(context)
        self.config = config
        self.request = request

    async def step(self):
        reply = await self.receive(MessageTemplate(thread_id=self.context.thread_id), self.config.timeout)

        if not reply:
            print(f"No response received in {self.config.timeout} seconds")
            return

        if reply.performative == consts.REQUEST_APPROVAL:
            action = self.extract_message(reply)
            response = await ainput(f"Do you want to approve the action '{action}'? [y/n] ")
            if response.lower() == "y":
                await (self.context.reply_with_acknowledge(reply).with_content(""))
            else:
                await (self.context.reply_with_refuse(reply).with_content(""))
            return

        print(f"Assistant: {self.extract_message(reply)}.")
        if reply.performative == consts.REQUEST:
            response = await ainput(self.config.prompt)
            await (self.context.reply_with_inform(reply).with_content(response))
            return

        self.set_is_done()

    def extract_message(self, reply):
        if reply.content.startswith("{"):
            try:
                return SingleMessage.model_validate_json(reply.content).message
            except Exception as e:
                # Do nothing, just return the content
                pass
        return reply.content


class InputBehavior(ContextBehaviour):
    def __init__(self, context: AgentContext, config: ConsoleAgentConf):
        super().__init__(context)
        self.config = config

    async def step(self):
        # Small hack to let all the logs to be printed before prompting
        await asleep(0.5)
        user_input = await ainput(self.config.prompt)
        if user_input.lower() in self.config.stop_words:
            self.agent.stop()
            self.set_is_done()
        else:
            thread = await self.context.fork_thread()
            await (thread
                   .request(AgentId(agent_type=self.config.delegate_type, agent_id=self.config.delegate_id))
                   .with_content(user_input))

            handler = HandleRequestBehavior(thread, self.config, user_input)
            self.agent.add_behaviour(handler)

            await handler.join()
            await thread.close()

@configuration(ConsoleAgentConf)
class ConsoleAgent(Agent, Configurable[ConsoleAgentConf]):

    def setup(self):
        self.add_behaviour(InputBehavior(self.default_context, self.config))