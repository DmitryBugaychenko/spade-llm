from spade_llm.core.agent import Agent
from spade_llm.core.api import AgentId, AgentContext
from spade_llm.core.tools import SingleMessage
from spade_llm.core.behaviors import MessageTemplate, ContextBehaviour
from spade_llm.core.conf import configuration, Configurable
from spade_llm.demo.platform.tg_example.bot import TelegramBot
from spade_llm.core.models import CredentialsUtils
from asyncio import sleep as asleep
from pydantic import BaseModel, Field


class TelegramAgentConf(BaseModel):
    allowed_chat_ids: list[int] = Field(default_factory=list, description="List of allowed chat IDs")
    delegate_type: str = Field(description="Type of the delegate agent to send messages to.")
    delegate_id: str = Field(default="default", description="ID of the delegate agent to send messages to.")
    stop_words: set[str] = Field(default={"exit", "stop"}, description="Stop words used to shut down the agent")
    timeout: float = Field(default=60.0, description="Timeout for receiving messages.")


class TgRequestBehavior(ContextBehaviour):
    def __init__(self, context: AgentContext, config: TelegramAgentConf, request: str, bot: TelegramBot):
        super().__init__(context)
        self.config = config
        self.request = request
        self.my_bot = bot

    async def step(self):
        reply = await self.receive(MessageTemplate(thread_id=self.context.thread_id), self.config.timeout)

        if not reply:
            self.logger.info(f"No response received in {self.config.timeout} seconds")
            return
        msg = "Assistant: " + self.extract_message(reply)
        await self.my_bot.bot_reply(reply_text=msg)
        self.set_is_done()

    def extract_message(self, reply):
        if reply.content.startswith("{"):
            try:
                return SingleMessage.model_validate_json(reply.content).message
            except Exception as e:
                # Do nothing, just return the content
                pass
        return reply.content


class TgChatBehavior(ContextBehaviour):
    def __init__(self, context: AgentContext, config: TelegramAgentConf):
        super().__init__(context)
        self.config = config
        TOKEN = CredentialsUtils.inject_env("env.BOT_TOKEN")
        self.my_bot = TelegramBot(TOKEN)

    async def step(self):
        # Small hack to let all the logs to be printed before prompting
        await asleep(0.5)

        user_input = await self.my_bot.get_last_message()
        self.logger.info(f"Processing input: {user_input}")
        if user_input is None:
            self.logger.warning('Input is none')
        elif user_input.lower() in self.config.stop_words:
            self.agent.stop()
            self.set_is_done()
        else:

            thread = await self.context.fork_thread()

            await (thread
                   .request(AgentId(agent_type=self.config.delegate_type, agent_id=self.config.delegate_id))
                   .with_content(user_input))

            handler = TgRequestBehavior(thread, self.config, user_input, self.my_bot)
            self.agent.add_behaviour(handler)

            await handler.join()
            await thread.close()


@configuration(TelegramAgentConf)
class Bot(Agent, Configurable[TelegramAgentConf]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self):
        self.add_behaviour(TgChatBehavior(self.default_context, self.config))
