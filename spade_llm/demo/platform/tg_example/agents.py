from spade_llm.core.agent import Agent
from spade_llm.core.api import AgentId, AgentContext
from spade_llm.core.tools import SingleMessage
from spade_llm.core.behaviors import MessageTemplate, ContextBehaviour
from spade_llm.core.conf import configuration, Configurable, EmptyConfig
from spade_llm.demo.platform.tg_example.bot import TelegramBot
from spade_llm.core.models import CredentialsUtils, ChatModelFactory
from spade_llm import consts
from asyncio import sleep as asleep
from pydantic import BaseModel, Field


class TelegramAgentConf(BaseModel):
    prompt: str = Field(default="Assistant is waiting for the answer.",
                        description="Prompt to send in chat with user when asking for input.")
    delegate_type: str = Field(description="Type of the delegate agent to send messages to.")
    delegate_id: str = Field(default="default_user", description="Telegram username")
    timeout: float = Field(default=60.0, description="Timeout for receiving messages.")
    tg_bot: str = Field(description="Bot to use for handling messages.")


class TgRequestBehavior(ContextBehaviour):
    def __init__(self, context: AgentContext, config: TelegramAgentConf, request: str, bot: TelegramBot, chat_id: int):
        super().__init__(context)
        self.config = config
        self.request = request
        self.my_bot = bot
        self.request_chat_id = chat_id

    async def step(self):
        reply = await self.receive(MessageTemplate(thread_id=self.context.thread_id), self.config.timeout)

        if not reply:
            self.logger.info(f"No response received in {self.config.timeout} seconds")
            return

        if reply.performative == consts.REQUEST_APPROVAL:
            action = self.extract_message(reply)
            await self.my_bot.bot_reply(self.request_chat_id,
                                        reply_text=f"Do you want to approve the action '{action}'? [y/n] ")

            response = await self.my_bot.get_last_message()
            if response.text.lower() == "y":
                await (self.context.reply_with_acknowledge(reply).with_content(""))
            else:
                await (self.context.reply_with_refuse(reply).with_content(""))

            self.set_is_done()
            return

        msg = "Assistant: " + self.extract_message(reply)
        await self.my_bot.bot_reply(self.request_chat_id, reply_text=msg)

        if reply.performative == consts.REQUEST:
            await self.my_bot.bot_reply(self.request_chat_id, reply_text=self.config.prompt)

            response = await self.my_bot.get_last_message()
            await (self.context.reply_with_inform(reply).with_content(response))

            self.set_is_done()
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


class TgChatBehavior(ContextBehaviour):
    def __init__(self, context: AgentContext, config: TelegramAgentConf, bot: TelegramBot):
        super().__init__(context)
        self.config = config
        self.my_bot = bot

    async def step(self):
        # Small hack to let all the logs to be printed before prompting
        await asleep(0.5)

        user_input = await self.my_bot.get_last_message()

        self.logger.info(f"Processing input: {user_input.text}")
        if user_input is None or user_input.text is None:
            self.logger.warning('Input is none')
        else:
            thread = await self.context.fork_thread()
            self.config.delegate_id = '@' + user_input.chat.username
            await (thread
                   .request(AgentId(agent_type=self.config.delegate_type, agent_id=self.config.delegate_id))
                   .with_content(user_input.text))

            handler = TgRequestBehavior(thread, self.config, user_input.text, self.my_bot, user_input.chat.id)
            self.agent.add_behaviour(handler)

            await handler.join()
            await thread.close()


@configuration(TelegramAgentConf)
class Bot(Agent, Configurable[TelegramAgentConf]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self):
        # Создаем бота
        bot = self.default_context.create_chat_model(self.config.tg_bot)
        self.add_behaviour(TgChatBehavior(self.default_context, self.config, bot))


@configuration(EmptyConfig)
class TelegramBotFactory(ChatModelFactory[TelegramBot], Configurable[EmptyConfig]):
    def create_model(self) -> TelegramBot:
        TOKEN = CredentialsUtils.inject_env("env.BOT_TOKEN")
        return TelegramBot(TOKEN)
